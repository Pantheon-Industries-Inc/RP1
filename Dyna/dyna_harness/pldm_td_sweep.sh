#!/bin/bash
# TD TEACHER SWEEP on PLDM, scored by TD+CEM success.
#
# WHY THIS IS THE RIGHT WAY TO SCREEN CRITIC KNOBS. The actor-side OFAT freezes
# the critic so every arm shares one teacher -- which makes critic knobs no-ops
# there by construction. They need their own screen, and there is a much better
# one available: evaluate the teacher DIRECTLY with search. TD+CEM needs no
# actor training at all, so each arm costs a ~2-minute TD train plus eval cells
# instead of a ~50-minute actor train. It also measures the critic on the thing
# we actually care about -- how well it ranks real rollouts -- rather than on
# its own training loss.
#
# WHY THESE KNOBS. --expectile 0.03 and --n-step 50 were both chosen on LeWM
# and inherited by PLDM unchanged. The campaign's standing constraint says the
# optimism level is LOAD-BEARING precisely because the planner queries V at
# WM-imagined off-manifold states (every value head that improved temporal
# ordering but removed optimism lost ~30 pts). How far off-manifold the
# imagination wanders is a property of the BASE -- so there is a direct
# mechanistic reason PLDM's optimum should differ, and nobody has ever checked.
#
# 3 SEEDS PER ARM. TD+CEM success is a 150-task measure with ~3-4 pts of noise,
# and this campaign has already been burned by a single lucky seed inventing a
# +2.0. Three teacher seeds per arm, 3 draws each, so arms are compared on
# 9-cell means. Still cheap: TD trains are ~2 min.
#
# The MRN quasimetric head is NOT swept -- standing constraint (IQE/QRL/
# Eikonal/rank-only were explored and rejected, each losing ~30 pts).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_tdsweep.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
QF1=/workspace/caches/pldm_tr8000_fs1.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
NSLOT=12; NGPU=8; SLOTDIR=/tmp/tdslots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do
    for s in $(seq 0 $((NSLOT-1))); do
      mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }
    done; sleep 3; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
rmdir "$SLOTDIR"/* 2>/dev/null || true

# tag|extra args for train_metric.py   (base = the inherited LeWM settings)
ARMS=(
  "base|"
  # --- optimism: the load-bearing knob, never re-tuned for a different base ---
  "tau001|--expectile 0.01" "tau010|--expectile 0.1"
  "tau030|--expectile 0.3"  "tau050|--expectile 0.5"
  # --- backup span ---
  "n10|--n-step 10" "n25|--n-step 25" "n100|--n-step 100"
  # --- discounting ---
  "g98|--gamma 0.98" "g995|--gamma 0.995"
  # --- optimisation / capacity ---
  "s3k|--steps 3000" "s12k|--steps 12000"
  "h512|--hidden-dim 512" "d3|--depth 3" "emb256|--embed-dim 256"
  # --- goal sampling ---
  "pc0|--p-cross 0.0" "pc6|--p-cross 0.6" "tol02|--tol 0.02"
)

log "=== PLDM TD SWEEP (pid $$): ${#ARMS[@]} arms x 3 seeds, scored by TD+CEM ==="
[ -f "$QF1" ] || die "missing $QF1"
log "P0: waiting for the actor sweeps to finish (evals need the box)"
T0=$(date +%s)
until grep -q "PLDM_FACTORIAL_DONE" "$L/driver_factorial.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && { log "WARN: proceeding after 12h"; break; }
  sleep 180
done
while pgrep -f "train_lip_a[c]" >/dev/null; do sleep 60; done
log "P0: box available"

# ------------------------------------------------ P1 train the teachers
log "P1: TD teachers (~2 min each)"
for e in "${ARMS[@]}"; do
  t="${e%%|*}"; x="${e#*|}"
  for s in $SEEDS; do
    out=/workspace/metrics/tdsw_${t}_s${s}.pt
    [ -f "$out" ] && continue
    slot=$(acquire); gpu=$(( slot % NGPU ))
    ( # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$P/train_metric.py" \
        --cache "$QF1" --learner td --head quasimetric \
        --expectile 0.03 --n-step 50 --gamma 1.0 --steps 6000 \
        --seed "$s" $x --out "$out" > "$L/tdsw_${t}_s${s}.log" 2>&1
      release "$slot" ) &
  done
done
wait
log "P1: $(ls /workspace/metrics/tdsw_*.pt 2>/dev/null | wc -l) teachers trained"

# --------------------------------------------- P2 score each with TD+CEM
ev(){ # slot tag seed draw
  local slot=$1 t=$2 s=$3 d=$4 gpu=$(( $1 % NGPU )) nm="tdcem_${2}_s${3}_e${4}"
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { release "$slot"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$PLDM" solver=cem "+metric=/workspace/metrics/tdsw_${t}_s${s}.pt" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
  release "$slot"
}
log "P2: TD+CEM scoring, ${#ARMS[@]}x3x3 cells"
for e in "${ARMS[@]}"; do
  t="${e%%|*}"
  for s in $SEEDS; do for d in $DRAWS; do
    [ -f "/workspace/metrics/tdsw_${t}_s${s}.pt" ] || continue
    slot=$(acquire); ev "$slot" "$t" "$s" "$d" &
  done; done
done
wait
log "P2: scoring done"

# ------------------------------------------------------------------ card
ARMS_STR=$(printf '%s ' "${ARMS[@]}")
ARMS_STR="$ARMS_STR" python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import os, sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
def arm(t):
    per = []
    for s in (0,1,2):
        vs = [rows.get(f"tdcem_{t}_s{s}_e{d}") for d in (42,43,44)]
        if all(v is not None for v in vs): per.append(mean(vs))
    return per
tags = [e.split("|")[0] for e in os.environ["ARMS_STR"].split()]
res = {}
for t in tags:
    p = arm(t)
    if len(p) >= 2: res[t] = p
base = res.get("base")
print("=== PLDM TD-TEACHER SWEEP, scored by TD+CEM (held-out, EGL) ===")
print(f"  reference: inherited-LeWM TD+CEM 73.3, latent+CEM 66.7, LIP(20 seeds) 72.8")
print(f"  {'arm':8s} {'mean':>6s} {'sd':>5s} {'n':>3s} {'vs base':>9s}")
for t, p in sorted(res.items(), key=lambda kv: -mean(kv[1])):
    m = mean(p); sd = math.sqrt(sum((x-m)**2 for x in p)/max(len(p)-1,1))
    d = f"{m-mean(base):+.1f}" if base else "?"
    star = ""
    if base and len(p) >= 2 and len(base) >= 2:
        se = math.sqrt(sd**2/len(p) + (math.sqrt(sum((x-mean(base))**2 for x in base)/max(len(base)-1,1))**2)/len(base))
        if se > 0 and abs(m-mean(base))/se > 2: star = "  <<<"
    print(f"  {t:8s} {m:6.1f} {sd:5.2f} {len(p):3d} {d:>9s}{star}")
print("  <<< marks arms more than 2 SE from base -- a screen, not a significance")
print("  test. A winner here matters twice over: it lifts the TD+CEM bar itself,")
print("  AND it is the teacher LIP warm-starts from, so it should propagate.")
print("PLDM_TDSWEEP_DONE")
PY
log "done"
