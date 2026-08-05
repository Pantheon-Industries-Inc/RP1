#!/bin/bash
# FROZEN CRITIC, THEN ACTOR-ONLY OPTIMIZATION.
#
# WHY. E_final plateaus by step ~1000 and sits flat for 5000 more. But in the
# default recipe the critic is co-trained the whole time (td_loss 2.37 -> 0.94
# between steps 1000 and 4500) AND the expectile anneals (tau 0.085 -> 0.034),
# which changes the objective the critic fits. So E_final = teacher(z_T, z_g)
# is a number whose UNITS DRIFT while the actor is scored on it: a flat trace
# is equally consistent with 'actor stopped learning' and 'actor improved
# while the teacher tightened by the same amount'. As instrumented, the
# plateau is not interpretable.
#
# Freezing the critic almost immediately (0.05 * steps = 300 warmup steps on
# top of the pretrained TD teacher) makes the teacher STATIC. E_final then
# becomes a proper learning curve, directly comparable across steps, and the
# question becomes decidable:
#   keeps descending past step 1000 -> the plateau was the moving yardstick
#   still flat                      -> the actor genuinely saturates, and the
#                                      binding limit is the 8-iteration budget
#                                      (which the optcap sweep tests)
#
# Two arms so the answer is not confounded with training length:
#   frz    freeze 0.05, steps 6000   -- like-for-like against the base run
#   frzL   freeze 0.05, steps 15000  -- 2.5x actor steps against a static
#                                       teacher; if it is going to improve at
#                                       all, this is where it shows
#
# Runs on GPUs 6-7, which the optimization-capacity sweep leaves idle.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_frzcrit.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"
AMAX=4.5; MW=0.1; ALR=3e-4; ALRF=3e-5
mkdir -p "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== FROZEN-CRITIC ACTOR OPTIMIZATION (pid $$) ==="
for f in "$QF1" "$QF5" "$QTD" "$AH5" "$PLDM/weights.pt"; do [ -e "$f" ] || die "missing $f"; done

run(){ # gpu tag steps
  local gpu=$1 t=$2 st=$3
  local out=/workspace/actors/lip4_pldm_${t}_s0.pt
  [ -f "$out" ] && { log "  $t present"; return 0; }
  log "  $t: freeze at 5% of $st steps, gpu $gpu"
  CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$P/train_lip_ac.py" \
    --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
    --horizon 5 --iters 8 --n-step 50 --amax "$AMAX" --mean-weight "$MW" \
    --actor-lr "$ALR" --actor-lr-final "$ALRF" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --arch v4 --seed 0 --steps "$st" --freeze-critic-frac 0.05 \
    --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/frz_${t}_s0.log" 2>&1
}
log "P1: two arms on GPUs 6-7"
run 6 frz  6000  &
run 7 frzL 15000 &
wait
log "P1: trains done"

# ------------------------------------------------- P2 the learning curves
log "P2: E_final learning curves (static teacher => comparable across steps)"
for t in frz frzL; do
  f="$L/frz_${t}_s0.log"; [ -f "$f" ] || continue
  log "  --- $t ---"
  grep -E "^step .*E_final" "$f" | awk '{print $2, $4, $6}' | \
    awk 'NR==1||NR%4==1{printf "    step %-6s E_final %-8s E_first %s\n",$1,$2,$3}' | tee -a "$DRV"
done
log "  --- base (critic live to step 4800, for contrast) ---"
grep -E "^step .*E_final" "$L/grid_train_mw01_a45_lr3e-4_s0.log" 2>/dev/null | \
  awk '{print $2, $4, $6}' | awk 'NR==1||NR%4==1{printf "    step %-6s E_final %-8s E_first %s\n",$1,$2,$3}' | tee -a "$DRV"

# --------------------------------------------------------------- P3 evals
ev(){ # gpu tag draw
  local gpu=$1 t=$2 d=$3 nm="frz_${t}_s0_e${d}"
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$PLDM" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_pldm_${t}_s0.pt" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
}
log "P3: evals (waiting for a quiet box)"
while pgrep -f "train_lip_a[c]|train_metri[c]" >/dev/null; do sleep 60; done
g=0
for t in frz frzL; do for d in $DRAWS; do
  ev "$g" "$t" "$d" & g=$((g+1))
done; done
wait

python3 - "$SUM" "$L" 2>&1 <<'PY' | tee -a "$DRV"
import sys, re
SUM, L = sys.argv[1], sys.argv[2]
rows = {}
for ln in open(SUM):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
def curve(p):
    try: t = open(p).read()
    except OSError: return []
    return [(int(s), float(e)) for s, e in
            re.findall(r"^step (\d+): E_final ([0-9.]+)", t, re.M)]
def m3(p):
    vs = [rows.get(f"{p}_e{d}") for d in (42,43,44)]
    return None if any(v is None for v in vs) else sum(vs)/3
print("=== FROZEN-CRITIC CARD (PLDM, held-out, EGL, seed 0) ===")
for tag, path, note in (
        ("base", f"{L}/grid_train_mw01_a45_lr3e-4_s0.log", "critic live to 4800"),
        ("frz",  f"{L}/frz_frz_s0.log",  "critic frozen at 300, 6000 steps"),
        ("frzL", f"{L}/frz_frzL_s0.log", "critic frozen at 750, 15000 steps")):
    c = curve(path)
    if not c: continue
    e1k = next((e for s, e in c if s >= 1000), None)
    elast = c[-1][1]
    drop = f"{100*(elast-e1k)/e1k:+.1f}%" if e1k else "?"
    sv = m3(f"frz_{tag}_s0") if tag != "base" else None
    print(f"  {tag:5s} ({note}): E_final@1k {e1k:.2f} -> final {elast:.2f}  "
          f"[{drop} after step 1000]" + (f"  success {sv:.1f}" if sv else ""))
print("  reading: base is NOT interpretable (drifting teacher). For frz/frzL the")
print("  teacher is static, so a further drop after step 1000 means the actor")
print("  was still learning and the plateau was an artifact; flat means the")
print("  actor really saturates and the 8-iteration budget is the binding limit.")
print("PLDM_FRZCRIT_DONE")
PY
log "done"
