#!/bin/bash
# PLDM OPTIMIZATION CAPACITY SWEEP -- can the actor drive the imagined cost
# lower, and does that help or hurt real success?
#
# WHAT WE ACTUALLY KNOW (two earlier readings were wrong and are retracted):
#   * E_final ~10.8 vs LeWM's 3.7-5.05 is NOT evidence of under-optimization.
#     The two numbers are in units of two differently-scaled learned
#     quasimetrics, so they are not comparable. LeWM's E_first is unrecoverable
#     (logs died with the volume), so the reduction RATIO cannot be compared
#     either.
#   * `td_loss nan` from step 5000 is NOT a divergence. train_lip_ac.py sets
#     cl = float("nan") every step and only assigns it while the critic is
#     live; freeze_at = int(0.8 * 6000) = 4800..5000. Cosmetic, by design.
#
# WHAT SURVIVES, from the logs:
#   * E_first ~21 -> E_final ~10.8: the 8 inner iterations halve the cost.
#   * E_first NEVER improves over training (21.7 / 22.5 / 20.5) -- the actor's
#     one-shot proposal does not get better; only refinement does.
#   * E_final plateaus by step ~1000 and sits there for 5000 more steps.
#
# WHY BOTH METRICS: E_final is NOT the objective. On LeWM, amax 3.5 + iters 16
# reached E_final 1.68 -- the lowest ever measured -- and planned WORSE
# (80.0 vs 87.8). Lower E can mean better plans or more exploitation of an
# imperfect model. Only E_final AND success together say which side of the
# peak a base sits on.
#
# DESIGN. Screen on E_final, which is a training-internal average over batches
# and therefore far lower-variance than a 150-task success rate -- important,
# because a single lucky seed just produced a phantom +2.0 in this campaign.
# Then spend eval cells on the arms that actually moved it.
#   base   iters 8,  steps 6000, freeze 0.8, critic_ratio 1   (E_final 10.8)
#   it16   iters 16          -- 2x inner refinement
#   it32   iters 32          -- 4x, deliberately into the exploitation regime
#   s12k   steps 12000       -- 2x actor training
#   nofrz  freeze 1.0        -- critic never frozen (default freezes at 0.8)
#   cr2    critic_ratio 2    -- 2 critic steps per actor step
#   combo  iters 16 + steps 12000
# 6 arms x 1 seed, one 8-GPU wave. Then 3 draws x the arms that moved E_final.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_optcap.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"
AMAX=4.5; MW=0.1; ALR=3e-4; ALRF=3e-5; NGPU=8
mkdir -p "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== PLDM OPT-CAPACITY SWEEP (pid $$) ==="
for f in "$QF1" "$QF5" "$QTD" "$AH5" "$PLDM/weights.pt"; do [ -e "$f" ] || die "missing $f"; done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do
  log "  waiting for the box to drain"; sleep 60; done
log "P0: box quiet"

# tag  ->  extra args
ARMS="it16 it32 s12k nofrz cr2 combo"
args_for(){ case "$1" in
  it16)  echo "--iters 16 --steps 6000  --freeze-critic-frac 0.8 --critic-ratio 1" ;;
  it32)  echo "--iters 32 --steps 6000  --freeze-critic-frac 0.8 --critic-ratio 1" ;;
  s12k)  echo "--iters 8  --steps 12000 --freeze-critic-frac 0.8 --critic-ratio 1" ;;
  nofrz) echo "--iters 8  --steps 6000  --freeze-critic-frac 1.0 --critic-ratio 1" ;;
  cr2)   echo "--iters 8  --steps 6000  --freeze-critic-frac 0.8 --critic-ratio 2" ;;
  combo) echo "--iters 16 --steps 12000 --freeze-critic-frac 0.8 --critic-ratio 1" ;;
esac; }

log "P1: training 6 arms, seed 0, one wave"
g=0
for t in $ARMS; do
  out=/workspace/actors/lip4_pldm_oc_${t}_s0.pt
  [ -f "$out" ] && { log "  $t present"; g=$((g+1)); continue; }
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$g timeout 43200 python3 "$P/train_lip_ac.py" \
    --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
    --horizon 5 --n-step 50 --amax "$AMAX" --mean-weight "$MW" \
    --actor-lr "$ALR" --actor-lr-final "$ALRF" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --arch v4 --seed 0 $(args_for "$t") \
    --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/optcap_${t}_s0.log" 2>&1 &
  g=$((g+1))
done
wait
log "P1: trains done"

# ---------------- E_final screen (low-variance, training-internal)
log "P2: E_final screen"
BASE_E=$(grep -oE "E_final [0-9.]+" "$L/grid_train_mw01_a45_lr3e-4_s0.log" 2>/dev/null | tail -1 | cut -d" " -f2)
log "  base (iters 8): E_final ${BASE_E:-10.831}"
MOVED=""
for t in $ARMS; do
  f="$L/optcap_${t}_s0.log"; [ -f "$f" ] || { log "  $t: no log"; continue; }
  ef=$(grep -oE "E_final [0-9.]+" "$f" | tail -1 | cut -d" " -f2)
  e1=$(grep -oE "E_first [0-9.]+" "$f" | tail -1 | cut -d" " -f2)
  log "  $t: E_final ${ef:-NA} E_first ${e1:-NA}"
  # keep any arm that moved E_final by >5% in either direction: a big DROP is
  # the capacity hypothesis, a big RISE is informative too (worse optimizer)
  [ -n "$ef" ] && awk -v a="$ef" -v b="${BASE_E:-10.831}" \
     'BEGIN{exit !(a < 0.95*b || a > 1.05*b)}' && MOVED="$MOVED $t"
done
log "P2: arms that moved E_final:${MOVED:- none}"
[ -n "$MOVED" ] || { log "no arm moved E_final -- capacity is NOT the binding constraint on PLDM"; MOVED="$ARMS"; log "evaluating all arms anyway for the record"; }

# ---------------------------------------------- P3 evals on the moved arms
ev(){ # gpu tag draw
  local gpu=$1 t=$2 d=$3 nm="oc_${t}_s0_e${d}"
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$PLDM" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_pldm_oc_${t}_s0.pt" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
}
log "P3: evals"
g=0
for t in $MOVED; do for d in $DRAWS; do
  ev "$g" "$t" "$d" & g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done
wait

# ------------------------------------------------------------------ P4 card
ARMS="$ARMS" python3 - "$SUM" "$L" 2>&1 <<'PY' | tee -a "$DRV"
import os, sys, re, glob
SUM, L = sys.argv[1], sys.argv[2]
rows = {}
for ln in open(SUM):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
def efin(path):
    try:
        t = open(path).read()
        f = re.findall(r"E_final ([0-9.]+)", t); s = re.findall(r"E_first ([0-9.]+)", t)
        return (float(f[-1]) if f else None), (float(s[-1]) if s else None)
    except OSError: return None, None
def m3(p):
    vs = [rows.get(f"{p}_e{d}") for d in (42,43,44)]
    return None if any(v is None for v in vs) else sum(vs)/3
print("=== PLDM OPT-CAPACITY CARD (frozen PLDM, held-out, EGL, seed 0) ===")
be, bs = efin(f"{L}/grid_train_mw01_a45_lr3e-4_s0.log")
print(f"  {'arm':8s} {'E_final':>8s} {'E_first':>8s} {'success':>8s}   reference: 20-seed LIP 72.8, TD+CEM 73.3")
print(f"  {'base':8s} {be if be else 0:8.2f} {bs if bs else 0:8.2f} {'75.1*':>8s}   (*n=3; 20-seed value is 72.8)")
for t in os.environ["ARMS"].split():
    ef, e1 = efin(f"{L}/optcap_{t}_s0.log")
    sv = m3(f"oc_{t}_s0")
    if ef is None: continue
    d = f"{100*(ef-be)/be:+.0f}%" if be else "?"
    print(f"  {t:8s} {ef:8.2f} {e1 if e1 else 0:8.2f} {(f'{sv:.1f}' if sv else '--'):>8s}   E_final {d}")
print("  reading: E_final DOWN + success UP  => capacity was the binding constraint")
print("           E_final DOWN + success DOWN=> exploitation, PLDM is past its peak")
print("           E_final FLAT              => capacity is not the lever at all")
print("  caveat: seed 0 only. Seed 0 produced this campaign's phantom +2.0, so any")
print("  success number here needs seeds before it is believed; E_final is the")
print("  low-variance screen and is what this card is really measuring.")
print("PLDM_OPTCAP_DONE")
PY
log "done"
