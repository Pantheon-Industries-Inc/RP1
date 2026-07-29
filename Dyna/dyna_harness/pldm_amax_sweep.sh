#!/bin/bash
# PLDM amax mini-sweep -- the pre-registered follow-up to the take-2 card.
# amax 1.6 (tuned on LeWM) gave PLDM-LIP 69.8 vs TD+CEM 73.3 with seed spread
# 9.3; the July PLDM campaign preferred amax ~3.5 under the old recipe. Arms:
# amax 2.6 and 3.5, canonical v4 recipe otherwise, same PLDM caches/TD as
# take-2. References (latent+CEM 66.0, TD+CEM 73.3, LIP@1.6 69.8) already in
# the CSV. Waits for the s12k loop to release the box.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_dynasplit.csv
DRV=$L/driver_pldm_amax.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; SEEDS="0 1 2"; DRAWS="42 43 44"
ARMS="2.6 3.5"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== PLDM amax sweep (pid $$): waiting for the s12k loop ==="
T0=$(date +%s)
until grep -q "S12K_LOOP_DONE" "$L/driver_s12kloop.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 28800 ] && die "s12k loop never finished in 8h"
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 120; done
log "box quiet"
for f in "$QF1" "$QF5" "$QTD" "$PLDM/weights.pt" "$PLDM/config.json" "$AH5"; do
  [ -e "$f" ] || die "missing $f (take-2 assets expected)"
done
[ -d "$EXPERT" ] || die "missing expert"
log "P0: preflight OK"

for A in $ARMS; do
  tag="a$(echo "$A" | tr -d .)"
  log "P1: LIP wave amax $A (GPUs 0-2)"
  g=0; for s in $SEEDS; do
    out=/workspace/actors/lip4_pldm_${tag}_s${s}.pt
    if [ ! -f "$out" ]; then
      CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
        --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
        --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$A" \
        --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
        --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$s" \
        --out "$out" --out-value "${out%.pt}_value.pt" \
        > "$L/pldm_lip_${tag}_s${s}.log" 2>&1 || log "  $tag/s$s TRAIN FAILED" &
    fi
    g=$((g+1))
  done; wait
  log "P1: $tag actors done"
done

log "P2: eval queue (18 cells, sequential, egl, held-out $EVAL_RANGE)"
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" >/dev/null; do sleep 60; done
for A in $ARMS; do
  tag="a$(echo "$A" | tr -d .)"
  for s in $SEEDS; do for d in $DRAWS; do
    nm="pldm_${tag}_s${s}_e${d}"
    c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; continue; }
    actor=/workspace/actors/lip4_pldm_${tag}_s${s}.pt
    [ -f "$actor" ] || { log "  $nm: actor missing, FAIL"; echo "${nm},FAIL" >> "$SUM"; continue; }
    CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
      seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
      policy="$PLDM" solver=lip "solver.actor_path=$actor" \
      output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
    grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" \
      || { log "  $nm: ep_range NOT APPLIED"; echo "${nm},FAIL" >> "$SUM"; continue; }
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
    echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
  done; done
done

python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
def sm(arm, s):
    vs = [rows.get(f"{arm}_s{s}_e{d}") for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else sum(vs) / 3
print("=== PLDM AMAX CARD (frozen PLDM, held-out 8000:10000, egl) ===")
for arm, label in (("pldm_lip", "LIP amax 1.6"), ("pldm_a26", "LIP amax 2.6"),
                   ("pldm_a35", "LIP amax 3.5")):
    ms = [sm(arm, s) for s in (0, 1, 2)]
    if all(m is not None for m in ms):
        mean = sum(ms)/3
        spread = max(ms) - min(ms)
        print(f"  {label:14s} {' '.join(f'{m:5.1f}' for m in ms)}   mean {mean:.1f}  spread {spread:.1f}")
print("  refs: latent+CEM 66.0, TD+CEM 73.3")
print("  reading: best amax beats 73.3 => LIP > CEM on PLDM too, at ~450x less compute;")
print("  best <= 73.3 => on a weak base the TD teacher is the better planner target.")
print("PLDM_AMAX_DONE")
PY
log "done"
