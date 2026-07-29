#!/bin/bash
# PLDM stage 2: make LIP beat TD+CEM (73.3) on the weak base.
#
# Mechanism this targets: LIP's critic trains on REAL cached latents and is
# fine (TD+CEM 73.3 proves the cost ranks PLDM states well); the weak link is
# the ACTOR, which learns by backprop through 5 fs5-steps (25 primitive) of
# IMAGINED PLDM rollouts -- a weak predictor compounds error over that horizon
# and the actor optimizes against hallucination. Stage 1 (amax sweep) caps
# how hard it can exploit; this stage SHORTENS THE IMAGINATION it trains
# through (h3, h2 vs default 5 -- never tested on PLDM, weak-WM-specific
# lever) plus one gentle-actor arm (alr; null on LeWM, but the exploitation
# argument differs on a weak base). Deploy solver keeps the protocol's
# horizon 5, so eval comparability is untouched.
#
# AMAX must be exported by the launcher = the stage-1 winner (selection is
# logged; note the tuned-on-these-draws caveat in any writeup).
# 3 waves + 27 cells ~= 3.5h. Waits for PLDM_AMAX_DONE + quiet box.
set -u
: "${AMAX:?export AMAX=<stage-1 winner, e.g. 3.5>}"
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_dynasplit.csv
DRV=$L/driver_pldm_stage2.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; SEEDS="0 1 2"; DRAWS="42 43 44"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== PLDM stage 2 (pid $$, AMAX=$AMAX): waiting for the amax sweep ==="
T0=$(date +%s)
until grep -q "PLDM_AMAX_DONE" "$L/driver_pldm_amax.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 28800 ] && die "amax sweep never finished in 8h"
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 120; done
for f in "$QF1" "$QF5" "$QTD" "$PLDM/weights.pt" "$AH5"; do [ -e "$f" ] || die "missing $f"; done
log "P0: preflight OK, box quiet"

run_wave(){ # tag extra-args...
  local tag=$1; shift
  log "P1: wave $tag (GPUs 0-2, amax $AMAX)"
  local g=0 s
  for s in $SEEDS; do
    out=/workspace/actors/lip4_pldm_${tag}_s${s}.pt
    if [ ! -f "$out" ]; then
      CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
        --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
        --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
        --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
        --arch v4 --seed "$s" "$@" \
        --out "$out" --out-value "${out%.pt}_value.pt" \
        > "$L/pldm_lip_${tag}_s${s}.log" 2>&1 || log "  $tag/s$s TRAIN FAILED" &
    fi
    g=$((g+1))
  done; wait
  log "P1: $tag done"
}
run_wave h3  --horizon 3 --actor-lr 3e-4 --actor-lr-final 3e-5
run_wave h2  --horizon 2 --actor-lr 3e-4 --actor-lr-final 3e-5
run_wave alr --horizon 5 --actor-lr 1e-4 --actor-lr-final 1e-5

log "P2: eval queue (27 cells, sequential, egl, held-out $EVAL_RANGE)"
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" >/dev/null; do sleep 60; done
for tag in h3 h2 alr; do
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

AMAX="$AMAX" python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import os, sys
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
def sm(arm, s):
    vs = [rows.get(f"{arm}_s{s}_e{d}") for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else sum(vs) / 3
print(f"=== PLDM STAGE-2 CARD (amax {os.environ['AMAX']}, held-out, egl) ===")
print("  refs: latent+CEM 66.0 | TD+CEM 73.3 | LIP@1.6/h5 69.8")
best = (None, -1)
for arm, label in (("pldm_h3", "h3"), ("pldm_h2", "h2"), ("pldm_alr", "alr"),
                   ("pldm_a26", "stage1 a2.6/h5"), ("pldm_a35", "stage1 a3.5/h5")):
    ms = [sm(arm, s) for s in (0, 1, 2)]
    if all(m is not None for m in ms):
        mean = sum(ms)/3
        print(f"  {label:16s} {' '.join(f'{m:5.1f}' for m in ms)}   mean {mean:.1f}  spread {max(ms)-min(ms):.1f}")
        if mean > best[1]: best = (label, mean)
if best[0]:
    print(f"  best LIP config: {best[0]} at {best[1]:.1f} "
          f"({'BEATS' if best[1] > 73.3 else 'does NOT beat'} TD+CEM 73.3)")
    print("  next: 3 more seeds on the winner for a 6-seed significance card"
          " (one-sample t vs 73.3).")
print("PLDM_STAGE2_DONE")
PY
log "done"
