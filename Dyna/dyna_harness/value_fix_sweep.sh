#!/bin/bash
# Value-fix sweep: {MRN, IQE} x {eikonal 0, 0.1, 1.0}, identical otherwise to the
# deployed recipe (td / expectile 0.03 / n-step 50 / 6k steps, r2w fs1 cache).
# Each config is probed with ordering_probe.py — the target is the 50v75 / 75v100
# ranking accuracy, which the deployed MRN value gets WRONG (40.3% / 38.1%).
# Deployed baseline: 5v10 94.2 | 25v50 69.4 | 50v75 40.3 | 75v100 38.1 | 150v200 94.1
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel TQDM_DISABLE=1
CODE=/workspace/code/stable-worldmodel; PLAN=$CODE/scripts/plan
CACHE=/workspace/caches/r2w_expert_fs1.pt
MET=/workspace/metrics; LOGS=/workspace/logs
OUT=$LOGS/value_fix_sweep.log
G=${1:-0}
: > "$OUT"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT"; }

for HEAD in quasimetric iqe; do
  for EIK in 0 0.1 1.0; do
    TAG="${HEAD}_eik${EIK/./}"
    PT=$MET/vfix_${TAG}.pt
    if [ ! -f "$PT" ]; then
      log "=== training $TAG (head=$HEAD eikonal=$EIK) ==="
      CUDA_VISIBLE_DEVICES=$G timeout 5400 python3 "$PLAN/train_metric.py" --cache "$CACHE" \
        --learner td --head "$HEAD" --expectile 0.03 --n-step 50 --gamma 1.0 \
        --steps 6000 --seed 0 --eikonal-weight "$EIK" --out "$PT" \
        > "$LOGS/vfix_train_${TAG}.log" 2>&1 || { log "$TAG TRAIN FAILED"; tail -3 "$LOGS/vfix_train_${TAG}.log" | tee -a "$OUT"; continue; }
      log "$TAG trained ($(grep -oE 'final loss=[0-9.]+' $LOGS/vfix_train_${TAG}.log | tail -1); $(grep -oE 'mean per-step latent displacement = [0-9.]+' $LOGS/vfix_train_${TAG}.log | tail -1))"
    else
      log "=== $TAG cached ==="
    fi
    log "--- ordering probe: $TAG ---"
    CUDA_VISIBLE_DEVICES=$G python3 /workspace/ordering_probe.py "$CACHE" "$PT" 2>/dev/null \
      | grep -E "vs|TEST A" | tee -a "$OUT"
    log "--- distance calibration: $TAG ---"
    CUDA_VISIBLE_DEVICES=$G python3 /workspace/value_probe.py "$CACHE" "$PT" 2>/dev/null \
      | sed -n '/Distance calibration/,$p' | tee -a "$OUT"
  done
done
log "VALUE_FIX_SWEEP_DONE"
