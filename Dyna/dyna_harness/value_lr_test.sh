#!/bin/bash
# Validate the long-horizon value fix: n-step=200 (exact MC labels full horizon,
# HER retained) vs the n-step=50 baseline. Probe the calibration curve for each.
# Pure latent compute (no MuJoCo/EGL) -> safe alongside other GPU work. GPU2.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel TQDM_DISABLE=1
CODE=/workspace/code/stable-worldmodel; PLAN=$CODE/scripts/plan
CACHE=/workspace/caches/r2w_expert_fs1.pt
LOGS=/workspace/logs; MET=/workspace/metrics
G=2
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

log "=== baseline n-step=50 (existing r2w_expert_TD.pt) ==="
CUDA_VISIBLE_DEVICES=$G python3 /workspace/value_probe.py "$CACHE" "$MET/r2w_expert_TD.pt" 2>/dev/null | grep -E "^\s*[0-9]|k |K |Approach|Distance|meanV|j/K"

for EXP in 0.03 0.5; do
  OUT=$MET/value_ns200_e${EXP/./}.pt
  log "=== train n-step=200 expectile=$EXP ==="
  CUDA_VISIBLE_DEVICES=$G python3 "$PLAN/train_metric.py" --cache "$CACHE" \
    --learner td --head quasimetric --n-step 200 --expectile "$EXP" \
    --gamma 1.0 --steps 6000 --seed 0 --out "$OUT" > "$LOGS/train_ns200_e${EXP/./}.log" 2>&1
  log "trained -> $OUT ($(grep -oE "final loss=[0-9.]+" $LOGS/train_ns200_e${EXP/./}.log | tail -1))"
  log "--- probe n-step=200 expectile=$EXP ---"
  CUDA_VISIBLE_DEVICES=$G python3 /workspace/value_probe.py "$CACHE" "$OUT" 2>/dev/null | grep -E "^\s*[0-9]|k |K |Approach|Distance|meanV|j/K"
done
log "VALUE_LR_TEST_DONE"
