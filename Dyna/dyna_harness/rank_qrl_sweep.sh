#!/bin/bash
# 1) rank-only  (td_weight=0)  -- control; math predicts scale degeneracy
# 2) qrl(iqe)                  -- local constraint + spreading, no far-pair labels
# 3) qrl(iqe)+rank             -- the tandem
# 4) qrl(mrn)+rank             -- head ablation of the tandem
# Reported per config: planner-relevant ranking (TEST B') + distance calibration
# (the latter also exposes scale blow-up, which is the degeneracy signature).
# Deployed TD baseline (TEST B'): 5v10 94.7 | 25v50 71.8 | 50v75 51.5 | 75v100 70.9 | 25v100 54.6
# Achievable ceiling:                          98.6 |      98.9 |       92.4
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel TQDM_DISABLE=1
P=/workspace/code/stable-worldmodel/scripts/plan
C=/workspace/caches/r2w_expert_fs1.pt
L=/workspace/logs; O=$L/rank_qrl_sweep.log
: > "$O"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$O"; }

run(){ # tag  <train args...>
  local T=$1; shift
  local PT=/workspace/metrics/vfix_${T}.pt
  log "=== $T ==="
  if [ ! -f "$PT" ]; then
    CUDA_VISIBLE_DEVICES=0 timeout 5400 python3 "$P/train_metric.py" --cache "$C" \
      --steps 6000 --seed 0 --out "$PT" "$@" > "$L/train_${T}.log" 2>&1 || {
        log "  TRAIN FAILED"; grep -iE "error|Traceback" -A3 "$L/train_${T}.log" | tail -6 | tee -a "$O"; return 1; }
  fi
  log "  $(grep -oE 'QRL trained.*|final loss=[0-9.]+' "$L/train_${T}.log" | tail -1)"
  CUDA_VISIBLE_DEVICES=0 python3 /workspace/ranking_ceiling_probe.py "$C" "$PT" 2>/dev/null \
    | sed -n '/TEST B/,/TEST C/p' | grep -E "closer" | tee -a "$O"
  log "  -- distance calibration (scale / monotonicity) --"
  CUDA_VISIBLE_DEVICES=0 python3 /workspace/value_probe.py "$C" "$PT" 2>/dev/null \
    | sed -n '/Distance calibration/,$p' | grep -E "^ *[0-9]" | tr '\n' ' ' | tee -a "$O"
  echo "" | tee -a "$O"
}

run rankonly   --learner td  --head quasimetric --expectile 0.03 --n-step 50 --gamma 1.0 --td-weight 0 --rank-weight 2.0 --rank-margin 0.5
run qrl_iqe    --learner qrl --head iqe
run qrl_iqe_rk --learner qrl --head iqe         --rank-weight 2.0 --rank-margin 0.5
run qrl_mrn_rk --learner qrl --head quasimetric --rank-weight 2.0 --rank-margin 0.5
log "RANK_QRL_SWEEP_DONE"
