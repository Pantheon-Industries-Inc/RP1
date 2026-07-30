#!/bin/bash
# h100 horizon condition on the idle 4xH100 pod: goal offset 100 / budget 200.
# Same three window arms, same hypers, same trained artifacts, same metric
# (latched@0.05) and 1-frame policy conditioning. NOT a paper cell -- the paper's
# reacher protocol is h25 only; h50/h100 are our horizon extension and must be
# reported separately, never against 86/78/79.
#
# Feasibility: episodes are 201 rows, so offset 100 leaves starts 0..100;
# horizon*action_block = 25 <= budget 200 (eval_wm asserts this).
#
# Reference at shorter horizons (lejepa, n=6):
#   h25  LIP-w 90.6 pooled | Latent+CEM-w 89.7 | TD+CEM-w 86.7
#   h50  LIP-w 97.9 pooled | Latent+CEM-w 92.3 | TD+CEM-w 98.0
# One EGL process per GPU; four independent streams.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
SUM=/workspace/results/summary_h100w3_lejepa.csv; touch "$SUM"
L=/workspace/logs/h100w3; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][h100] $*"; }

ev(){ local gpu=$1 nm=$2 seed=$3; shift 3
  grep -q "^${nm}," "$SUM" && { log "  ${nm}: cached"; return 0; }
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 14400 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=100 eval.eval_budget=200 \
    solver.batch_size=10 "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local held ever hist
  held=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  ever=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  hist=$(grep -c "hist-inject" "$L/${nm}.log")
  echo "${nm},latched=${ever:-FAIL},held=${held:-FAIL},hist=${hist}" >> "$SUM"
  log "  ${nm}: latched ${ever:-FAIL} | held ${held:-FAIL} (hist=${hist} must be 0)"
}
card6(){ local pre=$1 tag=$2; local t=0 n=0 e
  for s in 42 43 44 45 46 47; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "latched=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  [ "$n" -gt 0 ] && log "CARD6 ${tag}: latched h100 $(awk "BEGIN{printf \"%.1f\", $t/$n}") (n=$n)"
}

( for s in 42 43 44 45 46 47; do
    ev 0 "h100w3l2_s${s}" $s solver=cem solver.n_steps=10 "+metric=/workspace/metrics/l2window3.pt"
  done; card6 "h100w3l2_s" "Latent+CEM window [h25 89.7 | h50 92.3]" ) &
( for s in 42 43 44 45 46 47; do
    ev 1 "h100w3td_s${s}" $s solver=cem solver.n_steps=10 "+metric=/workspace/metrics/window3_lejepa.pt"
  done; card6 "h100w3td_s" "TD+CEM window [h25 86.7 | h50 98.0]" ) &
( for a in 0 1; do
    A=/workspace/actors/lip4_w3_lejepa_s${a}.pt; [ -f "$A" ] || continue
    for s in 42 43 44 45 46 47; do
      ev 2 "h100w3lip${a}_s${s}" $s solver=lip solver.actor_path="$A" solver.rollout_compat=false
    done; card6 "h100w3lip${a}_s" "LIP window s${a} plain"
  done ) &
( A=/workspace/actors/lip4_w3_lejepa_s2.pt
  if [ -f "$A" ]; then
    for s in 42 43 44 45 46 47; do
      ev 3 "h100w3lip2_s${s}" $s solver=lip solver.actor_path="$A" solver.rollout_compat=false
    done; card6 "h100w3lip2_s" "LIP window s2 plain"
  fi ) &
wait
log "H100W3_DONE"
