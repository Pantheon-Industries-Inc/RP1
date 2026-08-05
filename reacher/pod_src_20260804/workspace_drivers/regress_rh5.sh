#!/bin/bash
# Regression: the reported rh=5 protocol must be untouched by the I3 edits.
#
# Three source files changed (lip.py _V split, policy.py publish, eval_wm.py
# readout). Two of them sit on the path EVERY reported cell used, so "I3 is
# inert" has to be measured on the reported protocol itself, not argued from
# the guard condition. These cells run with NO deadline -- exactly the reported
# command line -- and must reproduce the committed per-cell numbers.
#
#   RLP  lejepa s0 seed 43   |  RLP  pldm s0 seed 42   |  Latent+CEM lejepa s42
#
# The third matters most: eval_wm.py's _MetricCost is where the goal-broadcast
# rewrite landed, and every sampling arm in the reported table goes through it.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
L=/workspace/logs/regress5; mkdir -p "$L"
L2=/workspace/metrics/l2window3.pt
log(){ echo "[$(date -u +%H:%M:%S)][regress] $*"; }

ev(){ local gpu=$1 B=$2 nm=$3 seed=$4; shift 4
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 5400 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy=/workspace/swm_home/checkpoints/${B}_reacher \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    plan_config.receding_horizon=5 "$@" output.filename=${nm}.txt \
    > "$L/${nm}.log" 2>&1
}

log "3 reported-protocol cells, no deadline, rh=5"
ev 0 lejepa r_rlp_lejepa_e43 43 solver=lip \
   solver.actor_path=/workspace/actors/lip4_leak6_lejepa_s0.pt \
   solver.rollout_compat=false &
ev 1 pldm r_rlp_pldm_e42 42 solver=lip \
   solver.actor_path=/workspace/actors/lip4_leak6_pldm_s0.pt \
   solver.rollout_compat=false &
wait
ev 2 lejepa r_l2cem_lejepa_e42 42 solver=cem solver.n_steps=10 "+metric=$L2" &
wait

log "================ REGRESSION ================"
for nm in r_rlp_lejepa_e43 r_rlp_pldm_e42 r_l2cem_lejepa_e42; do
  f="$L/${nm}.log"
  h=$(grep -oE "HELD-at-end [0-9.]+" "$f" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$f" | tail -1 | sed -E "s/0\.1rad //")
  printf "  %-22s @0.05=%-7s @0.1=%-7s I3lines=%s\n" "$nm" "${h:-FAIL}" "${t10:-FAIL}" \
    "$(grep -c '^\[I3\]' "$f")"
  grep -q Traceback "$f" && printf "      CRASH\n"
done
log "every I3lines count MUST be 0 -- no deadline was set on any of these"
log "REGRESS_DONE"
