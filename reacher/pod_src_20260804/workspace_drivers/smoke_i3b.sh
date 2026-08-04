#!/bin/bash
# I3 smoke, second pass. Same cell as before (RLP lejepa s0, env seed 42) so
# every number is directly comparable to:
#   A rh=5 +deadline  66.0 / 100.0 (inert)   C rh=1 no deadline  42.0 / 86.0
#   B rh=1 +deadline, min readout  24.0 / 86.0
#
#   A2  rh=5 +deadline, mode=deadline  MUST stay 66.0 / 100.0 and print no [I3]
#   B2  rh=1 +deadline, mode=deadline  the readout under test
#   B3  rh=1 +deadline, mode=min       MUST reproduce 24.0 / 86.0 (the old path
#                                      still works; the axis fix is RLP-inert
#                                      because _V never used the frame offset)
#   D2  rh=1 +deadline, Latent+CEM     MUST NOT crash; [I3] on frames [..]
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
L=/workspace/logs/i3fix; mkdir -p "$L"
ACT=/workspace/actors/lip4_leak6_lejepa_s0.pt
L2=/workspace/metrics/l2window3.pt
log(){ echo "[$(date -u +%H:%M:%S)][i3smoke2] $*"; }

ev(){ local gpu=$1 nm=$2 mode=$3; shift 3
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu I3_MODE=$mode \
  timeout 5400 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy=/workspace/swm_home/checkpoints/lejepa_reacher \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=42 \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
}
RLP="solver=lip solver.actor_path=$ACT solver.rollout_compat=false"

log "wave 1: A2 (inertness, mode=deadline) | B2 (rh=1, mode=deadline)"
ev 4 A2_rh5_dl deadline $RLP plan_config.receding_horizon=5 +plan_config.deadline=50 &
ev 5 B2_rh1_deadline deadline $RLP plan_config.receding_horizon=1 +plan_config.deadline=50 &
wait
log "wave 2: B3 (rh=1, mode=min regression) | D2 (Latent+CEM, was crashing)"
ev 4 B3_rh1_min min $RLP plan_config.receding_horizon=1 +plan_config.deadline=50 &
ev 5 D2_rh1_l2cem deadline solver=cem solver.n_steps=10 "+metric=$L2" \
     plan_config.receding_horizon=1 +plan_config.deadline=50 &
wait

log "================ RESULT ================"
printf "  %-18s %-9s %-9s %s\n" arm @0.05 @0.1 "I3 line"
for nm in A2_rh5_dl B2_rh1_deadline B3_rh1_min D2_rh1_l2cem; do
  f="$L/${nm}.log"
  h=$(grep -oE "HELD-at-end [0-9.]+" "$f" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$f" | tail -1 | sed -E "s/0\.1rad //")
  printf "  %-18s %-9s %-9s %s\n" "$nm" "${h:-FAIL}" "${t10:-FAIL}" \
    "$(grep -oE '^\[I3\].*' "$f" | head -1 | cut -c1-80)"
  grep -q "Traceback" "$f" && printf "      CRASH: %s\n" "$(grep -A1 Traceback "$f" | tail -1 | cut -c1-90)"
  grep -oE "\[I3-dbg\].*" "$f" | head -1 | sed "s/^/      /"
done
log "reference: A 66.0/100.0 no-I3 | B(min) 24.0/86.0 | C(no dl) 42.0/86.0"
log "I3SMOKE2_DONE"
