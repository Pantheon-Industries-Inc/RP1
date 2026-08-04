#!/bin/bash
# I3 smoke: does the readout fire where it must, and stay inert where it must?
#
# Four single cells, LeWM base, RLP actor lip4_leak6_lejepa_s0, env seed 42 --
# the exact cell the handoff's reference numbers were measured on.
#
#   A  rh=5 + deadline   MUST NOT print [I3]  and MUST score 66.0 / 100.0
#                        (chunks_remaining is 10 then 5, never < H=5)
#   B  rh=1 + deadline   MUST print [I3]      -- the fix under test
#   C  rh=1, no deadline MUST NOT print [I3]  -- unaligned control, 38-42 / 86.0
#   D  rh=1 + deadline, Latent+CEM            -- the sampling path's own readout
#                        (_MetricCost, _m>=3): [metric-hook] then [I3]
#
# A failing A means I3 leaked into the reported protocol. A silent B means the
# attribute still does not reach the solver. C is the number B has to beat.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
L=/workspace/logs/i3fix; mkdir -p "$L"
ACT=/workspace/actors/lip4_leak6_lejepa_s0.pt
L2=/workspace/metrics/l2window3.pt
log(){ echo "[$(date -u +%H:%M:%S)][i3smoke] $*"; }

ev(){ local gpu=$1 nm=$2; shift 2
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 5400 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy=/workspace/swm_home/checkpoints/lejepa_reacher \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=42 \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
}

RLP="solver=lip solver.actor_path=$ACT solver.rollout_compat=false"

log "wave 1: A (rh=5 +deadline, inertness) | B (rh=1 +deadline, the fix)"
ev 4 A_rh5_dl $RLP plan_config.receding_horizon=5 +plan_config.deadline=50 &
ev 5 B_rh1_dl $RLP plan_config.receding_horizon=1 +plan_config.deadline=50 &
wait
log "wave 2: C (rh=1 no deadline, control) | D (rh=1 +deadline, Latent+CEM)"
ev 4 C_rh1_nodl $RLP plan_config.receding_horizon=1 &
ev 5 D_rh1_dl_l2cem solver=cem solver.n_steps=10 "+metric=$L2" \
     plan_config.receding_horizon=1 +plan_config.deadline=50 &
wait

log "================ RESULT ================"
for nm in A_rh5_dl B_rh1_dl C_rh1_nodl D_rh1_dl_l2cem; do
  f="$L/${nm}.log"
  h=$(grep -oE "HELD-at-end [0-9.]+" "$f" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$f" | tail -1 | sed -E "s/0\.1rad //")
  i3=$(grep -c "^\[I3\]" "$f")
  hook=$(grep -c "metric-hook" "$f")
  dbg=$(grep -oE "\[I3-dbg\].*" "$f" | head -1)
  printf "  %-16s held@0.05=%-6s @0.1=%-6s [I3]x%s hook x%s\n" \
    "$nm" "${h:-FAIL}" "${t10:-FAIL}" "$i3" "$hook"
  [ -n "$dbg" ] && printf "      %s\n" "$dbg"
  grep -oE "^\[I3\].*" "$f" | sort -u | head -3 | sed "s/^/      /"
done
log "expect: A [I3]x0 and 66.0/100.0 | B [I3]x>=1 | C [I3]x0 | D hook>=1"
log "I3SMOKE_DONE"
