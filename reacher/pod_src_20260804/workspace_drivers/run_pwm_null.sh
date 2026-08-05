#!/bin/bash
# Is PWM's 99.0 on pldm real, or is the eval ignoring the actor?
#
# 99.0 @0.1 and 71.5 @0.05 from a memoryless MLP with zero search is a strong
# enough claim that it should be attacked before it is believed. Two candidate
# explanations were already eliminated:
#
#   replan cadence -- reacher.yaml sets receding_horizon 5 for every solver, so
#     PWM executes all 5 blocks before replanning exactly as LIP does. It is not
#     getting extra closed-loop corrections. (It pays 5 imagined rollouts to
#     fill the plan tail, against LIP's 8 refinement passes.)
#   training budget -- a real confound (8000 steps vs LIP's 1000), but that
#     would make PWM better, not fake. run_pwm_ablate.sh measures it.
#
# What remains is that the eval never actually uses the actor -- a silent
# fallback, an ignored actor_path, or a success metric that fires regardless of
# the actions. This is the null control that settles it:
#
#   ZERO   final layer zeroed -> tanh(0)*amax = 0 for every input, so the arm
#          never moves. Goals are 25 steps away, so any score materially above
#          zero means the actions are not reaching the environment.
#   RANDOM freshly initialised weights, same architecture -> flails.
#
# If either scores anywhere near 99, the PWM result is an artifact and so is
# every other number produced through this eval path. If both score near zero,
# the actor is genuinely driving the arm and 99.0 stands (subject to the
# separate budget question).
#
# Same protocol as the real card: pldm, episodes 8000:10000, h25, budget 50.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/pldm_reacher
SUM=/workspace/results/summary_pwmnull.csv; touch "$SUM"
L=/workspace/logs/pwmnull; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][null] $*"; }

python3 - <<'PYEOF'
import torch, collections
src = "/workspace/actors/pwm_pldm_s0.pt"
b = torch.load(src, map_location="cpu", weights_only=False)
sd = b["sd"]
last = [k for k in sd if k.endswith("weight")][-1]
lastb = [k for k in sd if k.endswith("bias")][-1]
print("zeroing final layer:", last, tuple(sd[last].shape), "|", lastb)

z = dict(b); z["sd"] = collections.OrderedDict(
    (k, (torch.zeros_like(v) if k in (last, lastb) else v.clone())) for k, v in sd.items())
torch.save(z, "/workspace/actors/pwm_pldm_ZERO.pt")

g = torch.Generator().manual_seed(1234)
r = dict(b); r["sd"] = collections.OrderedDict(
    (k, (torch.randn(v.shape, generator=g) * 0.05 if v.dtype.is_floating_point else v.clone()))
    for k, v in sd.items())
torch.save(r, "/workspace/actors/pwm_pldm_RANDOM.pt")
print("wrote ZERO and RANDOM actors")
PYEOF

ev(){ local nm=$1 seed=$2 actor=$3
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=5 CUDA_VISIBLE_DEVICES=5 \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=pwm solver.actor_path="$actor" solver.batch_size=10 \
    output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && log "  !! ${nm} no score: $(tail -1 "$L/${nm}.log" | cut -c1-60)"
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
  log "  ${nm}: @0.05 ${h:-FAIL} | @0.1 ${t10:-FAIL}"
}

log "null controls on pldm (trained actor carded 99.0 @0.1 / 71.5 @0.05)"
for s in 42 43; do ev "null_zero_s${s}"   $s /workspace/actors/pwm_pldm_ZERO.pt;   done
for s in 42 43; do ev "null_random_s${s}" $s /workspace/actors/pwm_pldm_RANDOM.pt; done

log "--- also: does a bogus actor_path fail loudly, or fall back silently? ---"
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=5 CUDA_VISIBLE_DEVICES=5 timeout 1200 python3 "$PLAN/eval_wm.py" \
  --config-name reacher policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
  +eval.ep_range=8000:10000 seed=42 eval.goal_offset_steps=25 eval.eval_budget=2 \
  solver=pwm solver.actor_path=/workspace/actors/DOES_NOT_EXIST.pt solver.batch_size=2 \
  output.filename=null_bogus.txt > "$L/bogus.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  log "  !! BOGUS PATH RETURNED 0 -- silent fallback, every PWM number is suspect"
  grep -oE "HELD-at-end [0-9.]+" "$L/bogus.log" | tail -1
else
  log "  bogus path failed loudly (exit ${rc}) -- no silent fallback: $(grep -iE 'error|not found|no such' "$L/bogus.log" | tail -1 | cut -c1-70)"
fi
log "NULL_DONE"
