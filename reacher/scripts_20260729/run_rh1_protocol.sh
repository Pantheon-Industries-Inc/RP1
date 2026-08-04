#!/bin/bash
# Re-card every arm at receding_horizon=1, the published replanning cadence.
#
# WHY. The campaign ran receding_horizon=5=horizon: all 25 primitive steps of a
# plan execute open-loop and the policy replans ONCE in a 50-step episode. The
# code justified this as matching "the LeWM paper's entire optimized action
# sequence is executed before replanning" -- but that sentence is not in the
# LeWM paper, and the three relevant papers say the opposite:
#
#   DINO-WM (2411.04983): "the first k actions a_0,...a_k is executed in the
#     environment. The process then repeats at the next time step"
#   PLDM (2502.14819): "re-plans at every k-th interaction with the
#     environment. Unless stated otherwise, we use k=1"
#   stable-worldmodel (2605.21800): "of which the K first actions are executed
#     before replanning at the next step"
#
# So the reported protocol is materially more open-loop than any of them. It was
# applied identically to every arm, so the comparison is internally consistent,
# and it is the setting under which the published Latent+CEM anchors were
# reproduced (86.7/77.3 vs 86/78). But it plausibly FAVOURS RLP: producing one
# good 25-step sequence is rewarded, while frequent replanning lets a cheap
# sampler correct its own errors -- which is the very advantage the paper
# attributes to learned refinement. That has to be measured, not argued.
#
# NO RETRAINING IS NEEDED. Training never touches the environment (frozen world
# model, cached latents, imagined rollouts), so every value, RLP actor and PWM
# actor stays valid. This is purely a re-card.
#
# WHAT THIS DOES AND DOES NOT MATCH. receding_horizon is counted in ACTION
# BLOCKS, so 1 = replan every 5 primitive steps = 10 cycles per episode, versus
# 2 before. That matches DINO-WM's "first k actions" and the platform's "K
# first actions". It does NOT match PLDM's literal k=1 primitive step: that
# needs action_block=1, which changes the action representation from 10-d to
# 2-d and would require retraining everything.
#
# Results go to a SEPARATE csv so the receding_horizon=5 numbers survive as the
# replanning-frequency ablation.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_rh1.csv; touch "$SUM"
L=/workspace/logs/rh1; mkdir -p "$L"
REP="42 43 44 45 46 47"
RH="plan_config.receding_horizon=1"
L2=/workspace/metrics/l2window3.pt
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][rh1] $*"; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
val_of(){ echo "/workspace/metrics/window3_${1}_e005_g098.pt"; }

ev(){ local gpu=$1 B=$2 nm=$3 seed=$4; shift 4
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 21600 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$(wm_of $B)" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    $RH "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && log "  !! ${nm} no score: $(tail -1 "$L/${nm}.log" | cut -c1-70)"
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
mean(){ grep "^${1}" "$SUM" | grep -oE "${2}=[0-9.]+" | cut -d= -f2 \
        | awk '{t+=$1;n++}END{printf "%.1f (n=%d)", (n?t/n:0), n}'; }

# ---------------------------------------------------------------- cheap arms first
log "RLP: 2 bases x 6 train seeds x 6 env seeds (~16 rollouts/decision, cheap)"
i=0
for B in lejepa pldm; do for sd in 0 1 2 3 4 5; do
  A=/workspace/actors/lip4_leak6_${B}_s${sd}.pt; [ -f "$A" ] || continue
  ( for s in $REP; do ev $((i % 6)) "$B" "rh1_rlp_${B}_s${sd}_e${s}" "$s" \
      solver=lip solver.actor_path="$A" solver.rollout_compat=false; done ) &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done
wait
log "PWM: 2 bases x 3 seeds (0 rollouts)"
i=0
for B in lejepa pldm; do for sd in 0 1 2; do
  A=/workspace/actors/pwmabl_${B}_s1000_terminal_s${sd}.pt; [ -f "$A" ] || continue
  ( for s in $REP; do ev $((i % 6)) "$B" "rh1_pwm_${B}_s${sd}_e${s}" "$s" \
      solver=pwm solver.actor_path="$A"; done ) &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done
wait

# ---------------------------------------------------------------- sampling arms (5x costlier now)
# 4-wide, not 6: sampling-solver evals have core-dumped under contention six
# times in this campaign, and each is now 5x longer.
log "sampling arms at 3000 rollouts x 10 planning cycles -- 4-wide"
i=0
for B in lejepa pldm; do
  ( for s in $REP; do ev $((i % 6)) "$B" "rh1_l2cem_${B}_s${s}"  "$s" solver=cem solver.n_steps=10 "+metric=$L2"; done ) &
  i=$((i + 1))
  ( for s in $REP; do ev $((i % 6)) "$B" "rh1_tdcem_${B}_s${s}"  "$s" solver=cem solver.n_steps=10 "+metric=$(val_of $B)"; done ) &
  i=$((i + 1))
  wait
  ( for s in $REP; do ev $((i % 6)) "$B" "rh1_l2adam_${B}_s${s}" "$s" solver=adam solver.n_steps=10 solver.num_samples=300 "+metric=$L2"; done ) &
  i=$((i + 1))
  ( for s in $REP; do ev $((i % 6)) "$B" "rh1_tdadam_${B}_s${s}" "$s" solver=adam solver.n_steps=10 solver.num_samples=300 "+metric=$(val_of $B)"; done ) &
  i=$((i + 1))
  wait
done

# ---------------------------------------------------------------- report
log "============ receding_horizon=1 (published cadence) vs =5 (reported) ============"
log "  arm                    base     rh=1 @0.1        rh=5 @0.1"
for B in lejepa pldm; do
  rlp5=$([ "$B" = lejepa ] && echo 98.2 || echo 94.2)
  l25=$([ "$B" = lejepa ] && echo 84.3 || echo 78.3)
  td5=$([ "$B" = lejepa ] && echo 51.3 || echo 51.3)
  l2a5=$([ "$B" = lejepa ] && echo 43.3 || echo 45.7)
  tda5=$([ "$B" = lejepa ] && echo 29.7 || echo 32.7)
  pwm5=$([ "$B" = lejepa ] && echo 64.1 || echo 20.4)
  log "  RLP (~16)              ${B}   $(mean rh1_rlp_${B}_ held10)   ${rlp5}"
  log "  Latent+CEM (3000)      ${B}   $(mean rh1_l2cem_${B}_ held10)   ${l25}"
  log "  Value+CEM (3000)       ${B}   $(mean rh1_tdcem_${B}_ held10)   ${td5}"
  log "  Latent+Adam (3000)     ${B}   $(mean rh1_l2adam_${B}_ held10)   ${l2a5}"
  log "  Value+Adam (3000)      ${B}   $(mean rh1_tdadam_${B}_ held10)   ${tda5}"
  log "  PWM (0)                ${B}   $(mean rh1_pwm_${B}_ held10)   ${pwm5}"
done
log "RH1_DONE -- if RLP's margin survives, the compute claim is much stronger;"
log "if it shrinks, the reported protocol was flattering it and we need to know."
