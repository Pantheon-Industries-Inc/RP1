#!/bin/bash
# Re-measure the reacher table at the published replanning cadence.
#
# Three protocol variants, held-at-end throughout (the campaign's metric):
#
#   A  rh=5              plan 5 blocks, execute all 25 steps, replan once.
#                        What the campaign reported. Already carded -- not re-run
#                        here, and provably unaffected by the new patches
#                        (rh5+deadline reproduced 66.0/100.0 bit-identically,
#                        and warm start is inert when the plan tail is empty).
#   B  rh=1 + deadline   plan 5 blocks, execute 1, replan -- but commit the whole
#                        plan once <=plan_len steps remain, so the final plan's
#                        terminal lands on the graded step. THE NEW HEADLINE:
#                        the papers' cadence, made fair for a terminal cost.
#   C  rh=1 no deadline  same but without alignment. Documents what the
#                        misalignment costs; on one RLP cell it was 100.0 -> 86.0
#                        @0.1 and 58.0 -> 38.0 @0.05.
#
# WHY THIS RE-MEASURE IS NEEDED. The reported protocol executed the entire
# 25-step plan before replanning, justified in-code as matching "the LeWM paper's
# entire optimized action sequence is executed before replanning" -- a sentence
# that is not in that paper. All three relevant papers replan after a short
# chunk: DINO-WM "the first k actions ... then repeats at the next time step"
# (2411.04983), PLDM "re-plans at every k-th interaction ... we use k=1"
# (2502.14819), stable-worldmodel "the K first actions are executed before
# replanning at the next step" (2605.21800).
#
# NO RETRAINING: training never touches the environment, so all actors and
# values are reused. action_block stays 5 -- actions remain chunked for every
# arm, and PLDM's literal k=1 primitive step (action_block=1) is out of scope
# because it would change the action representation.
#
# B runs every arm. C runs only RLP and Latent+CEM -- the row the claim rests on
# and its bar -- since its purpose is to size one effect, not to rebuild a table.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_grid.csv; touch "$SUM"
L=/workspace/logs/grid; mkdir -p "$L"
REP="42 43 44 45 46 47"
L2=/workspace/metrics/l2window3.pt
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][grid] $*"; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
val_of(){ echo "/workspace/metrics/window3_${1}_e005_g098.pt"; }

# ev <gpu> <base> <name> <seed> <deadline> -- solver args follow
ev(){ local gpu=$1 B=$2 nm=$3 seed=$4 dl=$5; shift 5
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 21600 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$(wm_of $B)" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    plan_config.receding_horizon=1 +plan_config.deadline=$dl \
    "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && log "  !! ${nm} no score: $(tail -1 "$L/${nm}.log" | cut -c1-70)"
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
mean(){ grep "^${1}" "$SUM" | grep -oE "${2}=[0-9.]+" | cut -d= -f2 \
        | awk '{t+=$1;n++}END{printf "%.1f(n=%d)", (n?t/n:0), n}'; }

# ================================================================ B: rh=1 aligned
log "=== B: rh=1 with deadline alignment (new headline) ==="
log "B/RLP: 2 bases x 6 seeds x 6 env seeds"
i=0
for B in lejepa pldm; do for sd in 0 1 2 3 4 5; do
  A=/workspace/actors/lip4_leak6_${B}_s${sd}.pt; [ -f "$A" ] || continue
  ( for s in $REP; do ev $((i % 6)) "$B" "B_rlp_${B}_s${sd}_e${s}" "$s" 50 \
      solver=lip solver.actor_path="$A" solver.rollout_compat=false; done ) &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done
wait
log "B/PWM: 2 bases x 3 seeds"
i=0
for B in lejepa pldm; do for sd in 0 1 2; do
  A=/workspace/actors/pwmabl_${B}_s1000_terminal_s${sd}.pt; [ -f "$A" ] || continue
  ( for s in $REP; do ev $((i % 6)) "$B" "B_pwm_${B}_s${sd}_e${s}" "$s" 50 \
      solver=pwm solver.actor_path="$A"; done ) &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done
wait
log "B/sampling arms (5x costlier at 10 planning cycles) -- 4-wide"
i=0
for B in lejepa pldm; do
  ( for s in $REP; do ev $((i % 6)) "$B" "B_l2cem_${B}_s${s}"  "$s" 50 solver=cem solver.n_steps=10 "+metric=$L2"; done ) &
  i=$((i+1))
  ( for s in $REP; do ev $((i % 6)) "$B" "B_tdcem_${B}_s${s}"  "$s" 50 solver=cem solver.n_steps=10 "+metric=$(val_of $B)"; done ) &
  i=$((i+1)); wait
  ( for s in $REP; do ev $((i % 6)) "$B" "B_l2adam_${B}_s${s}" "$s" 50 solver=adam solver.n_steps=10 solver.num_samples=300 "+metric=$L2"; done ) &
  i=$((i+1))
  ( for s in $REP; do ev $((i % 6)) "$B" "B_tdadam_${B}_s${s}" "$s" 50 solver=adam solver.n_steps=10 solver.num_samples=300 "+metric=$(val_of $B)"; done ) &
  i=$((i+1)); wait
done

# ================================================================ C: rh=1 unaligned
log "=== C: rh=1 WITHOUT alignment (sizes the misalignment effect) ==="
i=0
for B in lejepa pldm; do for sd in 0 1 2 3 4 5; do
  A=/workspace/actors/lip4_leak6_${B}_s${sd}.pt; [ -f "$A" ] || continue
  ( for s in $REP; do ev $((i % 6)) "$B" "C_rlp_${B}_s${sd}_e${s}" "$s" 0 \
      solver=lip solver.actor_path="$A" solver.rollout_compat=false; done ) &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done
wait
i=0
for B in lejepa pldm; do
  ( for s in $REP; do ev $((i % 6)) "$B" "C_l2cem_${B}_s${s}" "$s" 0 solver=cem solver.n_steps=10 "+metric=$L2"; done ) &
  i=$((i+1)); [ $((i % 2)) -eq 0 ] && wait
done
wait

# ================================================================ report
log "================== held-at-end @0.1, by protocol =================="
log "  arm                 base     A rh=5    B rh=1+align   C rh=1 raw"
for B in lejepa pldm; do
  a_rlp=$([ "$B" = lejepa ] && echo 98.2 || echo 94.2)
  a_l2=$([ "$B" = lejepa ] && echo 84.3 || echo 78.3)
  a_td=51.3
  a_l2a=$([ "$B" = lejepa ] && echo 43.3 || echo 45.7)
  a_tda=$([ "$B" = lejepa ] && echo 29.7 || echo 32.7)
  a_pwm=$([ "$B" = lejepa ] && echo 64.1 || echo 20.4)
  log "  RLP (~16)           ${B}   ${a_rlp}      $(mean B_rlp_${B}_ held10)   $(mean C_rlp_${B}_ held10)"
  log "  Latent+CEM (3000)   ${B}   ${a_l2}      $(mean B_l2cem_${B}_ held10)   $(mean C_l2cem_${B}_ held10)"
  log "  Value+CEM (3000)    ${B}   ${a_td}      $(mean B_tdcem_${B}_ held10)   --"
  log "  Latent+Adam (3000)  ${B}   ${a_l2a}      $(mean B_l2adam_${B}_ held10)   --"
  log "  Value+Adam (3000)   ${B}   ${a_tda}      $(mean B_tdadam_${B}_ held10)   --"
  log "  PWM (0)             ${B}   ${a_pwm}      $(mean B_pwm_${B}_ held10)   --"
done
log "and @0.05:"
for B in lejepa pldm; do
  log "  RLP ${B}: B $(mean B_rlp_${B}_ held) | C $(mean C_rlp_${B}_ held)"
  log "  L2+CEM ${B}: B $(mean B_l2cem_${B}_ held) | C $(mean C_l2cem_${B}_ held)"
done
log "GRID_DONE"
