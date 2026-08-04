#!/bin/bash
# Retrain RLP for receding_horizon=1 deployment. 6000 steps, 4 alignment modes.
#
# WHY RETRAIN. Deployed at rh=1 only the FIRST chunk of each plan executes, but
# the loss scores only the plan TERMINAL -- and a terminal cost does not
# constrain the path, so the executed prefix is the least-determined part of the
# plan. Measured at rh=1 without any alignment: 100.0 -> 86.0 @0.1.
#
# THE FOUR MODES (all = which rollout timesteps enter the loss):
#   terminal      z_H only. Control -- isolates the 6000-step change from the
#                 alignment change. NOTE this is the arm most at risk: at rh=5,
#                 1000 steps beat 3000 in ALL 12 swept pairings, so a longer
#                 budget under a terminal-only cost may over-optimise the WM.
#   dense         sum_t gamma^(t+1) V(z_t,z_g), normalised. Best-evidenced: PWM's
#                 dense-vs-terminal was +16.8 (lejepa) / +64.9 (pldm) at matched
#                 budget, and terminal-only collapsed to 0.3 at 8x steps while
#                 dense improved.
#   prefix        V(z_1,z_g) -- exactly the chunk rh=1 executes.
#   randhorizon   V(z_h,z_g), h ~ U{1..H}. Every prefix must land, so any
#                 truncation point is valid. Needs no positional-embedding change.
#
# Stage 1 (this script): 4 modes x amax {1.8,2.2} x 2 bases = 16 trainings, one
# seed each, screened on the SELECTION seeds. Stage 2 sweeps replay-prob and
# expand-weight at the winning mode, so the alignment question is answered before
# the knob grid multiplies it (the full cross is 64+ trainings at ~45 min each).
#
# Training does NOT depend on I3: I3 is an eval-only readout and is currently
# not firing (frame-vs-chunk axis bug, unverified). These actors are valid
# regardless; their rh=1 cards wait on I3 being fixed.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
L=/workspace/logs/rh1re; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][rh1re] $*"; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
lr_of(){ [ "$1" = "lejepa" ] && echo 1e-4 || echo 3e-4; }
lrf_of(){ [ "$1" = "lejepa" ] && echo 1e-5 || echo 3e-5; }
mw_of(){ [ "$1" = "lejepa" ] && echo 0.3 || echo 0.5; }

grep -q "align_mode" "$PLAN/train_lip_ac.py" || { log "FATAL: align patch missing"; exit 1; }

log "16 trainings: 4 modes x amax{1.8,2.2} x 2 bases, 6000 steps"
i=0
for M in terminal dense prefix randhorizon; do
  W=0.3; [ "$M" = "terminal" ] && W=0.0
  for B in lejepa pldm; do for AM in 1.8 2.2; do
    T="${B}_${M}_a${AM//./}"
    A=/workspace/actors/rh1_${T}.pt
    [ -f "$A" ] && { i=$((i+1)); continue; }
    CUDA_VISIBLE_DEVICES=$((i % 6)) timeout 43200 python3 "$PLAN/train_lip_ac.py" \
      --cache /workspace/caches/canon_${B}_fs5.pt \
      --cache-td /workspace/caches/canon_${B}_fs1.pt \
      --h5 /workspace/reacher_slim.h5 --wm "$(wm_of $B)" --pad-context \
      --init-value /workspace/metrics/window3_${B}_e005_g098.pt \
      --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
      --gamma 0.98 --expectile 0.1 --expectile-final 0.03 \
      --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --lambda-schedule uniform --mean-weight "$(mw_of $B)" --amax "$AM" \
      --replay-prob 0.5 --expand-weight 0 --steps 6000 \
      --align-mode "$M" --align-weight "$W" \
      --actor-lr "$(lr_of $B)" --actor-lr-final "$(lrf_of $B)" --seed 0 \
      --out "$A" --out-value "/workspace/metrics/rh1_${T}_value.pt" \
      > "$L/${T}.log" 2>&1 && log "  ${T} DONE" || log "  ${T} FAILED" &
    i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
  done; done
done
wait
log "actors: $(ls /workspace/actors/rh1_*.pt 2>/dev/null | grep -vc value)/16"
for f in "$L"/*.log; do
  printf "  %-28s %s\n" "$(basename $f .log)" \
    "$(grep -oE 'E_final [0-9.]+' "$f" | tail -1)"
done
log "RH1_RETRAIN_DONE -- cards pending I3 (eval-only readout, currently not firing)"
