#!/bin/bash
# Stage 2: train the rh=1 winner at the REPORTED protocol's 6 training seeds.
#
# WINNER, from the 108-cell selection-seed screen (seeds 50-51, never reported):
#   --align-mode randhorizon --align-weight 0.3 --amax 2.2, I3 readout OFF
#   lejepa 100.0 @0.1 / 96.0 @0.05   pldm 100.0 / 97.0
# It wins on BOTH bases at the SAME amax, which no other mode does, and its
# rationale is the one that matches the deployment: V(z_h,z_g) with h~U{1..5}
# asks every prefix to land, and at rh=1 any chunk can be the executed one.
#
# The reported RLP row is 6 training seeds x 6 env seeds. One training seed is
# not a row, so this trains seeds 0-5 per base; the screen's actor was seed 0.
#
# STEPS is a parameter because the screen's actors used 6000 while the reported
# recipe uses 1000, and the terminal control shows the step count ALONE is
# harmful at rh=1 (lejepa 79.0 -> 51.0 @0.1). Pass STEPS=1000 if the dense@1000
# probe says a 1000-step budget holds up -- that makes the rh=1 recipe the
# reported recipe plus one loss term at an identical budget.
set -u
STEPS=${STEPS:-6000}
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
L=/workspace/logs/rh1final; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][rh1fin] $*"; }
lr_of(){  [ "$1" = lejepa ] && echo 1e-4 || echo 3e-4; }
lrf_of(){ [ "$1" = lejepa ] && echo 1e-5 || echo 3e-5; }
mw_of(){  [ "$1" = lejepa ] && echo 0.3  || echo 0.5; }

log "12 trainings: randhorizon a2.2, seeds 0-5 x 2 bases, ${STEPS} steps"
i=0
for sd in 0 1 2 3 4 5; do for B in lejepa pldm; do
  A=/workspace/actors/rh1fin_${B}_s${sd}.pt
  [ -f "$A" ] && { log "  ${B}_s${sd} exists"; i=$((i+1)); continue; }
  CUDA_VISIBLE_DEVICES=$((i % 6)) timeout 43200 python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/canon_${B}_fs5.pt \
    --cache-td /workspace/caches/canon_${B}_fs1.pt \
    --h5 /workspace/reacher_slim.h5 --wm /workspace/swm_home/checkpoints/${B}_reacher \
    --pad-context --init-value /workspace/metrics/window3_${B}_e005_g098.pt \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --gamma 0.98 --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight "$(mw_of $B)" --amax 2.2 \
    --replay-prob 0.5 --expand-weight 0 --steps "$STEPS" \
    --align-mode randhorizon --align-weight 0.3 \
    --actor-lr "$(lr_of $B)" --actor-lr-final "$(lrf_of $B)" --seed "$sd" \
    --out "$A" --out-value "/workspace/metrics/rh1fin_${B}_s${sd}_value.pt" \
    > "$L/${B}_s${sd}.log" 2>&1 && log "  ${B}_s${sd} DONE" || log "  ${B}_s${sd} FAILED" &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done
wait
log "actors: $(ls /workspace/actors/rh1fin_*.pt 2>/dev/null | grep -vc value)/12"
for f in "$L"/*_s?.log; do
  printf "  %-20s %s\n" "$(basename $f .log)" "$(grep -oE 'E_final [0-9.]+' "$f" | tail -1)"
done
log "RH1FINAL_TRAIN_DONE"
