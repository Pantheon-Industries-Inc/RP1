#!/bin/bash
# Train any stage-A grid actors that are missing, and NOTHING else.
#
# Safe to run on the osmesa pod even though the card will be measured on the
# egl pod: training does not render. train_lip_ac.py works on cached latents
# and rolls out in latent space through the frozen WM -- no MuJoCo, no EGL.
# The actors it writes are renderer-agnostic and live on the volume, so the
# egl pod picks them up and only has to run eval cells.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export TQDM_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; DRV=$L/driver_stagea.log
PLDM=/workspace/models/PLDM_OgBench_lewm
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
NGPU=8
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
cd "$CODE"
log "=== stage-A completion (pid $$) ==="
for f in "$QF1" "$QF5" "$QTD" "$AH5"; do [ -e "$f" ] || { log "FATAL missing $f"; exit 1; }; done

g=0; launched=0
for mw in 0.1 0.3 1.0; do for am in 3.5 4.5; do for alr in 3e-4 1e-4; do
  case "$alr" in 3e-4) alrf=3e-5;; *) alrf=1e-5;; esac
  tag="mw$(echo "$mw" | tr -d .)_a$(echo "$am" | tr -d .)_lr${alr}"
  out=/workspace/actors/lip4_pldm_${tag}_s0.pt
  [ -f "$out" ] && { log "  $tag: present"; continue; }
  log "  $tag: training on gpu $g"
  CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$am" \
    --mean-weight "$mw" --actor-lr "$alr" --actor-lr-final "$alrf" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --arch v4 --seed 0 --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/grid_train_${tag}_s0.log" 2>&1 &
  g=$((g+1)); launched=$((launched+1))
  [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done; done
wait
log "stage-A: launched $launched, now $(ls /workspace/actors/ | grep -cE 'lip4_pldm_mw.*_s0\.pt$')/12 present"
log "STAGEA_DONE"
