#!/bin/bash
# r1 fine-tune RESUME: continue each arm from its epoch-1 weights (weights-only
# resume, fresh optimizer at the same fixed low lr — fine for a fine-tune).
# Outputs land in NEW dirs (dyna_r1_{arm}b) so epoch numbering stays unambiguous:
#   5050b epoch_1 = semantic epoch 2 of 2   (done after 1 more epoch)
#   8020b epoch_2 = semantic epoch 3 of 3   (done after 2 more epochs)
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel
LOGS=/workspace/logs
MIXDIR=/workspace/dyna_data
CKROOT=/workspace/swm_home/checkpoints
DRV=$LOGS/driver_r1_finetune.log
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }

ft(){ # arm gpu epochs init
  local arm=$1 gpu=$2 epochs=$3 init=$4
  local name="dyna_r1_${arm}b"
  log "finetune ${name}: resume from ${init} (gpu${gpu}, ${epochs} epochs, lr 1e-5, w24)"
  cd "$CODE"
  INIT_WEIGHTS=$init CUDA_VISIBLE_DEVICES=$gpu timeout 86400 python3 \
    scripts/train/lewm_expert.py data=ogb \
    data.dataset.name="$MIXDIR/mix_r1_${arm}.lance" \
    "data.dataset.keys_to_load=[pixels,action]" \
    "data.dataset.keys_to_cache=[action]" \
    "data.dataset.keys_to_merge=null" \
    num_workers=24 \
    optimizer.lr=1e-5 trainer.max_epochs="$epochs" trainer.devices=1 \
    output_model_name="$name" subdir="$name" \
    +action_stats_pin=expert wandb.enabled=false \
    > "$LOGS/ft_${name}.log" 2>&1 \
    || { log "finetune ${name}: FAILED"; return 1; }
  log "finetune ${name}: done"
}

ft 5050 0 1 "$CKROOT/dyna_r1_5050/weights_epoch_1.pt" &
P1=$!
ft 8020 1 2 "$CKROOT/dyna_r1_8020/weights_epoch_1.pt" &
P2=$!
wait $P1 || log "arm 5050b failed"
wait $P2 || log "arm 8020b failed"
ls -la "$CKROOT"/dyna_r1_*b/ | tee -a "$DRV"
log "R1_FINETUNE_RESUME_DONE"
