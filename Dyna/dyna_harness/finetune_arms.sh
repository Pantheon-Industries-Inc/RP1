#!/bin/bash
# Dyna round-1 fine-tune arms from v2WM (both in parallel, one GPU each).
#   arm5050: expert + 25x on-policy (~50/50 gradient exposure), 2 epochs
#   arm8020: expert + 6x  on-policy (~80/20),                   3 epochs
# Recipe: single-GPU batch 128 (precedented), lr 1e-5 (5x down from 5e-5),
# expert action-stats pin, warm-start INIT_WEIGHTS=v2WM. Per-epoch ckpts.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel
LOGS=/workspace/logs
DRV=$LOGS/driver_arms.log
INIT=/workspace/models/v2WM/weights_epoch_22.pt
mkdir -p "$LOGS"

log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }

cd "$CODE"
for arm in arm5050 arm8020; do
  cat > "scripts/train/config/data/${arm}_cube.yaml" <<YAML
dataset:
  name: /workspace/dyna_data/${arm}.lance
  num_steps: \${eval:'\${wm.num_preds} + \${wm.history_size}'}
  frameskip: 5
  keys_to_load:
    - pixels
    - action
  keys_to_cache:
    - action
YAML
done
log "data configs written"

train_arm(){ # name gpu epochs
  local name=$1 gpu=$2 epochs=$3
  STABLEWM_HOME=/workspace/runs/${name}/wm_home INIT_WEIGHTS=$INIT \
  CUDA_VISIBLE_DEVICES=$gpu timeout 21600 python3 scripts/train/lewm_expert.py \
    --config-name lewm data=${name}_cube \
    output_model_name=wm1_${name} subdir=wm1_${name} +action_stats_pin=expert \
    optimizer.lr=1e-5 trainer.max_epochs=$epochs trainer.devices=1 \
    loader.batch_size=128 num_workers=16 wandb.enabled=false \
    > "$LOGS/train_${name}.log" 2>&1
  log "train_${name}: exit=$?"
}

mkdir -p /workspace/runs/arm5050/wm_home /workspace/runs/arm8020/wm_home
log "launching arms: 5050 on gpu0 (2 epochs), 8020 on gpu1 (3 epochs)"
train_arm arm5050 0 2 &
train_arm arm8020 1 3 &
wait
log "ARMS_TRAIN_DONE"
