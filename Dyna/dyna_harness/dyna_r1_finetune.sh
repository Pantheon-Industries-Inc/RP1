#!/bin/bash
# Dyna round-1 fine-tune driver (loop pod). Gated on COLLECT_R1_DONE.
# Builds the two arm lances (expert ⊕ dup(on-policy) by ROW fraction), then
# fine-tunes v2WM on each arm (single GPU each, parallel):
#   arm5050: on-policy 50% of rows, 2 epochs  (≈ 4 expert-set epochs of steps)
#   arm8020: on-policy 20% of rows, 3 epochs
# lr 1e-5 (10x down), expert action-pin, NO rollout loss. Per-epoch ckpts land
# in $STABLEWM_HOME/checkpoints/dyna_r1_{5050,8020}*/.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel
LOGS=/workspace/logs
DRV=$LOGS/driver_r1_finetune.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
ONP="/workspace/dyna_data/onpolicy_r1_a0.lance /workspace/dyna_data/onpolicy_r1_a1.lance /workspace/dyna_data/onpolicy_r1_a2.lance"
MIXDIR=/workspace/dyna_data
V2W=/workspace/models/v2WM/weights_epoch_22.pt

log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }

log "waiting for collection to finish"
while ! grep -q "COLLECT_R1_DONE" "$LOGS/driver_collect_r1.log"; do sleep 300; done
log "collection done"

for arm in 5050 8020; do
  frac=0.5; [ "$arm" = "8020" ] && frac=0.2
  out="$MIXDIR/mix_r1_${arm}.lance"
  if [ ! -f "$out/.done" ]; then
    log "building mix_r1_${arm} (frac $frac)"
    python3 /workspace/build_dyna_mix.py --expert "$EXPERT" --onpolicy $ONP \
      --out "$out" --onpolicy-frac "$frac" > "$LOGS/mix_${arm}.log" 2>&1 \
      || die "mix ${arm} build failed"
    grep -q "BUILD_MIX_DONE" "$LOGS/mix_${arm}.log" || die "mix ${arm} incomplete"
    touch "$out/.done"
  fi
  log "mix_r1_${arm} ready"
done

ft(){ # arm gpu epochs
  local arm=$1 gpu=$2 epochs=$3
  local name="dyna_r1_${arm}"
  log "finetune ${name}: start (gpu${gpu}, ${epochs} epochs, lr 1e-5)"
  cd "$CODE"
  INIT_WEIGHTS=$V2W CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 \
    scripts/train/lewm_expert.py data=ogb \
    data.dataset.name="$MIXDIR/mix_r1_${arm}.lance" \
    "data.dataset.keys_to_load=[pixels,action]" \
    "data.dataset.keys_to_cache=[action]" \
    "data.dataset.keys_to_merge=null" \
    optimizer.lr=1e-5 trainer.max_epochs="$epochs" trainer.devices=1 \
    output_model_name="$name" subdir="$name" \
    +action_stats_pin=expert wandb.enabled=false \
    > "$LOGS/ft_${name}.log" 2>&1 \
    || { log "finetune ${name}: FAILED (see $LOGS/ft_${name}.log)"; return 1; }
  log "finetune ${name}: done"
}

ft 5050 0 2 &
P1=$!
ft 8020 1 3 &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
[ "$R1" = 0 ] || log "arm 5050 failed"
[ "$R2" = 0 ] || log "arm 8020 failed"

log "checkpoints:"
ls -la /workspace/swm_home/checkpoints/dyna_r1_* 2>/dev/null | tee -a "$DRV"
log "R1_FINETUNE_DONE"
