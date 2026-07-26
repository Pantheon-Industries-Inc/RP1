#!/bin/bash
# Reacher Dyna round-1 fine-tune for the PLDM base. Uses pldm_expert.py
# (pldm.py + INIT_WEIGHTS + action-pin + GradGuard + NestedResize + cfg.model
# save-fix). Default arm 8020 (lejepa established 80/20 > 50/50). GPU arg lets
# it slot onto whichever GPU is free (GPU0 is busy with DINO stage-1).
# Gated on the PLDM on-policy collection marker.
set -u
export PYTHONPATH=/workspace/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
ARM=${1:-8020}
GPU=${2:-1}
EPOCHS=${3:-3}
CODE=/workspace/stable-worldmodel
LOGS=/workspace/logs
MIXDIR=/workspace/dyna_data
DRV=$LOGS/driver_reacher_r1_ft_pldm.log
STATS=/workspace/reacher_action_stats.json
INIT=/workspace/swm_home/checkpoints/pldm_reacher/weights.pt
EXPERT=/workspace/swm_home/datasets/dmc/reacher_random.lance
ONP="$MIXDIR/reacher_onpolicy_r1_pldm_a0.lance $MIXDIR/reacher_onpolicy_r1_pldm_a1.lance $MIXDIR/reacher_onpolicy_r1_pldm_a2.lance"

log(){ echo "[$(date -u +%m%d-%H:%M:%S)][pldm-ft] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }

log "waiting for PLDM on-policy collection"
while ! grep -q "REACHER_COLLECT_R1_DONE" "$LOGS/driver_reacher_collect_r1_pldm.log" 2>/dev/null; do sleep 120; done
log "collection done"

[ -f "$STATS" ] || die "missing $STATS (built by lejepa driver)"

frac=0.2; [ "$ARM" = "5050" ] && frac=0.5
out="$MIXDIR/reacher_mix_r1_pldm_${ARM}.lance"
if [ ! -f "$out/.done" ]; then
  log "building mix ${ARM} (frac $frac)"
  python3 /workspace/build_dyna_mix.py --expert "$EXPERT" --onpolicy $ONP \
    --out "$out" --onpolicy-frac "$frac" > "$LOGS/rmix_pldm_${ARM}.log" 2>&1 \
    || die "mix build failed"
  grep -q "BUILD_MIX_DONE" "$LOGS/rmix_pldm_${ARM}.log" || die "mix incomplete"
  touch "$out/.done"
fi
log "mix ${ARM} ready: $(grep 'dup K=' $LOGS/rmix_pldm_${ARM}.log)"

name="dyna_reacher_r1_pldm_${ARM}"
log "finetune ${name}: start (gpu${GPU}, ${EPOCHS} epochs, lr 1e-5)"
cd "$CODE"
INIT_WEIGHTS=$INIT CUDA_VISIBLE_DEVICES=$GPU timeout 86400 python3 \
  scripts/train/pldm_expert.py data=dmc \
  data.dataset.name="$out" \
  "data.dataset.keys_to_load=[pixels,action]" \
  "data.dataset.keys_to_cache=[action]" \
  num_workers=24 \
  optimizer.lr=1e-5 trainer.max_epochs="$EPOCHS" trainer.devices=1 \
  output_model_name="$name" subdir="$name" \
  +action_stats_pin="$STATS" wandb.enabled=false \
  > "$LOGS/ft_${name}.log" 2>&1 \
  || die "finetune ${name} FAILED (see $LOGS/ft_${name}.log)"
log "finetune ${name}: done"
ls -la /workspace/swm_home/checkpoints/${name}/ | tee -a "$DRV"
log "PLDM_R1_FT_DONE"
