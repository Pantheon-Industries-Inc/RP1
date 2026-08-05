#!/bin/bash
# Reacher Dyna round-1 fine-tune driver (lejepa base).
# 1. Pin json = action stats of the random training set (the convention the
#    WM/caches/eval all z-score with; the mix's own stats would drift because
#    on-policy LIP actions are clipped/optimized, not uniform).
# 2. Build the two arm lances: random-10k ⊕ dup(on-policy) at 50%/20% by rows.
# 3. Fine-tune the converted lejepa base on each arm, one GPU each:
#    lr 1e-5, epochs 2/3 (~4 base-set epochs of steps), w24 loaders.
# Ckpts -> $STABLEWM_HOME/checkpoints/dyna_reacher_r1_{5050,8020}/.
set -u
export PYTHONPATH=/workspace/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/stable-worldmodel
LOGS=/workspace/logs
MIXDIR=/workspace/dyna_data
DRV=$LOGS/driver_reacher_r1_ft.log
STATS=/workspace/reacher_action_stats.json
INIT=/workspace/swm_home/checkpoints/lejepa_reacher/weights.pt
EXPERT=/workspace/swm_home/datasets/dmc/reacher_random.lance
ONP="$MIXDIR/reacher_onpolicy_r1_lejepa_a0.lance $MIXDIR/reacher_onpolicy_r1_lejepa_a1.lance $MIXDIR/reacher_onpolicy_r1_lejepa_a2.lance"

log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }

if [ ! -f "$STATS" ]; then
  python3 - << "EOF" || die "stats json failed"
import json, h5py, numpy as np
with h5py.File("/workspace/caches/reacher_random_train.h5") as h:
    a = h["action"][:]
mu, sd = np.nanmean(a, 0), np.nanstd(a, 0) + 1e-6
json.dump({"mean": mu.tolist(), "std": sd.tolist()},
          open("/workspace/reacher_action_stats.json", "w"))
print("stats:", mu.tolist(), sd.tolist())
EOF
fi
log "stats json ready: $(cat $STATS)"

for arm in 5050 8020; do
  frac=0.5; [ "$arm" = "8020" ] && frac=0.2
  out="$MIXDIR/reacher_mix_r1_${arm}.lance"
  if [ ! -f "$out/.done" ]; then
    log "building reacher_mix_r1_${arm} (frac $frac)"
    python3 /workspace/build_dyna_mix.py --expert "$EXPERT" --onpolicy $ONP \
      --out "$out" --onpolicy-frac "$frac" > "$LOGS/rmix_${arm}.log" 2>&1 \
      || die "mix ${arm} build failed"
    grep -q "BUILD_MIX_DONE" "$LOGS/rmix_${arm}.log" || die "mix ${arm} incomplete"
    touch "$out/.done"
  fi
  log "reacher_mix_r1_${arm} ready: $(grep 'dup K=' $LOGS/rmix_${arm}.log)"
done

ft(){ # arm gpu epochs
  local arm=$1 gpu=$2 epochs=$3
  local name="dyna_reacher_r1_${arm}"
  log "finetune ${name}: start (gpu${gpu}, ${epochs} epochs, lr 1e-5, w24)"
  cd "$CODE"
  INIT_WEIGHTS=$INIT CUDA_VISIBLE_DEVICES=$gpu timeout 86400 python3 \
    scripts/train/lewm_expert.py data=dmc \
    data.dataset.name="$MIXDIR/reacher_mix_r1_${arm}.lance" \
    "data.dataset.keys_to_load=[pixels,action]" \
    "data.dataset.keys_to_cache=[action]" \
    num_workers=24 \
    optimizer.lr=1e-5 trainer.max_epochs="$epochs" trainer.devices=1 \
    output_model_name="$name" subdir="$name" \
    +action_stats_pin="$STATS" wandb.enabled=false \
    > "$LOGS/ft_${name}.log" 2>&1 \
    || { log "finetune ${name}: FAILED (see $LOGS/ft_${name}.log)"; return 1; }
  log "finetune ${name}: done"
}

ft 5050 0 2 &
P1=$!
ft 8020 1 3 &
P2=$!
wait $P1 || log "arm 5050 failed"
wait $P2 || log "arm 8020 failed"
ls -la /workspace/swm_home/checkpoints/dyna_reacher_r1_* 2>/dev/null | tee -a "$DRV"
log "REACHER_R1_FT_DONE"
