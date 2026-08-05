#!/bin/bash
# LIPv4-on-DINO sweep, round 1: amax {1.2, 1.6, 2.0} at the canonical recipe.
#
# Recipe = the exact train_lip_ac configuration behind LeWM's 87.6 (arch v4,
# co-trained critic, expectile 0.1->0.03, critic-lr 1e-3->1e-4, actor-lr
# 3e-4->3e-5, iters 8, horizon 5, n-step 50, warm-start from the 24.6M-sample
# pooled TD teacher), with only the data/rollout layer swapped for token space.
# Deviations from canonical, both forced and recorded:
#   batch 32 (not 128): token rollouts OOM beyond ~b64 (139 GB measured at
#     b256/iters4); smoke confirmed b32/iters8 fits.
#   steps 3000 (not 6000): ~6-9 s/step vs LeWM's sub-second => 6000 steps would
#     be 2 days/cell. Checkpoints at s1000/s2000/s3000 let the eval daemon
#     read the training curve; if s3000 is still climbing, extend the winner.
# amax axis first (LeWM plateau was 1.4-2.2); actor-lr axis on the winner next.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1
CODE=/workspace/code/stable-worldmodel
L=/workspace/logs
mkdir -p "$L" /workspace/actors /workspace/metrics
cd "$CODE"

launch(){ # gpu amax
  local g=$1 A=$2 tag
  tag="lip4d_a${A/./}"
  if [ -f "/workspace/actors/${tag}_s3000.pt" ]; then
    echo "[sweep] $tag complete, skip"; return 0
  fi
  nohup env CUDA_VISIBLE_DEVICES=$g python3 scripts/plan/train_lip_ac_dino.py \
    --cache /workspace/caches/dinopool_tr8000_fs5.pt \
    --cache-td /workspace/caches/dinopool_tr8000_fs1.pt \
    --dataset /root/datasets/ogb_cube_single/ogb_cube_single.lance \
    --h5 /workspace/datasets/expert_actions.h5 \
    --wm /workspace/ckpts/dinowm_noprop_cube \
    --init-value /workspace/metrics/dinopool_td_24k.pt \
    --horizon 5 --iters 8 --steps 3000 --batch 32 --n-step 50 --amax "$A" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed 0 --ckpt-every 1000 \
    --out "/workspace/actors/${tag}.pt" \
    --out-value "/workspace/metrics/${tag}_value.pt" \
    > "$L/${tag}.log" 2>&1 &
  echo "[sweep] $tag launched on GPU $g (pid $!)"
}

launch 1 1.2
launch 2 1.6
launch 3 2.0
echo "[sweep] all cells launched $(date -u +%H:%M:%S)"
