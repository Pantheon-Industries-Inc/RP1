#!/usr/bin/env bash
# DINO-WM cube canonical reproduction (handoff recipe) — smoke then full run.
# Usage: run_dinowm_cube.sh [smoke|full] [gpu]
set -euo pipefail
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export OMP_NUM_THREADS=4
cd /workspace/code/stable-worldmodel

MODE=${1:-smoke}
GPU=${2:-5}

if [ "$MODE" = ogb ] || [ "$MODE" = ogbsmoke ]; then
  # PRIMARY: DINO-WM on the SWM paper's own dataset galilai-group/ogb_cube_single
  # (Lance, 224px JPEG, OGBench-derived, 10k ep x 201) — the dataset behind the
  # repo's OGB-Cube 86%, NOT the quentinll expert h5 (that gave the action-blind WM).
  # ACTION-ONLY (np): ogb_cube_single has NO native proprio column (only qpos/qvel
  # + split proprio_* parts); qpos||qvel proprio self-inflated & killed the expert
  # runs, so drop it. np = stable, drivable by LIP/TD, and tests the core question:
  # does ogb_cube_single's action-diversity yield an action-SENSITIVE WM?
  # Frozen dinov2_small, b64, lr 5e-4, wd 0.01, bf16, 100k-step cap (~4.5h).
  STEPS=100000; SUB=ogbench_cube_single_dino_ogb_20260719; NAME=ogbench_cube_single_dino_ogb; EXTRA=""
  if [ "$MODE" = ogbsmoke ]; then STEPS=30; SUB=smoke_ogb; NAME=smoke_ogb; EXTRA="+save_every_steps=10"; fi
  CUDA_VISIBLE_DEVICES=$GPU python -u scripts/train/prejepa_cube.py \
    dataset_name=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance \
    output_model_name=$NAME subdir=$SUB \
    batch_size=64 num_workers=12 seed=42 \
    trainer.strategy=auto trainer.devices=1 trainer.precision=bf16-mixed \
    trainer.max_epochs=10 +trainer.max_steps=$STEPS $EXTRA \
    optimizer.weight_decay=0.01 \
    '~wm.encoding.proprio'
elif [ "$MODE" = ogbp ]; then
  # FAITHFUL exact-86 attempt: WITH proprio = qpos||qvel (dim 41, = handoff's proprio),
  # proprio_clip=10 + NaN guards as the anti-instability measures. May still tip
  # (expert-structured data); guards will report. Runs on a separate GPU in parallel.
  CUDA_VISIBLE_DEVICES=$GPU python -u scripts/train/prejepa_cube.py \
    dataset_name=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance \
    output_model_name=ogbench_cube_single_dinop_ogb \
    subdir=ogbench_cube_single_dinop_ogb_20260719 \
    batch_size=64 num_workers=12 seed=42 \
    trainer.strategy=auto trainer.devices=1 trainer.precision=bf16-mixed \
    trainer.max_epochs=10 +trainer.max_steps=100000 \
    optimizer.weight_decay=0.01 \
    '+merge_proprio_from=[qpos,qvel]' +proprio_clip=10.0
elif [ "$MODE" = np ]; then
  # no-proprio variant (like tworoom dinowm[np]): kills the self-inflating
  # proprio-embedder branch (target = own detached output, unanchored scale)
  # that destabilized runs 1-4; action-only = drivable by the LIP/TD stack.
  CUDA_VISIBLE_DEVICES=$GPU python -u scripts/train/prejepa_cube.py \
    dataset_name=/workspace/datasets/lewm_cube_full/cube_single_expert.h5 \
    output_model_name=ogbench_cube_single_dinonp_rep \
    subdir=ogbench_cube_single_dinonp_rep_20260719 \
    batch_size=64 num_workers=12 \
    trainer.strategy=auto trainer.devices=1 trainer.precision=bf16-mixed \
    trainer.max_epochs=10 +trainer.max_steps=100000 \
    optimizer.weight_decay=0.01 \
    '~wm.encoding.proprio'
elif [ "$MODE" = smoke ]; then
  CUDA_VISIBLE_DEVICES=$GPU python -u scripts/train/prejepa_cube.py \
    dataset_name=/workspace/datasets/lewm_cube_full/cube_single_expert.h5 \
    output_model_name=smoke_dinorep subdir=smoke_dinorep \
    batch_size=64 num_workers=8 \
    trainer.strategy=auto trainer.devices=1 trainer.precision=bf16-mixed \
    trainer.max_epochs=1 +trainer.max_steps=30 +save_every_steps=10 \
    '+merge_proprio_from=[qpos,qvel]'
else
  CUDA_VISIBLE_DEVICES=$GPU python -u scripts/train/prejepa_cube.py \
    dataset_name=/workspace/datasets/lewm_cube_full/cube_single_expert.h5 \
    output_model_name=ogbench_cube_single_dino_rep \
    subdir=ogbench_cube_single_dino_rep_20260719 \
    batch_size=64 num_workers=12 \
    trainer.strategy=auto trainer.devices=1 trainer.precision=bf16-mixed \
    trainer.max_epochs=10 +trainer.max_steps=100000 \
    optimizer.weight_decay=0.01 \
    '+merge_proprio_from=[qpos,qvel]' +proprio_clip=10.0
fi
