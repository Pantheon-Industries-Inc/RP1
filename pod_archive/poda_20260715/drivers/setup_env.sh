#!/bin/bash
set -x
apt-get update && apt-get install -y libosmesa6 libgl1-mesa-dri libegl1 libglx-mesa0 rsync zstd
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install mujoco==3.10.0 gymnasium==1.3.0 h5py==3.16.0 hdf5plugin==7.0.0 \
  hydra-core==1.3.4 omegaconf==2.3.1 numpy==2.4.6 einops==0.8.2 scikit-learn==1.9.0 \
  transformers==5.13.0 stable-pretraining==0.1.7 ogbench==1.2.1 imageio==2.37.3 \
  imageio-ffmpeg==0.6.0 pygame pymunk shapely lancedb==0.34.0 pylance==8.0.0 zstandard==0.25.0 \
  loguru tqdm
echo ok > /workspace/env.done
