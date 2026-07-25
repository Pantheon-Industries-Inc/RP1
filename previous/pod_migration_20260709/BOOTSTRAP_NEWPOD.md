# New pod bootstrap — LeWM planner project (migrated 2026-07-09)

Everything irreplaceable from pod 31.24.80.34 (volume 062p6ai5m5) is in
`pod_bundle_20260709.tgz` (1.59GB, md5 `197d3ba5a946989567a40047fb7c7eaf`).
Big datasets are NOT in the bundle — they re-download from public sources (§3).

**Pod shape:** 4×H100 (or similar), **network volume ≥ 300GB** mounted at `/workspace`.
Storage budget: lewm-cube full h5 102GB + ogbench_swm rebuilds ~25GB + eval sets ~14GB
+ caches/ckpts/metrics ~6GB + headroom.

## 1. Upload + unpack

```bash
bash upload_to_newpod.sh <HOST> <PORT>          # from this folder, uses ~/.ssh/id_ed25519
# on the pod:
cd /workspace && tar -xzf pod_bundle_20260709.tgz && rm pod_bundle_20260709.tgz
```

Bundle layout (extracts into /workspace): `snapshot2/` (working repo copy, PYTHONPATH-style,
includes all project modifications — metric hook in eval_wm.py, trm/, solver/mppi_actor.py),
`valscripts/` (all training/eval/diag scripts), `ckpts/` (ALL frozen WMs: v2WM
`ogbench_cube_single_v2WM/weights_epoch_22.pt`, 3×DINO, 3×LeWM + 412 result .txt),
`metrics/` (trained value functions + LIP actors), 3 v2WM latent caches in `caches/`,
all top-level runner `.sh`, result `.log`s, and notes `.md`.

## 2. Environment (old pod: Python 3.11.10, CUDA 12.4 image)

```bash
apt-get update && apt-get install -y libosmesa6 libgl1-mesa-dri libegl1 libglx-mesa0 rsync zstd
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install mujoco==3.10.0 gymnasium==1.3.0 h5py==3.16.0 hdf5plugin==7.0.0 \
  hydra-core==1.3.4 omegaconf==2.3.1 numpy==2.4.6 einops==0.8.2 scikit-learn==1.9.0 \
  transformers==5.13.0 stable-pretraining==0.1.7 ogbench==1.2.1 imageio==2.37.3 \
  imageio-ffmpeg==0.6.0 pygame pymunk shapely lancedb==0.34.0 pylance==8.0.0 zstandard==0.25.0
# imageio-ffmpeg is LOAD-BEARING: eval saves rollout mp4s BEFORE printing metrics —
# without it every eval crashes at the end with all results lost (cost us a 2h rerun).
# pygame/pymunk/shapely: imported at stable_worldmodel.envs package import time.
```

(`pip_freeze_oldpod.txt` in this folder has the full pin list if anything else is missed.
`mujoco==3.10.0` is load-bearing: physics parity with the lewm-cube dataset was verified
byte-exact on this version. MUJOCO_GL=osmesa; renders verified identical to dataset pixels.)

Usage pattern everywhere: `export PYTHONPATH=/workspace/snapshot2 MUJOCO_GL=osmesa TQDM_DISABLE=1`
— the repo is NOT pip-installed.

## 3. Datasets (re-download, ~1h)

```bash
# lewm-cube full (102GB h5; the 84.0-replication dataset) — onto the BIG volume:
mkdir -p /workspace/datasets/lewm_cube_full
curl -sL "https://huggingface.co/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst" \
  | zstd -dc | tar -x -C /workspace/datasets/lewm_cube_full \
  && echo OK > /workspace/datasets/lewm_cube_full/extract.done
# (~28 MiB/s single stream ≈ 55 min. Every episode is exactly 201 steps, 10k episodes.)

# OGBench raw (only needed for the official-protocol thread; scripts in valscripts):
python -c "import ogbench; ogbench.download_datasets(['visual-cube-single-play-v0','visual-cube-double-play-v0'], '/workspace/datasets/ogbench_raw')"
# then valscripts/convert_ogbench_eval.py + augment_h5.py rebuild ogbench_swm/ogbench_eval_h5 sets.
```

If the OLD pod is still alive, direct copy beats rebuild for the 14GB eval sets + 25GB ogbench_swm:
`rsync -av -e "ssh -p 15169 -i ~/.ssh/id_ed25519" root@31.24.80.34:/workspace/datasets/ogbench_eval_h5/ /workspace/datasets/ogbench_eval_h5/`

## 4. FIRST JOB: finish the 84-replication (state as of migration)

Goal: replicate the documented Latent+CEM 84.0 on lewm-cube with the received v2WM
(= their exact checkpoint `weights_epoch_22.pt`; doc: stable-worldmodel/health-check/
replication_cube/REPLICATION_CUBE.md in the local repo).

Verified so far (all cleared as gap candidates):
- osmesa re-renders == stored dataset pixels (pixel + v2WM-latent identity)
- action z-score stats match their documented constants to ≤1%
- physics replay parity ≤1e-3 qpos drift over 175 steps (mujoco 3.10.0)
- cem.py identical to clean repo; eval_wm.py mods inactive for plain CEM
- **seed-42 draw on the FULL 10k-ep dataset == their exact printed 50 row indices**
  (verified analytically; 39/50 of their tasks lie beyond episode 2550 — prior ~67%
  fair-shake runs used a 150-episode subset of the first 2550 = different task mix;
  random floor ~42 on both, which masked it)

Run (after §3 lewm-cube download; `replicate84_newpod.sh` in this folder = updated paths):

```bash
bash /workspace/replicate84_newpod.sh   # latent+CEM s42/43/44 + random s42, one per GPU
python /workspace/valscripts/compare_rep84.py   # headline + per-episode agreement vs their arrays
```

Read: s42 ≥ ~80 (n=50, binomial sd ≈ 5pts) with high per-episode agreement on their
8 failures {11,14,17,22,27,31,32,38} ⇒ replication achieved. If it lands ~70 instead,
next probes: per-episode flips vs their array, watch env_*.mp4 of flipped episodes.

## 5. After replication: planner thread on the full protocol

The fair-shake chain (TD value → LIP v1/v2) re-runs on the full dataset with
`valscripts/cache_render224.py` (cache episodes beyond 2550 too), `train_metric.py`
(td / quasimetric / expectile 0.03 / n-step 5), `learned_planner.py` (champion: iters 8,
lr 3e-4, 8000 steps). Existing artifacts from the 2550-prefix era are in metrics/
(cf_lewm_v2wm.pt, lip_fs_v1/v2.pt) and caches/ — comparable only against prefix-era evals.
Fair-shake prefix-era anchors (150-ep stratified set, 3-seed means): random 40.0,
latent 67.3, TD 76.0, LIPv1 82.7, LIPv2 75.3.
