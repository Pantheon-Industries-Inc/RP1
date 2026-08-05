---
title: OGBench Cube Pretraining
summary: Verified OGBench cube data, examples, and launch plan for DINO/PreJEPA and LeWM pretraining.
---

# OGBench Cube Pretraining

Last updated: 2026-07-01.

This guide is the durable handoff document for training six Stable World Model
checkpoints on OGBench cube:

- DINO/PreJEPA 1-cube
- DINO/PreJEPA 2-cube
- DINO/PreJEPA 3-cube
- LeWM 1-cube
- LeWM 2-cube
- LeWM 3-cube

The current instruction is to finish data/example verification and
documentation first. Do not launch long training jobs until explicitly told to
do so.

## Objective

Prepare a reproducible pretraining path for OGBench cube visual data so the
resulting checkpoints can be shared with a collaborator for inference and an RL
stack on top.

Concrete deliverables:

- Verify the local Stable World Model codebase and training entrypoints.
- Verify OGBench cube 1/2/3 dataset mapping, counts, and file sizes.
- Convert official OGBench visual datasets to the HDF5 format consumed by SWM.
- Generate offline trajectory MP4 examples and SSH-friendly previews.
- Provide smoke-test commands for both model families.
- Provide full training commands for one H100 and for a three-H100 wave schedule.
- Document checkpoint paths, resume limitations, packaging, and inference usage.

## Codebase And Runtime

Primary repo:

```text
/workspace/src/Pantheon-SWM/stable-worldmodel
```

Use this repo explicitly in `PYTHONPATH`, otherwise Python may resolve
`stable_worldmodel` from another checkout such as `/workspace/src/stable-worldmodel`.

Recommended environment variables:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel
export PYTHONPATH=/workspace/src/Pantheon-SWM/stable-worldmodel:${PYTHONPATH:-}
export STABLEWM_HOME=/workspace/.stable_worldmodel
export SPT_CACHE_DIR=/workspace/.stable_pretraining
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
```

Verified local hardware during this setup:

```text
1x NVIDIA H100 80GB HBM3
```

The current machine exposes one H100. If a three-H100 node is available later,
use the wave schedule below.

## Variant Mapping

The requested "1 cube / 2 cube / 3 cube" datasets map to:

| Requested name | OGBench environment | Visual dataset |
| --- | --- | --- |
| 1 cube | `cube-single-v0` / `visual-cube-single-v0` | `visual-cube-single-play-v0.npz` |
| 2 cube | `cube-double-v0` / `visual-cube-double-v0` | `visual-cube-double-play-v0.npz` |
| 3 cube | `cube-triple-v0` / `visual-cube-triple-v0` | `visual-cube-triple-play-v0.npz` |

OGBench also includes larger cube variants such as `quadruple` and `octuple`.
They are not part of this six-model request.

## Trajectories Vs Transitions

Important correction: these are not one-million-trajectory datasets.

The verified counts are:

| Visual dataset | Trajectories / episodes | Usable transitions / timesteps | Raw NPZ rows | Why raw rows are larger |
| --- | ---: | ---: | ---: | --- |
| `visual-cube-single-play-v0` | 1,000 | 1,000,000 | 1,001,000 | 1 terminal dummy row per episode |
| `visual-cube-double-play-v0` | 1,000 | 1,000,000 | 1,001,000 | 1 terminal dummy row per episode |
| `visual-cube-triple-play-v0` | 3,000 | 3,000,000 | 3,003,000 | 1 terminal dummy row per episode |

The converter drops terminal dummy rows. The SWM loader then turns each HDF5
episode into sliding training windows. With `num_steps=4` and `frameskip=5`,
the verified SWM window counts are:

| HDF5 | SWM windows |
| --- | ---: |
| `visual_cube_single_play_v0.h5` | 981,000 |
| `visual_cube_double_play_v0.h5` | 981,000 |
| `visual_cube_triple_play_v0.h5` | 2,943,000 |

This is the basis for optimizer-step estimates. It is not evidence for
"10,000 trajectories for 10 epochs." In this repo:

- DINO/PreJEPA config has `trainer.max_epochs: 10`.
- `10000` appears as a checkpoint cadence in LeWM/YAM-style launch configs.
- The recommended serious-run control is a fixed step budget, e.g.
  `MAX_STEPS=100000`, not "full 10 epochs" for every model.

## Dataset Sizes And Local Paths

Official Berkeley compressed-size context:

| Dataset | Train `.npz` | Val `.npz` | Usable transitions | Episodes |
| --- | ---: | ---: | ---: | ---: |
| `cube-single-play-v0` | 245 MB | 24 MB | 1.0M | 1,000 |
| `cube-double-play-v0` | 284 MB | 28 MB | 1.0M | 1,000 |
| `cube-triple-play-v0` | 957 MB | 96 MB | 3.0M | 3,000 |
| `visual-cube-single-play-v0` | 1.6 GB | 161 MB | 1.0M | 1,000 |
| `visual-cube-double-play-v0` | 1.6 GB | 161 MB | 1.0M | 1,000 |
| `visual-cube-triple-play-v0` | 4.8 GB | 495 MB | 3.0M | 3,000 |

Downloaded official visual train files:

```text
/workspace/datasets/ogbench_npz/visual-cube-single-play-v0.npz
/workspace/datasets/ogbench_npz/visual-cube-double-play-v0.npz
/workspace/datasets/ogbench_npz/visual-cube-triple-play-v0.npz
```

Converted SWM HDF5 files:

```text
/workspace/datasets/ogbench_swm/visual_cube_single_play_v0.h5
/workspace/datasets/ogbench_swm/visual_cube_double_play_v0.h5
/workspace/datasets/ogbench_swm/visual_cube_triple_play_v0.h5
```

Verified HDF5 sizes on disk:

| HDF5 | Size |
| --- | ---: |
| `/workspace/datasets/ogbench_swm/visual_cube_single_play_v0.h5` | 12.68 GB |
| `/workspace/datasets/ogbench_swm/visual_cube_double_play_v0.h5` | 12.78 GB |
| `/workspace/datasets/ogbench_swm/visual_cube_triple_play_v0.h5` | 38.56 GB |

Verified schemas:

| Variant | `pixels` | `action` | `proprio` / `observation` | Episodes |
| --- | --- | --- | --- | ---: |
| 1 cube | `(1000000,64,64,3) uint8` | `(1000000,5) float32` | `(1000000,41) float32` | 1,000 |
| 2 cube | `(1000000,64,64,3) uint8` | `(1000000,5) float32` | `(1000000,54) float32` | 1,000 |
| 3 cube | `(3000000,64,64,3) uint8` | `(3000000,5) float32` | `(3000000,67) float32` | 3,000 |

`proprio` and `observation` are currently raw `qpos+qvel`, not the scaled
OGBench state-observation vectors. Base LeWM configs load only `pixels` and
`action`; DINO/PreJEPA loads `pixels`, `proprio`, and `action`.

## Verification Commands

HDF5 and SWM-window verification:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel
PYTHONPATH=/workspace/src/Pantheon-SWM/stable-worldmodel python - <<'PY'
from pathlib import Path
import h5py
import stable_worldmodel as swm

base = Path('/workspace/datasets/ogbench_swm')
for variant in ['single', 'double', 'triple']:
    path = base / f'visual_cube_{variant}_play_v0.h5'
    print(f'\n{variant}: {path}')
    with h5py.File(path, 'r') as f:
        print('pixels', f['pixels'].shape, f['pixels'].dtype)
        print('action', f['action'].shape, f['action'].dtype)
        print('proprio', f['proprio'].shape, f['proprio'].dtype)
        print('episodes', len(f['ep_len']), 'transitions', int(f['ep_len'][:].sum()))
    ds = swm.data.load_dataset(
        str(path),
        num_steps=4,
        frameskip=5,
        keys_to_load=['pixels', 'action'],
    )
    sample = ds[0]
    print('windows', len(ds))
    print('sample pixels', tuple(sample['pixels'].shape), sample['pixels'].dtype)
    print('sample action', tuple(sample['action'].shape), sample['action'].dtype)
PY
```

Auth and machine checks:

```bash
gh auth status
python - <<'PY'
import wandb
api = wandb.Api(timeout=10)
print(api.viewer.username)
PY
python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().whoami())
PY
nvidia-smi
df -h /workspace
```

Current auth evidence:

- GitHub CLI is logged in as `YashDagade`.
- W&B API is usable from `/root/.netrc` as `yash21-dagade`.
- Public Hugging Face downloads work, and DINOv2 downloaded successfully, but
  no local HF CLI token was found for `whoami`. Log in with `hf auth login`
  before private uploads or authenticated Hub operations.

## Conversion

SWM cannot directly read the official OGBench `.npz` files because SWM HDF5
datasets need `ep_len` and `ep_offset`.

Conversion script:

```text
scripts/ogbench/convert_cube_npz_to_swm_h5.py
```

Commands:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel
python scripts/ogbench/convert_cube_npz_to_swm_h5.py \
  /workspace/datasets/ogbench_npz/visual-cube-single-play-v0.npz \
  /workspace/datasets/ogbench_swm/visual_cube_single_play_v0.h5 --overwrite
python scripts/ogbench/convert_cube_npz_to_swm_h5.py \
  /workspace/datasets/ogbench_npz/visual-cube-double-play-v0.npz \
  /workspace/datasets/ogbench_swm/visual_cube_double_play_v0.h5 --overwrite
python scripts/ogbench/convert_cube_npz_to_swm_h5.py \
  /workspace/datasets/ogbench_npz/visual-cube-triple-play-v0.npz \
  /workspace/datasets/ogbench_swm/visual_cube_triple_play_v0.h5 --overwrite
```

Known converter caveats:

- The converter accesses arrays from compressed NPZ files and can require large
  memory for visual triple; this succeeded on the current machine but is not a
  streaming compressed-NPZ converter.
- HDF5 pixel chunks are large enough that random short-window reads have
  noticeable read amplification. Treat dataloader throughput as a scheduling
  risk.

## Offline Trajectory Examples

Generate MP4 examples:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel
python scripts/ogbench/make_cube_examples.py \
  --h5-dir /workspace/datasets/ogbench_swm \
  --out-dir /workspace/datasets/ogbench_examples
```

Dataset scratch outputs:

```text
/workspace/datasets/ogbench_examples/ogbench_1cube_example.mp4
/workspace/datasets/ogbench_examples/ogbench_2cube_example.mp4
/workspace/datasets/ogbench_examples/ogbench_3cube_example.mp4
```

Repo-local SSH-friendly outputs:

```text
outputs/ogbench_cube_offline_trajectory_examples/
docs/assets/ogbench_cube_offline_trajectory_examples/
```

Important files:

- `ogbench_1cube_offline_trajectory.mp4`
- `ogbench_2cube_offline_trajectory.mp4`
- `ogbench_3cube_offline_trajectory.mp4`
- `ogbench_cube_1v2v3_side_by_side.mp4`
- `ogbench_cube_contact_sheet.png`
- `ogbench_1cube_offline_trajectory.gif`
- `ogbench_2cube_offline_trajectory.gif`
- `ogbench_3cube_offline_trajectory.gif`
- `ogbench_cube_1v2v3_side_by_side.gif`
- `index.html`

The contact sheet is the easiest SSH/Codex preview. The GIFs avoid needing a
video player. To view the gallery from a laptop:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel/outputs/ogbench_cube_offline_trajectory_examples
python -m http.server 8765
```

Then from the laptop:

```bash
ssh -L 8765:localhost:8765 <host>
```

Open:

```text
http://localhost:8765/index.html
```

## Six Target Models

The target checkpoint directories are:

| Family | Variant | Portable checkpoint directory |
| --- | --- | --- |
| DINO/PreJEPA | 1 cube | `/workspace/.stable_worldmodel/checkpoints/ogbench_cube_single_dino/` |
| DINO/PreJEPA | 2 cube | `/workspace/.stable_worldmodel/checkpoints/ogbench_cube_double_dino/` |
| DINO/PreJEPA | 3 cube | `/workspace/.stable_worldmodel/checkpoints/ogbench_cube_triple_dino/` |
| LeWM | 1 cube | `/workspace/.stable_worldmodel/checkpoints/ogbench_cube_single_lewm/` |
| LeWM | 2 cube | `/workspace/.stable_worldmodel/checkpoints/ogbench_cube_double_lewm/` |
| LeWM | 3 cube | `/workspace/.stable_worldmodel/checkpoints/ogbench_cube_triple_lewm/` |

Each portable checkpoint directory should contain:

```text
config.json
weights_step_<N>.pt
```

Lightning resume checkpoints are separate and live under:

```text
/workspace/.stable_pretraining/runs/<date>/<time>/<run_id>/checkpoints/
```

## Optimizer Step Estimates

Training uses `train_split: 0.9`, so optimizer-step estimates use 90 percent of
the SWM windows.

| Family | Batch | 1 cube steps/epoch | 2 cube steps/epoch | 3 cube steps/epoch |
| --- | ---: | ---: | ---: | ---: |
| DINO/PreJEPA | 64 | ~13.8k | ~13.8k | ~41.4k |
| LeWM | 256 | ~3.4k | ~3.4k | ~10.3k |

DINO/PreJEPA has `max_epochs: 10`, which would be roughly:

- Single/double: ~138k optimizer steps.
- Triple: ~414k optimizer steps.

For deadline control, use a fixed step budget such as `MAX_STEPS=100000`
instead of assuming full 10-epoch training for every model.

## Smoke Tests

Run smoke tests before any long job. These are short and are safe to run before
launching full pretraining.

LeWM smoke:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel
PYTHONPATH=/workspace/src/Pantheon-SWM/stable-worldmodel \
STABLEWM_HOME=/workspace/.stable_worldmodel \
SPT_CACHE_DIR=/workspace/.stable_pretraining \
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 \
python scripts/train/lewm.py \
  data=ogb_cube_single_visual \
  output_model_name=ogbench_cube_single_lewm_smoke \
  subdir=ogbench_cube_single_lewm_smoke \
  trainer.devices=1 trainer.accelerator=gpu trainer.precision=bf16-mixed \
  ++trainer.max_steps=2 trainer.max_epochs=1 \
  ++trainer.limit_train_batches=2 ++trainer.limit_val_batches=0 \
  loader.batch_size=8 loader.num_workers=2 num_workers=2 \
  checkpoint.every_n_train_steps=1 checkpoint.epoch_interval=0 \
  wandb.enabled=false
```

DINO/PreJEPA smoke:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel
PYTHONPATH=/workspace/src/Pantheon-SWM/stable-worldmodel \
STABLEWM_HOME=/workspace/.stable_worldmodel \
SPT_CACHE_DIR=/workspace/.stable_pretraining \
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 \
python scripts/train/prejepa.py \
  dataset_name=/workspace/datasets/ogbench_swm/visual_cube_single_play_v0.h5 \
  output_model_name=ogbench_cube_single_dino_smoke \
  subdir=ogbench_cube_single_dino_smoke \
  frameskip=5 batch_size=4 num_workers=2 \
  trainer.devices=1 trainer.accelerator=gpu trainer.precision=bf16-mixed \
  ++trainer.max_steps=2 trainer.max_epochs=1 \
  ++trainer.limit_train_batches=2 ++trainer.limit_val_batches=0 \
  trainer.strategy=auto \
  checkpoint.every_n_train_steps=1 checkpoint.epoch_interval=0 \
  wandb.enabled=false
```

Prior smoke evidence:

- LeWM reached real GPU train steps, wrote portable `.pt`, and the saved model
  loaded with `swm.wm.utils.load_pretrained`.
- DINO/PreJEPA reached real GPU train steps, wrote portable `.pt`, and the saved
  model loaded with `swm.wm.utils.load_pretrained`.

## Full Training: 1x H100 Sequential

Use this only after the user explicitly approves long training.

The sequential launcher queues all six jobs on one GPU:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel
SESSION=ogbench_cube_pretrain_$(date -u +%Y%m%dT%H%M%SZ) \
MAX_STEPS=100000 \
LEWM_BATCH_SIZE=256 \
DINO_BATCH_SIZE=64 \
NUM_WORKERS=12 \
WANDB_ENABLED=true \
WANDB_ENTITY=yash21-dagade-duke-university \
WANDB_PROJECT=pantheon-swm \
LOG_DIR=/workspace/datasets/ogbench_logs \
./scripts/ogbench/launch_cube_pretraining.sh
```

To generate the exact launcher script without starting jobs:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel
DRY_RUN=true MAX_STEPS=100000 ./scripts/ogbench/launch_cube_pretraining.sh
```

Monitor:

```bash
nvidia-smi
tail -f /workspace/datasets/ogbench_logs/<run>.log
find /workspace/.stable_worldmodel/checkpoints -maxdepth 2 -name 'weights_step_*.pt' -print
```

The launcher uses `tmux` when available and falls back to `nohup` with a PID
file when `tmux` is not installed.

## Full Training: 3x H100 Wave Schedule

Use this only on a machine with at least three visible H100s and only after the
user explicitly approves long training.

Recommended schedule:

1. Finish data prep, validation, and examples.
2. Run smoke tests.
3. Wave 1: train three DINO/PreJEPA models in parallel.
4. Wave 2: train three LeWM models in parallel.

Wave 1, DINO/PreJEPA:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel
export PYTHONPATH=/workspace/src/Pantheon-SWM/stable-worldmodel:${PYTHONPATH:-}
export STABLEWM_HOME=/workspace/.stable_worldmodel
export SPT_CACHE_DIR=/workspace/.stable_pretraining
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p /workspace/datasets/ogbench_logs

CUDA_VISIBLE_DEVICES=0 python scripts/train/prejepa.py \
  dataset_name=/workspace/datasets/ogbench_swm/visual_cube_single_play_v0.h5 \
  output_model_name=ogbench_cube_single_dino \
  subdir=ogbench_cube_single_dino_${STAMP} \
  frameskip=5 batch_size=64 num_workers=12 \
  trainer.devices=1 trainer.accelerator=gpu trainer.precision=bf16-mixed \
  ++trainer.max_steps=100000 trainer.max_epochs=1000 trainer.strategy=auto \
  checkpoint.every_n_train_steps=10000 checkpoint.epoch_interval=0 \
  wandb.enabled=true wandb.config.entity=yash21-dagade-duke-university \
  wandb.config.project=pantheon-swm \
  wandb.config.name=ogbench_cube_single_dino_${STAMP} \
  wandb.config.id=ogbench_cube_single_dino_${STAMP} \
  2>&1 | tee /workspace/datasets/ogbench_logs/ogbench_cube_single_dino_${STAMP}.log &

CUDA_VISIBLE_DEVICES=1 python scripts/train/prejepa.py \
  dataset_name=/workspace/datasets/ogbench_swm/visual_cube_double_play_v0.h5 \
  output_model_name=ogbench_cube_double_dino \
  subdir=ogbench_cube_double_dino_${STAMP} \
  frameskip=5 batch_size=64 num_workers=12 \
  trainer.devices=1 trainer.accelerator=gpu trainer.precision=bf16-mixed \
  ++trainer.max_steps=100000 trainer.max_epochs=1000 trainer.strategy=auto \
  checkpoint.every_n_train_steps=10000 checkpoint.epoch_interval=0 \
  wandb.enabled=true wandb.config.entity=yash21-dagade-duke-university \
  wandb.config.project=pantheon-swm \
  wandb.config.name=ogbench_cube_double_dino_${STAMP} \
  wandb.config.id=ogbench_cube_double_dino_${STAMP} \
  2>&1 | tee /workspace/datasets/ogbench_logs/ogbench_cube_double_dino_${STAMP}.log &

CUDA_VISIBLE_DEVICES=2 python scripts/train/prejepa.py \
  dataset_name=/workspace/datasets/ogbench_swm/visual_cube_triple_play_v0.h5 \
  output_model_name=ogbench_cube_triple_dino \
  subdir=ogbench_cube_triple_dino_${STAMP} \
  frameskip=5 batch_size=64 num_workers=12 \
  trainer.devices=1 trainer.accelerator=gpu trainer.precision=bf16-mixed \
  ++trainer.max_steps=100000 trainer.max_epochs=1000 trainer.strategy=auto \
  checkpoint.every_n_train_steps=10000 checkpoint.epoch_interval=0 \
  wandb.enabled=true wandb.config.entity=yash21-dagade-duke-university \
  wandb.config.project=pantheon-swm \
  wandb.config.name=ogbench_cube_triple_dino_${STAMP} \
  wandb.config.id=ogbench_cube_triple_dino_${STAMP} \
  2>&1 | tee /workspace/datasets/ogbench_logs/ogbench_cube_triple_dino_${STAMP}.log &

wait
```

Wave 2, LeWM:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel
export PYTHONPATH=/workspace/src/Pantheon-SWM/stable-worldmodel:${PYTHONPATH:-}
export STABLEWM_HOME=/workspace/.stable_worldmodel
export SPT_CACHE_DIR=/workspace/.stable_pretraining
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p /workspace/datasets/ogbench_logs

CUDA_VISIBLE_DEVICES=0 python scripts/train/lewm.py \
  data=ogb_cube_single_visual \
  output_model_name=ogbench_cube_single_lewm \
  subdir=ogbench_cube_single_lewm_${STAMP} \
  trainer.devices=1 trainer.accelerator=gpu trainer.precision=bf16-mixed \
  ++trainer.max_steps=100000 loader.batch_size=256 num_workers=12 \
  checkpoint.every_n_train_steps=10000 checkpoint.epoch_interval=0 \
  wandb.enabled=true wandb.config.entity=yash21-dagade-duke-university \
  wandb.config.project=pantheon-swm \
  wandb.config.name=ogbench_cube_single_lewm_${STAMP} \
  wandb.config.id=ogbench_cube_single_lewm_${STAMP} \
  2>&1 | tee /workspace/datasets/ogbench_logs/ogbench_cube_single_lewm_${STAMP}.log &

CUDA_VISIBLE_DEVICES=1 python scripts/train/lewm.py \
  data=ogb_cube_double_visual \
  output_model_name=ogbench_cube_double_lewm \
  subdir=ogbench_cube_double_lewm_${STAMP} \
  trainer.devices=1 trainer.accelerator=gpu trainer.precision=bf16-mixed \
  ++trainer.max_steps=100000 loader.batch_size=256 num_workers=12 \
  checkpoint.every_n_train_steps=10000 checkpoint.epoch_interval=0 \
  wandb.enabled=true wandb.config.entity=yash21-dagade-duke-university \
  wandb.config.project=pantheon-swm \
  wandb.config.name=ogbench_cube_double_lewm_${STAMP} \
  wandb.config.id=ogbench_cube_double_lewm_${STAMP} \
  2>&1 | tee /workspace/datasets/ogbench_logs/ogbench_cube_double_lewm_${STAMP}.log &

CUDA_VISIBLE_DEVICES=2 python scripts/train/lewm.py \
  data=ogb_cube_triple_visual \
  output_model_name=ogbench_cube_triple_lewm \
  subdir=ogbench_cube_triple_lewm_${STAMP} \
  trainer.devices=1 trainer.accelerator=gpu trainer.precision=bf16-mixed \
  ++trainer.max_steps=100000 loader.batch_size=256 num_workers=12 \
  checkpoint.every_n_train_steps=10000 checkpoint.epoch_interval=0 \
  wandb.enabled=true wandb.config.entity=yash21-dagade-duke-university \
  wandb.config.project=pantheon-swm \
  wandb.config.name=ogbench_cube_triple_lewm_${STAMP} \
  wandb.config.id=ogbench_cube_triple_lewm_${STAMP} \
  2>&1 | tee /workspace/datasets/ogbench_logs/ogbench_cube_triple_lewm_${STAMP}.log &

wait
```

## Checkpoint Handoff

Portable checkpoints are produced by `stable_worldmodel.wm.utils.save_pretrained`
and loaded by `stable_worldmodel.wm.utils.load_pretrained`.

For sharing with a collaborator, package the portable checkpoint directories:

```bash
mkdir -p /workspace/datasets/ogbench_checkpoint_exports
tar -C /workspace/.stable_worldmodel/checkpoints -czf \
  /workspace/datasets/ogbench_checkpoint_exports/ogbench_cube_swm_checkpoints.tar.gz \
  ogbench_cube_single_dino \
  ogbench_cube_double_dino \
  ogbench_cube_triple_dino \
  ogbench_cube_single_lewm \
  ogbench_cube_double_lewm \
  ogbench_cube_triple_lewm
```

The collaborator needs:

- This code branch or a commit with the same model class/config changes.
- The checkpoint directory containing `config.json` and `weights_*.pt`.
- Compatible Python dependencies.
- HF network/cache access for DINO/PreJEPA backbone reconstruction.

Example load:

```python
import stable_worldmodel as swm

model = swm.wm.utils.load_pretrained(
    'ogbench_cube_single_dino/weights_step_100000.pt',
    cache_dir='/workspace/.stable_worldmodel',
)
model.eval()
```

Planner/evaluation code that calls `swm.wm.utils.load_pretrained(cfg.policy)`
should use the portable checkpoint path, not a Lightning `.ckpt`.

## Resume Notes

There are two checkpoint formats:

- Portable inference checkpoints:
  `/workspace/.stable_worldmodel/checkpoints/<model>/weights_step_<N>.pt`.
- Lightning/training resume checkpoints:
  `/workspace/.stable_pretraining/runs/.../checkpoints/last.ckpt`.

The portable `.pt` format is the right artifact for inference and handoff. It
does not include optimizer/scheduler/trainer state.

The current LeWM and PreJEPA scripts instantiate `spt.Manager` with a
`ckpt_path` under the portable checkpoint directory only if a matching
`<output_model_name>_weights.ckpt` file exists. A robust long-run resume
workflow should be smoke-tested before relying on it. Practical options:

- Keep the full `/workspace/.stable_pretraining/runs/.../checkpoints/last.ckpt`
  directories for any serious run.
- If a job is interrupted, inspect the latest `last.ckpt` and verify a short
  resume before restarting a 100k-step run.
- Treat portable `.pt` checkpoints as inference handoff artifacts, not training
  resume artifacts.

## Known Risks

- Long jobs are currently paused by instruction; do not start them until the
  user approves.
- Current machine exposes one H100. The three-H100 schedule requires a different
  machine or allocation.
- HF CLI is not locally logged in. Public model downloads worked, but private HF
  upload/download workflows require `hf auth login`.
- HDF5 random-window dataloading is not especially fast. A data-only benchmark
  measured roughly:
  - LeWM batch 256: 325-367 windows/s for raw batches.
  - DINO/PreJEPA batch 64: 231-349 windows/s for raw batches.
- The converter stores `proprio`/`observation` as raw `qpos+qvel`, not scaled
  OGBench state observations.
- The converter is not a fully streaming compressed-NPZ converter and can have
  high memory use.
- The launch/data config files are normally hidden by broad `.gitignore`
  patterns; force-add them or add ignore exceptions when committing.
- Reusing the same `output_model_name` overwrites same-named portable step files
  on repeated runs. Use unique names if preserving multiple attempts matters.

## Pre-Launch Checklist

Before any long job:

- Confirm user approval to launch long training.
- Confirm no stale training process is active:

  ```bash
  ps -eo pid,etime,cmd | grep -E 'python .*scripts/train/(lewm|prejepa)\.py|ogbench_cube_pretrain' | grep -v grep || true
  ```

- Confirm GPU count and memory:

  ```bash
  nvidia-smi
  ```

- Confirm `PYTHONPATH` points to this repo:

  ```bash
  python - <<'PY'
  import stable_worldmodel
  print(stable_worldmodel.__file__)
  PY
  ```

- Confirm HDF5 existence and SWM window counts.
- Confirm MP4/contact-sheet examples exist.
- Confirm GitHub and W&B auth.
- Log into HF if authenticated Hub operations are needed.
- Run the LeWM and DINO/PreJEPA smoke tests above.
- Check disk:

  ```bash
  df -h /workspace
  ```

Current status at doc update time:

- HDF5 data verified.
- Examples and SSH-friendly previews verified.
- LeWM and DINO/PreJEPA smoke tests had previously passed.
- No long training job is active.
