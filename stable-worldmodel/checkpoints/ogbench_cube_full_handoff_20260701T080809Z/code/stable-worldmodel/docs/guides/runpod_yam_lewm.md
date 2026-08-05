---
title: Runpod YAM LeWM Dataset
summary: Preparing the real YAM teleoperation dataset for Stable World-Model and LeWorldModel training on a Runpod network volume.
---

# Runpod YAM LeWM Dataset

This guide records the Runpod setup used to turn the real YAM teleoperation
data into a single Stable World-Model compatible HDF5 file for LeWorldModel
training.

Current status as of 2026-06-06 04:22 UTC: old RealSense conversion completed,
incoming MCAP conversion was intentionally stopped after 631 episodes, the H5
is valid, W&B is authenticated, and the next production run is the ViT-B YAM
LeWM config in `scripts/train/config/lewm_yam.yaml`.

The goal is to keep the simulation training path as unchanged as possible:
use the same LeWM data loader shape as the OGBench Cube config, keep
`frameskip: 5` in the training config, normalize numeric columns in the
training transform, and replace only the dataset with real robot trajectories.

## Runpod Layout

The network volume is mounted at:

```bash
/workspace
```

The project layout on the volume is:

```text
/workspace/
  datasets/yam_lewm/
    raw/
      yam_realsense_202604/
      incoming_rgb_mcap/
    h5/
      yam_ogbcube_real_mosaic224.h5
    logs/
    manifests/
    configs/
    convert_yam_to_swm_h5.py
    run_conversion_loop.sh
    run_lewm_yam_train.sh
    smoke_test_h5.py
  src/
    Pantheon-SWM/
      stable-worldmodel/
      sbatch_scripts/
  venvs/
    stable-worldmodel/
```

Stable World-Model is installed in editable mode from:

```bash
/workspace/src/Pantheon-SWM/stable-worldmodel
```

The environment activation file is:

```bash
source /workspace/datasets/yam_lewm/configs/env.sh
```

That file sets `STABLEWM_HOME=/workspace/.stable_worldmodel` and activates
the shared Python environment.

## Source Data

Two source trees are copied to Runpod:

```text
/mnt/sensitive-xdof/data/s3-replica/pantheon-xdof-teleop-0422-final/yam_realsense_202604
/mnt/robot-pool/robot05-teleop/incoming
```

Only RGB camera, timestamp, robot state, and action/proprioception data are
copied. Depth MKVs are deliberately excluded because they dominate storage and
are not used for the LeWM pretraining run.

The old RealSense episodes contain:

```text
top_camera-images-rgb.mp4
left_camera-images-rgb.mp4
right_camera-images-rgb.mp4
timestamp.npy
left-joint_pos.npy
right-joint_pos.npy
left-gripper_pos.npy
right-gripper_pos.npy
action-left-pos.npy
action-right-pos.npy
metadata.json
```

The incoming episodes contain:

```text
exo_cam-images-rgb.mp4
left_wrist_cam-images-rgb.mp4
right_wrist_cam-images-rgb.mp4
exo_cam-rgb-timestamp.npy
left_wrist_cam-rgb-timestamp.npy
right_wrist_cam-rgb-timestamp.npy
yam_left.mcap
yam_right.mcap
yam_leader_left.mcap
yam_leader_right.mcap
session_meta.json
```

## Transfer

The transfer runs from the source storage machine to the Runpod pod with
`rsync` over SSH:

```bash
tmux attach -t yam_raw_to_runpod
```

The script is:

```bash
/mnt/sensitive-xdof/data/s3-replica/runpod_lewm_yam/rsync_to_runpod.sh
```

It uses `--partial` and `--partial-dir=.rsync-partial`, so it can be rerun if
the connection drops. It also uses `--no-owner --no-group` because Runpod
network volumes are NFS-backed and can reject ownership changes from rsync.

The destination is:

```bash
/workspace/datasets/yam_lewm/raw
```

Monitor transfer progress from the source host:

```bash
tmux capture-pane -pt yam_raw_to_runpod -S -80 | tail -80
```

Monitor copied size from Runpod:

```bash
du -sh /workspace/datasets/yam_lewm/raw
df -h /workspace
```

## HDF5 Conversion

Conversion runs on the Runpod pod in a persistent tmux session:

```bash
tmux attach -t yam_h5_convert
```

The loop entrypoint is:

```bash
/workspace/datasets/yam_lewm/run_conversion_loop.sh
```

The loop repeatedly calls:

```bash
/workspace/datasets/yam_lewm/convert_yam_to_swm_h5.py
```

This lets conversion overlap with rsync. Episodes that are already complete on
the network volume are appended to the HDF5 file. Later loop passes pick up
newly copied episodes. Successful episode ids are recorded in:

```bash
/workspace/datasets/yam_lewm/manifests/conversion_manifest.jsonl
```

The converter supports source selection:

```bash
--sources old
--sources incoming
--sources both
```

During the initial Runpod migration the loop defaults to `CONVERT_SOURCES=old`
so the old RealSense tree can be fully converted before incoming episodes are
appended. This gives a safe cleanup point: after all 3283 old episodes are
`status: ok` in the manifest, `/workspace/datasets/yam_lewm/raw/yam_realsense_202604`
can be removed if extra space is needed for incoming conversion.

The overnight supervisor is:

```bash
/workspace/datasets/yam_lewm/overnight_pipeline.sh
```

It watches the old conversion loop, waits until all 3283 old episodes are in
the manifest, waits for the HDF5 writer to close, runs the HDF5 smoke test,
deletes only the old raw staging tree on Runpod, converts the 1077 complete
incoming episodes, and finishes with both the HDF5 smoke test and a one-batch
LeWM training smoke run.

Monitor it with:

```bash
tmux attach -t yam_overnight_supervisor
tail -f /workspace/datasets/yam_lewm/logs/overnight_pipeline_*.log
```

The final HDF5 path is:

```bash
/workspace/datasets/yam_lewm/h5/yam_ogbcube_real_mosaic224.h5
```

Current partial-training H5 summary:

```text
episodes: 3914
frames: 14248254
sources: 3283 yam_realsense_202604 + 631 incoming_rgb_mcap
size: about 1024 GiB
```

Monitor conversion:

```bash
tail -f /workspace/datasets/yam_lewm/logs/convert_loop_*.log
tail -5 /workspace/datasets/yam_lewm/manifests/conversion_manifest.jsonl
du -sh /workspace/datasets/yam_lewm/h5
```

Do not restart conversion while the H100 is training unless intentionally
trading training time for more converted episodes.

The old tiny baseline and benchmark sessions are intentionally stopped before a
new production launch. Check active sessions with:

```bash
tmux list-sessions
nvidia-smi
```

## Weights & Biases

W&B is available in the Runpod environment:

```bash
source /workspace/datasets/yam_lewm/configs/env.sh
wandb login --relogin
wandb status
```

Paste the API key from `https://wandb.ai/authorize` into the pod terminal.
LeWM enables W&B through Hydra:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel
python scripts/train/lewm.py \
  --config-name lewm_yam \
  data=yam_real \
  wandb.enabled=true \
  wandb.config.entity=yash21-dagade-duke-university \
  wandb.config.project=pantheon-swm \
  wandb.config.name=yam_swm_lewm_mosaic224_vitb_proprio_decoder
```

## HDF5 Schema

The file is a root-level Stable World-Model HDF5 dataset with `ep_len` and
`ep_offset`, matching `stable_worldmodel.data.HDF5Dataset`.

Columns:

| Column | Dtype | Shape per row | Meaning |
| --- | --- | --- | --- |
| `pixels` | `uint8` | `(224, 224, 3)` | Three-camera RGB mosaic |
| `action` | `float32` | `(14,)` | Relative target minus current proprioception |
| `action_absolute` | `float32` | `(14,)` | Absolute recorded leader/target command |
| `proprio` | `float32` | `(14,)` | Current left/right joint and gripper state |
| `observation` | `float32` | `(14,)` | Alias of `proprio` for OGBench-style configs |
| `timestamp` | `float64` | `()` | Camera timestamp in seconds |
| `ep_len` | `int32` | `()` | Episode length |
| `ep_offset` | `int64` | `()` | Episode start row |
| `ep_source` | string | `()` | Source collection |
| `ep_task` | string | `()` | Task/date grouping |
| `ep_id` | string | `()` | Episode directory name |
| `ep_path` | string | `()` | Original path used during conversion |

The `pixels` column is a 2x2 mosaic. The first three tiles contain the three
RGB cameras and the fourth tile is blank. This keeps three-camera context in a
single `pixels` key so the stock LeWM image path can train immediately without
changing the encoder input signature.

Both converted source families use the same HDF5 image layout:

| Source | Raw cameras | H5 mosaic tiles |
| --- | --- | --- |
| `yam_realsense_202604` | `top_camera`, `left_camera`, `right_camera` | top-left, top-right, bottom-left |
| `incoming_rgb_mcap` | `exo_cam`, `left_wrist_cam`, `right_wrist_cam` | top-left, top-right, bottom-left |

An audit on 2026-06-06 checked the middle frame of all 3914 episodes. Every
episode had populated tiles `(top_left, top_right, bottom_left)` and an empty
bottom-right tile:

```text
yam_realsense_202604: 3283 / 3283 episodes are (True, True, True, False)
incoming_rgb_mcap:     631 / 631 episodes are (True, True, True, False)
```

The W&B visual eval panel is different from a raw dataset image. It unwraps
each 2x2 H5 mosaic into a 3-camera strip, then shows column groups:

```text
gt future cam1 cam2 cam3 | encoder recon cam1 cam2 cam3 | wm pred cam1 cam2 cam3
```

Multiple validation clips are stacked vertically. This display format avoids
the blank fourth mosaic tile and makes it easier to compare the same camera
view across the ground-truth future frame and the decoder reconstruction of the
encoder latent. By default it also decodes the world-model predictor latent as
`wm pred`; this is only a visual probe run under `torch.no_grad()` and does not
train extra parameters. Disable it with `timed_eval.show_pred=false` if the
pure encoder reconstruction probe is preferred.

## Action Semantics

The training config follows the OGBench Cube path as closely as possible:

```yaml
dataset:
  frameskip: 5
  keys_to_load:
    - pixels
    - action
    - observation
  keys_to_cache:
    - action
    - observation
  keys_to_merge:
    proprio: proprio
```

The HDF5 stores every frame. Do not pre-apply frameskip during conversion.

Stable World-Model applies frameskip in the dataset reader. Observations are
strided by `frameskip`, while dense actions are reshaped by the base dataset
class. With `frameskip: 5` and a 14-dimensional robot action, a training sample
sees an action width of `5 * 14 = 70`.

The `action` column stores deltas:

```text
action = action_absolute - proprio
```

This mirrors the relative-action convention used by OGBench-style robot
control better than storing raw absolute joint targets. The raw absolute target
is kept in `action_absolute` so future experiments can switch semantics without
rebuilding the raw staging data.

LeWM normalizes non-pixel columns at training time with dataset statistics.
The HDF5 therefore stores physical values, not pre-normalized values.

## Smoke Test

Run this before training:

```bash
source /workspace/datasets/yam_lewm/configs/env.sh
python /workspace/datasets/yam_lewm/smoke_test_h5.py \
  /workspace/datasets/yam_lewm/h5/yam_ogbcube_real_mosaic224.h5
```

Expected sample shapes:

```text
pixels: (4, 3, 224, 224)
action: (4, 70)
observation: (4, 14)
proprio: (4, 14)
```

`action` is `(num_steps, frameskip * action_dim)` because the loader keeps
dense actions between strided observations.

## LeWM Training

The LeWM data config is installed at:

```bash
/workspace/src/Pantheon-SWM/stable-worldmodel/scripts/train/config/data/yam_real.yaml
```

All new training should run through the Pantheon-SWM repo:

```bash
cd /workspace/src/Pantheon-SWM
source /workspace/datasets/yam_lewm/configs/env.sh
```

The main YAM config is:

```bash
/workspace/src/Pantheon-SWM/stable-worldmodel/scripts/train/config/lewm_yam.yaml
```

It uses Stable World-Model's native LeWM implementation with:

| Setting | Value |
| --- | --- |
| Encoder | ViT-B/14, random init |
| Image input | 224x224 three-camera mosaic |
| State latent | 768 |
| Predictor depth | 6 transformer blocks |
| Batch size | 128 |
| Max steps | 450000 |
| Frameskip | 5 in the dataset loader |
| Action width | 70, from 5 dense 14D actions |
| Proprio key | `observation` |
| W&B eval cadence | Start of training, then every 3600 seconds |
| Checkpoint cadence | Portable weights every 10000 train steps |

The launcher also sets:

```bash
SPT_CACHE_DIR=/workspace/.stable_pretraining
```

This keeps stable-pretraining's run metadata and automatic `last.ckpt` files
on the 2 TB Runpod network volume. Do not let those files default to
`/root/.cache/stable-pretraining`, because the container disk is only 20 GB.

Launch a short smoke run:

```bash
SMOKE=1 CONFIG_NAME=lewm_yam ./sbatch_scripts/launch_lewm.sh
```

Launch the production ViT-B run:

```bash
SMOKE=0 ./sbatch_scripts/launch_lewm_big.sh
```

The launcher writes logs under:

```bash
/workspace/datasets/yam_lewm/logs
```

Attach to the run:

```bash
tmux attach -t yam_swm_lewm_vitb_<timestamp>
```

### Model Size and Time Budget

The measured ViT-B YAM model has about 177M trainable parameters with the
proprio fusion, decoder probe, and proprio probe included.

Batch-size benchmark results on the Runpod H100 SXM 80GB:

| Batch | VRAM | Rate | Effective clips/s | Notes |
| --- | --- | --- | --- | --- |
| 64 | about 32 GB | about 5.6 it/s | about 358 | Good, but underuses memory |
| 128 | about 61 GB | about 3.0 it/s | about 384 | Chosen default |
| 160 | about 75 GB | about 2.4 to 2.5 it/s | about 384 to 400 | Fits, but has little memory margin |

The batch-128 train split has about 99660 optimizer steps per epoch. At about
3.1 it/s, a 42 hour wall-clock target is roughly 450000 optimizer steps after
allowing for startup, hourly validation, and checkpoint overhead. The current
config uses 450000 steps, which is roughly 4.5 epochs.

### Proprio, Decoder, and Losses

The real LeWM state encoder uses both vision and proprioception:

```text
pixels -> ViT CLS -> pixel_emb
observation -> proprio MLP -> proprio_emb
concat(pixel_emb, proprio_emb) -> fusion MLP -> emb
```

The predictor still predicts the fused latent `emb`; it does not predict raw
images or raw proprioception as its primary objective.

The decoder and proprio heads are probes:

- The decoder receives detached latents and trains only the decoder head. It
  is used for W&B reconstruction and predicted-future videos.
- The proprio head receives detached predicted latents and trains only the
  proprio probe head. `loss.proprio` is not IDM; it is a future-proprio
  decodability probe.
- These probe losses do not send gradients into the encoder or predictor.

The optimized LeWM path remains latent prediction plus SIGReg. Probe losses
make the monitoring heads useful without turning the world model into a
reconstruction or supervised proprio model.

### Offline Eval

YAM does not currently have an OGBench-style simulator/reset loop, so there is
no online CEM/MPC success-rate validation. The hourly eval is offline:

- Average validation losses on a bounded validation slice.
- Latent variance and covariance diagnostics.
- Decoder videos for context, target future, reconstruction, and predicted
  future.
- Proprio probe MSE when the probe head is enabled.

Online CEM/MPC eval should be added later against a real robot loop or a YAM
simulator.

## Troubleshooting

Check that Stable World-Model is editable:

```bash
source /workspace/datasets/yam_lewm/configs/env.sh
python -m pip show stable-worldmodel
python - <<'PY'
import stable_worldmodel
print(stable_worldmodel.__file__)
PY
```

The Runpod environment was pinned to avoid known import mismatches:

```text
pyarrow==20.0.0
datasets==2.21.0
```

If conversion stops because free space is low, inspect:

```bash
df -h /workspace
du -sh /workspace/datasets/yam_lewm/raw
du -sh /workspace/datasets/yam_lewm/h5
tail -20 /workspace/datasets/yam_lewm/manifests/conversion_manifest.jsonl
```

If rsync is interrupted, rerun the rsync script. It is safe to resume because
partial files are kept and the conversion manifest prevents duplicate HDF5
episodes.
