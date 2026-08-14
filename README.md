# RLP — Reinforcement Learned Planning with Latent World Models

Reference implementation of **RLP** (paper: *Reinforcement Learned Planning
with Latent World Models*): a goal-conditioned quasimetric critic plus a
neural plan-refiner, trained on top of any frozen pretrained latent world
model, that replaces hand-designed planners (CEM, MPPI, gradient descent)
with a learned search procedure — 9 world-model rollouts per decision instead
of 3,000–9,000.

Naming note: the paper's **RLP** planner is called **LIP** (Learned Iterative
Planner) throughout the code and checkpoints; they are the same method. The
specific refiner architecture the paper presents is generation 4 — configs
say `architecture: v4` (`configs/core/planner/lip.yaml`), checkpoints record
`kind: lip4`, and campaign records call it **LIPv4**. All of these name the
paper's residual plan refiner (Eq. 10); earlier generations (`lip`…`lip3`)
remain loadable by the solver for old checkpoints but are not the paper's
method.

## Start here

- **[`docs/replication/REPLICATION_RLP.md`](docs/replication/REPLICATION_RLP.md)**
  — the paper replication command sheet: paper-component → code map,
  one-command RLP training, per-table evaluation commands, world-model bases,
  the Dyna round, and the data-split protocol.
- [`docs/replication/cube/REPLICATION_CUBE.md`](docs/replication/cube/REPLICATION_CUBE.md)
  — bit-level replication of the tracked OGBench Cube LeWM base.
- [`docs/lip/README_lip.md`](docs/lip/README_lip.md) — method notes and recipe
  lessons; [`docs/campaigns/`](docs/campaigns/) — dated experiment records.
- [`docs/PARALLELIZATION_ANALYSIS.md`](docs/PARALLELIZATION_ANALYSIS.md) — historical
  serialization/throughput audit (some citations target pre-refactor paths).

## Replicating RLP: LeWM and PLDM

Both pretrained OGBench Cube world models are tracked in-tree via Git LFS —
`assets/core/world_model/lewm_cube` (LeWM) and
`assets/core/world_model/pldm_cube` (the authors' PLDM checkpoint, converted
1:1 into the LeWM key layout; converter: `pixi run tool tool=convert_pldm`).
Replication is therefore self-contained: fetch the public dataset once, then
train and evaluate per base.

```bash
pixi run tool tool=fetch_dataset dataset=ogb_cube   # ~20 GiB, public
```

**Train RLP** (`model=rlp` runs the paper's full stack — latent caching,
offline quasimetric value learning with TD + hindsight relabeling + expectile
regression, then actor-critic planner training through the frozen world
model). The only setting that differs between the two bases is the residual
clip range `amax`:

```bash
# LeWM base (amax 1.6)
pixi run train model=rlp wm=assets/core/world_model/lewm_cube \
    dataset=$RLP_DATA_HOME/datasets/ogb_cube_single.lance \
    name=cube_lewm planner.amax=1.6

# PLDM base (amax 4.5)
pixi run train model=rlp wm=assets/core/world_model/pldm_cube \
    dataset=$RLP_DATA_HOME/datasets/ogb_cube_single.lance \
    name=cube_pldm planner.amax=4.5
```

Each run writes `checkpoints/planner.pt` (the RLP refiner), `value_td`
(offline critic), and `value_ac` (co-trained teacher) into its
`logs/<date>/<time>/` directory. Reusable latent caches land in
`$RLP_DATA_HOME/caches/` (rerun with `skip=[cache,subsample,actions]` to iterate
on recipes without re-encoding).

**Evaluate** the trained planner against the paper's baselines
(`model=lewm` / `model=pldm` selects the base; horizons:
`evaluation.goal_offset_steps=25 evaluation.budget=50` for h25, `=100`/`=200`
for h100; report seeds 42/43/44):

```bash
pixi run eval model=lewm core/solver=lip core.solver.actor_path=<planner.pt>  # RLP, 9 rollouts
pixi run eval model=lewm core/solver=cem                                      # CEM,  9,000 rollouts
pixi run eval model=lewm core/solver=mppi                                     # MPPI, 9,000 rollouts
pixi run eval model=lewm core/solver=adam                                     # Adam, 3,000 rollouts
pixi run eval model=lewm core/policy=no_move                                  # no-op floor (skill normalization)
pixi run eval model=pldm core/solver=lip core.solver.actor_path=<planner.pt>  # same grid on the PLDM base
```

To run the baselines under the learned value objective instead of latent
distance, add `core/value=metric core.value.checkpoints=[<value_td>]`. The
complete per-table command sheet (including TwoRoom and Reacher) is in
[REPLICATION_RLP.md](docs/replication/REPLICATION_RLP.md).

## Layout

| path | what |
|---|---|
| `src/rlp/core/` | planning stack: solvers (`solver/` — LIP/RLP, CEM, MPPI, Adam), planner network (`planner/`), value functions (`value/`), world-model backends (`world_model/`), shared differentiable unroll (`rollout.py`) |
| `src/rlp/train/` | trainers: `rlp.py` (composed replication pipeline), `lip_ac.py` (RLP actor-critic), `metric.py` (offline value), `lewm.py` (world-model base) |
| `src/rlp/eval/` | `world_model.py` — the table-producing evaluation driver |
| `src/rlp/data/` | frozen-latent caching (`LatentCache`, `encode_dataset`) |
| `src/rlp/environment/` | dataset-evaluation world behavior (reset/record hooks) |
| `src/rlp/tools/` | Hydra-configured data preparation (dataset fetch, latent caches, action h5, TwoRoom collection) |
| `configs/` | Hydra configuration tree mirroring the `src/rlp/` subsystems |
| `docs/` | replication sheets, campaign records, method notes |
| `assets/core/world_model/` | all eight pretrained world models (Git LFS, ~69 MiB each): cube `lewm_cube/`+`pldm_cube/` and their Dyna-finetuned variants `*_cube_dyna/`, TwoRoom `lejepa_tworoom/`+`pldm_tworoom/`, Reacher `lejepa_reacher/`+`pldm_reacher/` |
| `logs/` | generated run directories, grouped by local date and start time |

## Fresh-machine setup

A clone contains the source and Git LFS pointers, but it cannot bootstrap the
host tools needed to retrieve and run them. Install Git, Git LFS, `curl`, and
Pixi on the host first. Pixi then supplies Python 3.13, PyTorch, MuJoCo,
OGBench, Lance/HDF5, FFmpeg, the Hugging Face and W&B CLIs, and all development
tools. Do **not** install a separate Python, Conda environment, FFmpeg, Hugging
Face CLI, or W&B CLI.

The supported platforms are macOS 14+ on Apple Silicon and Linux x86-64. A
full development checkout plus both Pixi environments and the Cube dataset can
use more than 25 GiB, so start with at least 35 GiB free.

### 1. Install host prerequisites

On macOS, verify the Apple command-line tools are present. If the first command
fails, run the installer and wait for it to finish:

```bash
xcode-select -p
xcode-select --install
```

Install [Homebrew](https://brew.sh/) if it is not already available, then Git
and [Git LFS](https://git-lfs.com/):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install git git-lfs
git lfs install
```

On Ubuntu/Linux:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git git-lfs
git lfs install
```

Git LFS currently recommends version 3.7.1 because it contains a security
update. If the distribution package is older, use the current package from
[the official Git LFS downloads](https://git-lfs.com/). Install Pixi with its
[official installer](https://pixi.prefix.dev/latest/installation/):

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"

git --version
git lfs version
pixi --version
```

Linux GPU machines also need a working host NVIDIA driver. Pixi installs the
locked CUDA-enabled PyTorch wheel; it does not install the kernel driver, and a
separate CUDA toolkit is not required.

```bash
nvidia-smi
```

### 2. Clone, materialize LFS files, and install

Install Git LFS before cloning so checkout can replace the model pointer with
the actual checkpoint. For a private GitHub repository, configure an SSH key
or GitHub credential first; GitHub CLI is optional and is not used by RLP.

```bash
git clone https://github.com/armin-sommer/RLP_original.git
cd RLP_original

git lfs pull
git lfs fsck
ls -lh assets/core/world_model/lewm_cube/weights_epoch_22.pt

pixi install --all
pixi run -e dev hooks
```

The checkpoint should be about 69 MiB, not a small LFS pointer. Runtime-only
users can run `pixi install -e default` instead of installing all environments.

### 3. Choose data storage and fetch the public dataset

Reusable datasets and caches default to `~/.cache/rlp`. To place them on a
larger disk, set `RLP_DATA_HOME` in every shell used for fetching, training, or
evaluation (or persist the export in that shell's startup file):

```bash
export RLP_DATA_HOME=/absolute/path/to/rlp-data
mkdir -p "$RLP_DATA_HOME"

pixi run tool tool=fetch_dataset dataset=ogb_cube dry_run=true
pixi run tool tool=fetch_dataset dataset=ogb_cube
```

The registered Cube dataset is public, so Hugging Face authentication is not
required. If a public request unexpectedly returns `401 Unauthorized`, remove a
stale environment token (`unset HF_TOKEN`) and retry anonymously.

W&B defaults to disabled. Online tracking is the only case that needs login
(`pixi run -e default -x wandb login`); set `logging.wandb.mode` to `online`,
`offline`, or `disabled`.

### 4. Verify the installation

CI is intentionally disabled in this research tree, so contributors must run
the complete local checks before merging:

```bash
git lfs fsck
pixi lock --check
pixi run -e default -x python -m pip check
pixi run -e dev -x python -m pip check
pixi run -e dev check
pixi run -e dev -x python -m pip wheel --no-deps --wheel-dir /tmp/rlp-wheel .
pixi run tool tool=fetch_dataset dataset=ogb_cube dry_run=true
```

After fetching the dataset, exercise the real local checkpoint and evaluation
path with one CPU episode:

```bash
HF_HUB_OFFLINE=1 pixi run eval model=lewm evaluation.num_episodes=1 runtime.device=cpu
```

## Commands

All maintained commands use Hydra overrides; there are no `argparse`
entrypoints. Every command creates a run at `logs/YYYY-MM-DD/HH-MM-SS/` with
the resolved `config.yaml`, lifecycle `metadata.json`, structured `run.log`,
and dedicated `checkpoints/`, `metrics/`, `videos/`, `artifacts/`,
`tracking/`, and `stages/` directories.

```bash
pixi run train model=rlp wm=<ckpt> dataset=<lance>   # full RLP pipeline (cache -> value -> planner)
pixi run train model=lip_ac cache=<fs5> cache_td=<fs1> h5=<h5> wm=<ckpt>  # planner stage alone
pixi run train model=metric cache=<fs1> learner=td   # offline value alone
pixi run train model=lewm                             # LeWM world model, OGBench Cube
pixi run eval  model=lewm|pldm [core/solver=lip|cem|mppi|adam] [core/policy=no_move]
pixi run tool  tool=fetch_dataset dataset=ogb_cube    # fetch the public Cube dataset (~20 GiB)
pixi run tool  tool=cache_latents wm=<ckpt> dataset=<lance> out=<cache.pt>
pixi run tool  tool=convert_pldm src=<pldm.pt> dst=<out.pt>  # authors' PLDM -> LeWM key layout
```

## `stable-worldmodel` is an installed dependency

The framework is pinned to `stable-worldmodel[train]==0.1.1`. RLP's planner,
value stack, and world-model backends are first-party code under `src/rlp/`;
the small behavior deltas required by the paper campaigns (checkpoint
compatibility, image-resized evaluation, recording, the no-move baseline,
gradient-solver portability) live beside the RLP subsystem that owns each
behavior. See
[docs/stable-worldmodel-compatibility.md](docs/stable-worldmodel-compatibility.md)
for the pin rationale and the upstream migration checklist.

## What is deliberately excluded

Large training datasets and generated checkpoints are not part of the source
tree. The Cube expert set is ~20 GB and lives on HF
(`galilai-group/ogb_cube_single`). The repository's prerequisite cube LeWM
checkpoint is the exception: its 69 MiB payload is tracked through Git LFS.
Built wheels include the RLP package and Hydra configs but intentionally omit
that checkpoint; wheel-only users must supply an explicit local checkpoint
path or supported Hugging Face identifier.

PLDM base checkpoints are external (converted from the authors' release into
the LeWM key layout); see the replication sheet. The pre-refactor campaign
harnesses (including the Dyna collection scripts) remain available through Git
history on `main`.
