# Value_Metric_LeWM — LIP / Dyna research tree

Research implementation of LIP (Learned Iterative Planner), learned value
metrics, and Dyna-style closed-loop world-model training.

## Start here

**[`PARALLELIZATION_ANALYSIS.md`](PARALLELIZATION_ANALYSIS.md)** — a historical
serialization and parallelization audit. Source citations target either the
current RLP package or the original campaign harness on Git `main`.

Headline: LIP does 16.9 GFLOP per decision and takes 560 ms on an H100 (**0.004% of peak**) — it is
entirely kernel-launch bound. But the planner is invoked twice per episode; the actual wall-clock
sits in a **single-threaded 50-env MuJoCo render loop running the GPU at ~1.5% duty**.

Read §0 (TL;DR), §1 (three regimes — they have *different* bottlenecks), then §7 (ranked fix list).
§9 lists what is still open.

## Layout

| path | what |
|---|---|
| `PARALLELIZATION_ANALYSIS.md` | the audit |
| `src/rlp/core/` | the RL loop's policy and planning stack (`solver/`, `planner/`), value function (`value/`), world models (`world_model/`), and shared model unroll (`rollout.py`) |
| `src/rlp/environment/` | RLP environment variants and dataset-evaluation world behavior |
| `src/rlp/data/` | frozen-latent caching (`LatentCache`, `encode_dataset`) — kept outside `core/` so probes and cache builders do not import the control stack |
| `src/rlp/train/` | model, metric, planner, online-TD, and composed pipeline trainers |
| `src/rlp/eval/` | world-model, TRM, hard-set, and SCSA evaluation drivers |
| `configs/` | Hydra configuration tree mirroring the corresponding `src/rlp/` subsystems |
| `src/rlp/tools/` | import-safe, Hydra-configured data preparation and latent-cache tools |
| `docs/lip/` | LIP writeup + benchmark results |
| `logs/` | generated run directories, grouped by local date and start time |
| `assets/core/world_model/lewm_cube/` | the prerequisite cube LeWM checkpoint; generated checkpoints remain pipeline outputs and are not kept in the source tree |

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
# Only needed when the repository requires GitHub authentication.
# Install `gh` with `brew install gh` (macOS) or `apt-get install gh` (Ubuntu), then:
gh auth login --git-protocol https --web
gh auth setup-git
```

```bash
git clone https://github.com/armin-sommer/Value_Metric_LeWM.git
cd Value_Metric_LeWM

git lfs pull
git lfs fsck
git lfs ls-files
ls -lh assets/core/world_model/lewm_cube/weights_epoch_22.pt

pixi install --all
pixi run -e dev hooks
```

The checkpoint should be about 69 MiB, not a small LFS pointer. Runtime-only
users can run `pixi install -e default` instead of installing all environments.
Contributors can verify that Pixi installed every command used by this tree:

```bash
pixi run -e default -x python --version
pixi run -e default -x hf version
pixi run -e default -x wandb --version
pixi run -e dev -x pre-commit --version
pixi run -e dev -x ruff --version
pixi run -e dev -x mypy --version
pixi run -e dev -x pytest --version
```

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
required. Pixi already installs the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
Login only for private or gated assets:

```bash
pixi run -e default -x hf auth login
pixi run -e default -x hf auth whoami
```

If a public request unexpectedly returns `401 Unauthorized`, remove a stale
environment token and retry anonymously:

```bash
unset HF_TOKEN
pixi run tool tool=fetch_dataset dataset=ogb_cube dry_run=true
```

W&B defaults to disabled. Online tracking is the only case that needs login:

```bash
pixi run -e default -x wandb login
```

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

## `stable-worldmodel` is an installed dependency

The framework is pinned to `stable-worldmodel[train]==0.1.1`. LIP, TRM, and the
three RLP world-model backends (DINO-WM, State-WM, HWM) remain first-party code
under `src/rlp/`; they were never part of the published framework API. The
small behavior delta required by the campaigns—checkpoint compatibility,
state-only/image-resized evaluation, recording, the no-move baseline, PushT
geometry, and gradient-solver portability—lives beside the RLP subsystem that
owns each behavior under `src/rlp/core/` and `src/rlp/environment/`.

> ⚠️ Corollary worth acting on: before this repo existed, that code was a single unversioned copy on
> one laptop. Two root-level `.md` files were lost to a folder reorg during the audit itself.

## What is deliberately excluded

Excluded via `.gitignore`:

- `**/.venv/` (1.9 GB regenerable virtualenv), `__pycache__`, `.DS_Store`
- `*.tgz` / `*.tar.gz` / `*.tar.zst` / `*.zip` (3.9 GB of pod snapshots, largely redundant)

Large training datasets and generated checkpoints are not part of the source
tree. The expert set is ~20 GB and lives on HF
(`galilai-group/ogb_cube_single`) and on the pods. Git LFS or an artifact/data
registry is the appropriate home for additional binaries.

The repository's prerequisite LeWM checkpoint is the exception: its 69 MiB
payload is tracked through Git LFS. Built wheels include the RLP package and
Hydra configs but intentionally omit that checkpoint; wheel-only users must
supply an explicit local checkpoint path or supported Hugging Face identifier.

## Fixes retained from the audit

| file | what |
|---|---|
| `rlp/core/solver/lip.py` | configurable duplicate-rollout reuse and verification (`core.solver.reuse_trajectory`) |
| `rlp/tools/data/cache_lance_shard.py` | **bug fix** — `int + list` made 4-way sharded caching impossible; validated to 1.7e-6 against the reference cache |

## Commands

All maintained commands use Hydra overrides; there are no `argparse` entrypoints.

```bash
pixi run train model=lewm                          # LeWM, OGBench Cube
pixi run train model=prejepa                       # PreJEPA, OGBench Cube
pixi run eval model=lewm                           # LeWM, OGBench Cube
pixi run eval model=prejepa core.world_model.checkpoint=<checkpoint>
pixi run tool tool=fetch_dataset dataset=ogb_cube  # Fetch the public Cube dataset (~20 GiB)
pixi run tool tool=cache_latents wm=<checkpoint> dataset=<data> out=<cache.pt>
```

The fetch command downloads the pinned public Hugging Face Lance dataset into
`$RLP_DATA_HOME/datasets` (default: `~/.cache/rlp/datasets`), resumes partial
downloads, checks available disk space, and validates the result. The Cube
train/eval configs use that location by default. Inspect the size and target
without downloading with `pixi run tool tool=fetch_dataset dry_run=true`.

Every command creates a run at `logs/YYYY-MM-DD/HH-MM-SS/` (with a numeric
suffix when two runs start in the same second). The run contains the resolved
`config.yaml`, lifecycle `metadata.json`, structured `run.log`, and dedicated
`checkpoints/`, `metrics/`, `videos/`, `artifacts/`, `tracking/`, and `stages/`
directories. Reusable datasets and latent caches remain outside a run and are
supplied through their corresponding data config.

Set `logging.wandb.mode` to `online`, `offline`, or `disabled`. W&B's local
state is contained under the run's `tracking/` directory.

`environment.visualize_info=true` adds diagnostic information to rendered
frames; it does not open a live simulator window. Evaluation videos are saved
under the run's `videos/` directory. A live window on Linux additionally needs
an active X11/Wayland desktop session, which Pixi cannot provision.

## Reproducing the measurements

The audit's `[measured]` numbers came from a historical 4×H100 campaign. Its
pod-specific orchestration remains available through Git history on `main`,
but is deliberately not part of the current application tree.
The FLOP/shape figures need only
`assets/core/world_model/lewm_cube/config.json`.
See [Stable-WorldModel compatibility](docs/stable-worldmodel-compatibility.md)
for the pin rationale and the upstream migration checklist.
