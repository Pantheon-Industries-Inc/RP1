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
pixi run tool tool=cache_latents wm=<checkpoint> dataset=<data> out=<cache.pt>
```

Every command creates a run at `logs/YYYY-MM-DD/HH-MM-SS/` (with a numeric
suffix when two runs start in the same second). The run contains the resolved
`config.yaml`, lifecycle `metadata.json`, structured `run.log`, and dedicated
`checkpoints/`, `metrics/`, `videos/`, `artifacts/`, `tracking/`, and `stages/`
directories. Reusable datasets and latent caches remain outside a run and are
supplied through their corresponding data config.

Set `logging.wandb.mode` to `online`, `offline`, or `disabled`. W&B's local
state is contained under the run's `tracking/` directory.

## Reproducing the measurements

The audit's `[measured]` numbers came from a historical 4×H100 campaign. Its
pod-specific orchestration remains available through Git history on `main`,
but is deliberately not part of the current application tree.
The FLOP/shape figures need only
`assets/core/world_model/lewm_cube/config.json`.
See [Stable-WorldModel compatibility](docs/stable-worldmodel-compatibility.md)
for the pin rationale and the upstream migration checklist.
