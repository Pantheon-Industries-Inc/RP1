# Value_Metric_LeWM — LIP / Dyna research tree

Private snapshot of the LIP (Learned Iterative Planner) and Dyna closed-loop campaigns,
packaged so a systems/kernel engineer can read the performance audit **and** run the code the
audit refers to.

## Start here

**[`PARALLELIZATION_ANALYSIS.md`](PARALLELIZATION_ANALYSIS.md)** — the serialization &
parallelization audit. Every claim is anchored to `file:line` against the tree in this repo, so
references resolve directly. Numbers are labelled `[measured]` vs `[derived]`.

Headline: LIP does 16.9 GFLOP per decision and takes 560 ms on an H100 (**0.004% of peak**) — it is
entirely kernel-launch bound. But the planner is invoked twice per episode; the actual wall-clock
sits in a **single-threaded 50-env MuJoCo render loop running the GPU at ~1.5% duty**.

Read §0 (TL;DR), §1 (three regimes — they have *different* bottlenecks), then §7 (ranked fix list).
§9 lists what is still open.

## Layout

| path | what |
|---|---|
| `PARALLELIZATION_ANALYSIS.md` | the audit |
| `stable-worldmodel/` | the framework — **vendored, see below** |
| `Dyna/dyna_harness/` | pod orchestration scripts (collect / mix / fine-tune / gate / ladder) + the fixes from the audit |
| `Dyna/dyna_r1_results/` | measured round-1 ladder timings (`driver_r1_ladder.log` is the source for §5.1) |
| `Dyna/dyna_ckpts/` | Dyna fine-tuned world-model checkpoints |
| `v2WM/` | the base world model (`weights_epoch_22.pt`) + `config.json` — the shape source for every `[derived]` FLOP figure |
| `previous/` | prior campaigns (PushT, TwoRoom, PLDM, HWM, retrains) incl. writeups and videos |
| `failure_analysis/`, `reacher/`, `results_h100/`, `pod_archive/` | supporting artifacts |

## `stable-worldmodel` is vendored, not a submodule

Upstream is public: `galilai-group/stable-worldmodel` @ `b298aa70`.

It is **copied in rather than referenced** because the working tree carries **9 modified + 29
untracked** files, and the untracked set includes `stable_worldmodel/solver/lip.py`,
`scripts/plan/train_lip_ac.py`, and `scripts/plan/config/solver/{lip,hlip}.yaml` — i.e. **the entire
LIP implementation, which exists in no upstream commit.** A submodule pin would ship a tree with no
LIP in it.

> ⚠️ Corollary worth acting on: before this repo existed, that code was a single unversioned copy on
> one laptop. Two root-level `.md` files were lost to a folder reorg during the audit itself.

## What is deliberately excluded

Excluded via `.gitignore` — nothing here is required to read the audit or run the harness:

- `**/.venv/` (1.9 GB regenerable virtualenv), `__pycache__`, `.DS_Store`
- `*.tgz` / `*.tar.gz` / `*.tar.zst` / `*.zip` (3.9 GB of pod snapshots, largely redundant)
- files over GitHub's hard 100 MB per-file limit: `lewm_cube_negatives/` shards,
  `pod_bundle_20260709.tgz`, and the three `weights_step_100000.pt` DINO checkpoints (175 MB each)

The large training datasets are not here either — the expert set is ~20 GB and lives on HF
(`galilai-group/ogb_cube_single`) and on the pods. If you want the excluded binaries too, Git LFS
with a paid data pack or a HF dataset repo is the right vehicle, not this repo.

## Fixes produced by the audit (in `Dyna/dyna_harness/`)

| file | what |
|---|---|
| `egl_device_probe.py`, `egl_device_test.sh` | prove `MUJOCO_EGL_DEVICE_ID` (not `CUDA_VISIBLE_DEVICES`) controls MuJoCo EGL placement |
| `egl_3way_eval_test.sh` | 3-way concurrent-eval reproduction — **still needs one clean run**, see §9 |
| `collect_r1_fixed.sh` | drop-in replacement for `collect_r1.sh`: EGL pinning, no `PAR` latch, per-actor workers, no batch barrier |
| `patch_lip_reuse_traj.py` | env-gated fix for the duplicate rollout at `solver/lip.py:553` (`LIP_REUSE_TRAJ=1`, or `=verify` to assert value-identity) |
| `cache_lance_shard.py` | **bug fix** — `int + list` made 4-way sharded caching impossible; validated to 1.7e-6 against the reference cache |

## Reproducing the measurements

The audit's `[measured]` numbers come from a 4×H100 pod. `Dyna/dyna_harness/*.sh` assume
`/workspace/...` paths and `PYTHONPATH=<repo>/stable-worldmodel`. The FLOP/shape figures need only
`v2WM/config.json`.
