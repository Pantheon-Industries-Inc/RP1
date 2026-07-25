# Pod a (31.24.80.32:15419, 4×H100) — final archive before deletion, 2026-07-15

Pod a hosted two campaigns: **OGBench cube LIP-AC** (2026-07-12 → 07-14,
rounds AC/90/b/g/h/s1-s7 + failure autopsy + negatives collection) and
**TwoRoom min0/LIPv4** (2026-07-14/15). Everything of value is now local;
the pod is CLEAR TO DELETE. Sister pod b (87.120.211.204:18099) is NOT
covered by this manifest — it holds only S7 pod-b screen arms (documented
losers; summary CSVs already merged locally) — deletable at your discretion.

## What is in this folder

| item | size | content |
|---|---|---|
| `lewm_cube_negatives/` | 3.5G | 24 h5 shards: 1050 physics-verified grasp-and-miss + 1050 drops + 201 perturbed-success positives (2301 eps / 150k frames), collected for the planned WM contact fine-tune (FAILURE_AUTOPSY §7). **Checksum-verified against the pod.** Collector: `../../lip_ac_20260712/s7_collect/collect_negatives.py` |
| `logs_poda.tar.gz` | 0.7M | the pod's complete /workspace/logs (939 files: every train/eval log incl. episode_successes arrays, driver logs, both campaigns) |
| `results_poda.tar.gz` | 210M | the pod's complete /workspace/results (all result txts + ALL eval videos of both campaigns + summary CSVs) |
| `code_poda.tar.gz` | 79M | the pod's /workspace/code tree as last run (NOTE: the local repo is STRICTLY AHEAD — it has LIPv4 plus newer local features (replay-prob, init_mode, symmetric); this tar is provenance only, do not restore over the local repo) |
| `drivers/` | 41 files | every /workspace/*.sh driver: cube (chain*, run_ac90*, pipeline, setup_env, download_data) + tworoom (phase1, min0 v1/v2, lip4/b/c, a20ext, scores, r8/rm4/n025) |
| `staging_gd/` | 42K | old pod-side staging copies of lip.py/train_lip_ac.py (superseded; kept for provenance) |

## What went to the campaign folders instead

- `../../tworoom_min0_20260714/` — TwoRoom: all 39 actors + teachers, CSV,
  train logs, code snapshot, failure videos, writeup + scores report.
  Winner md5s verified vs pod (trm_v4c_md12_s0/1/2 + values).
- `../../lip_ac_20260712/actors/` (60) + `metrics/` (61) — ALL cube campaign
  actors/teachers (champion schedamax, min0 s0-s3, every S1-S7 arm, cf_dE
  teacher) — previously pod-only.
- `../../lip_ac_20260712/s7_collect/` — negatives collector + failure
  attribution + gallery harvester scripts and their logs.
- `../../stable-worldmodel/checkpoints/tworoom_lip4/` — canonical LIPv4
  deliverable (3 triple-perfect actors + teachers + TD warm-start + recipe
  README).

## Already local before this archive (verified, not re-fetched)

- WM checkpoints: `stable-worldmodel/checkpoints/tworoom/`
  (weights_epoch_16_partial) and `ogbench_cube_single_v2WM/` (weights_epoch_22).
- TwoRoom TD teacher family: `tworoom_lip_20260710/metrics/td_*.pt` (11 files)
  + old v1 actors.
- Cube failure gallery (4.4M): `lip_ac_20260712/failure_gallery/`.
- All writeups/results markdowns in the two campaign folders.

## NOT fetched (regenerable / re-downloadable) — and how to regenerate

| item | why skipped | regeneration |
|---|---|---|
| `tworoom_play.lance` (dataset) | deterministic | `collect_play.py --episodes 1000 --num-envs 32 --seed 7` (~4 min); script in tworoom_min0_20260714/code_snapshot + drivers/tworoom_phase1.sh |
| tworoom fs1/fs5 caches (60M) | 90 s on GPU | `scripts/trm/cache_latents.py --wm lewm_tworoom --state-key state` + `subsample_cache.py --frameskip 5` |
| cube fs1/fs5 caches (1.9G) | rebuildable | `lip_ac_20260712/cache_cube_full.py` + `merge_caches.py` over the ogbench h5 |
| `lewm_cube_full` h5 (102G) | public re-download | `drivers/download_data.sh` / ogbench release |
| hydra `outputs/`, `s7_bundle.tar`, `env.done` | junk / duplicates | — |
| k12 round-C checkpoints | never existed (trainings timed out pre-save) | rerun round-C k12 arm with a >4h timeout if ever needed |

## Environment (for a future pod)

Bootstrap per `BOOTSTRAP_NEWPOD.md` §2 (torch 2.4.1+cu124, py3.11) +
loguru/tqdm; network volumes reject chown → rsync with
`--no-perms --no-owner --no-group`; absolute /workspace paths for --wm
(relative triggers HF fallback); cap OMP_NUM_THREADS≈18 for env-loop evals.
Actor checkpoints record their teacher's absolute path in `ck["value"]` —
rebind when relocating (one-liner in checkpoints/tworoom_lip4/README.md).
