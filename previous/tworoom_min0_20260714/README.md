# tworoom_min0_20260714 — TwoRoom min0 + LIPv4 campaign (2026-07-14/15)

Campaign of record for: (1) the tandem min0 perfect card on TwoRoom, (2) the
introduction of **LIPv4** (`--arch v4`, kind `lip4`) as the code default, and
(3) the triple-perfect LIPv4 recipe. Ran on pod a (4×H100, deleted after
archival — see `../pod_archive/poda_20260715/MANIFEST.md`).

## Read these first

| file | content |
|---|---|
| `WRITEUP_tworoom_min0.md` | full writeup: env/WM/protocol, min0 + LIPv4 architecture with equations, tandem training, every hyperparameter, all sweep rounds, lessons |
| `RESULTS_tworoom_scores.md` | every score: per-cell matrices for LIPv4 / min0 / old-v1 anchor / Latent+CEM / random |
| `make_scores_md.py` | regenerates RESULTS from `results/summary_tworoom.csv` (no hand-typed numbers) |

**Headline:** LIPv4 recipe `--arch v4 --amax 2.2 --max-delta 12` (+ tandem
warm start + schedules) = **100.0 on all 12 cells for all 3 training seeds,
plain deploy** (3600/3600 episodes). Canonical checkpoints are ALSO installed
in the repo: `../stable-worldmodel/checkpoints/tworoom_lip4/` (with recipe
README).

## Directory map

| path | content |
|---|---|
| `results/summary_tworoom.csv` | raw per-cell success rates, every eval of the campaign (~340 rows incl. rejected restart-deploy variants) |
| `actors/` | all 39 trained actors: `trm_min0_*` (gated min0 arms + sched seeds), `trm_v4_*` (round A: amax/head-scale), `trm_v4b_*` (round B: amax bracket, mw03, 6k), `trm_v4c_*` (round C: md12 winner, pc05, ax22 s3-5) |
| `metrics/` | each actor's tandem teacher value (`*_value.pt`, MRN quasimetric) |
| `logs/` | driver logs + per-arm training logs (E_final curves) + phase1 data-build log |
| `code_snapshot/` | exact code that ran: train_lip_ac.py, train_lip.py, lip.py (with LIPv4), train_metric.py, env.py + all drivers (phase1, min0_v2, lip4, lip4b, lip4c, a20ext, scores) |
| `failure_videos/` | keyframe strips + mp4s of the two wall-trap failures that motivated max-delta 12 |
| `make_scores_md.py` | report generator |

## Key facts (details in the writeup)

- Eval protocol: replay eval from `tworoom_play.lance`, n=50/cell, 12 cells =
  {std, hard-cross-wall} × {h25, h50} × eval seeds {42,43,44}. LIP evals are
  pure single-pass (restarts=1). Restart-based deploys were explored and
  rejected by directive (rows kept in the CSV: `card_v4a20{r8,rm4,n025}-*`).
- Dataset is deterministic: `collect_play.py --episodes 1000 --seed 7`
  regenerates it byte-equivalently in ~4 min (statistically identical to the
  2026-07-10 campaign's; task draws re-rolled — see writeup §1.4).
- Prior TwoRoom campaign (sequential v1 LIP, perfect card, protocol origin):
  `../tworoom_lip_20260710/`.
- Ops lesson that made this campaign fast: cap `OMP_NUM_THREADS` (=18) for
  the CPU-bound env evals → ~40× wall-clock speedup at 12-way concurrency.
