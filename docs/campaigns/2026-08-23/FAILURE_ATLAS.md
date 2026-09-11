# Failure Atlas — why each cell fails where it fails (2026-08-23)

Question (user): diagnose ALL failure cases in TwoRoom, OGBench Cube, and
Reacher under the unified config. Method: (1) paired per-task re-evaluation
of every banked depth-2 FINAL actor and deep-capacity actor with persisted
per-task success arrays — task draws depend only on the draw seed, so every
actor faces IDENTICAL tasks, making failure sets directly comparable;
(2) per-episode closest-approach dumps for reacher latched evals;
(3) a local CPU mechanistic deep-dive on TwoRoom (checkpoints from W&B,
value landscapes on the room lattice, ridge-probe-decoded plans).

Data: `rlp-atlas-{tw,cu}[-p]-{d2,dc}-20260823` (rawlogs/),
`rlp-atlas-re-*-20260823` (results/per_episode.jsonl),
`docs/figures/tworoom_failures/` (mechanistic figures + DIAGNOSIS.md).

## Verdict: three environments, three unrelated mechanisms

| env | mechanism | key numbers |
|---|---|---|
| TwoRoom | actor seed-lottery; ZERO task-intrinsic hardness | 0 always-fail tasks in every config; deep-capacity LeWM: 147/150 never-fail |
| Cube | task-intrinsic hard core + a LeWM-specific extension | 12 tasks (8%) fail everything; LeWM core 19–22 vs PLDM 13 (PLDM ⊂ LeWM) |
| Reacher | universal near-miss; terminal-precision bound | every miss reaches 0.05–0.13 rad; zero planning failures |

### TwoRoom — the lottery is a WM-hallucination basin (actor-only)

Full mechanism in `docs/figures/tworoom_failures/DIAGNOSIS.md`. Summary:
bad PLDM seeds converge to a ~80%-constant plan family — a fixed
adversarial input the frozen WM hallucinates as reaching the goal
(imagined direction-cosine +0.86/+0.91) while the executed motion points
away (−0.46/−0.57). The critic is exonerated: all seeds' value landscapes
track the BFS geodesic at Spearman rho 0.983–0.992; cross-swap is causal
(good actor x bad critic = 100%, bad actor x good critic = 19%). Because
the training loss is imagined energy through the frozen WM, both basins
are equally good minima — training-level failure, invisible to loss
curves, only partially recoverable by ES.
- Explains the deep-capacity crater on tw-PLDM (bigger refiners find the
  exploit more easily) and the futility of every critic-side stabilizer.
- Rollout-free canary: plan-constancy 0.77–0.80 (bad) vs 0.26–0.39 (good)
  + cos(plan-sum, imagined direction). Candidate unified fix (untested):
  anti-constancy regularizer on the refiner across the (z0, zg) batch.
- Atlas corroboration: dc tw-pldm has 0 always-fail tasks but only 27/150
  never-fail — failures scatter broadly across tasks, as a fixed plan
  family that succeeds only where geometry happens to cooperate would.

### Cube — a hard core no configuration solves

- 12 of 150 h100 tasks fail for EVERY seed of EVERY config (both bases,
  both capacities): environment/task-intrinsic floor of 8%.
- LeWM's own always-fail core is 19–22 tasks and is a superset of PLDM's
  13 → ~6–9 tasks are LeWM-representation-specific. Cube-PLDM (smaller
  core) indeed outscores LeWM at h100 under the unified config (82.1 vs
  81.7 median) despite LeWM's h25 advantage.
- Ceiling arithmetic: LeWM h100 max ≈ 87 if it won its entire
  seed-dependent band; paper 82.4 and unified 81.7 both sit mid-band.
  "Cube-LeWM to 90" is NOT achievable under this eval for this method
  class; the honest target is the contested band (~2–5 pts).
- h25 hard core (10–14 tasks, also cross-base consistent) is DISJOINT
  from the h100 core — short- and long-horizon hardness are different
  task properties.
- Hard-task frame dump (start/goal images of the 24-task h100 union +
  14-task h25 core): job `dump-hard-frames` →
  `rlp-atlas-hardframes-20260823/hardcore_h{25,100}.png` (visual
  identification of what these segments contain — pending at writing).

### Reacher — misses are near-misses, bounded by w=1

Per-episode closest-approach (all four actor families, tau=0.05):
every failed episode still reaches within 0.05–0.13 rad of the goal;
most misses sit at 0.05–0.10. No episode ever fails to approach. So
tau=0.05 failures are terminal precision, not planning — consistent with
the hardcoded w=1 critic being velocity-blind (w=3 was worth +6/+9 at
tau=0.05; that gap is a representation limit, not a tuning gap).
Deep-capacity approaches SLOWER (first-hit ~20–26 steps vs ~9–15 for
depth-2), which is its -4 at tau=0.05.

## Method notes / traps encountered (for reruns)

- Eval digest logs carry only per-draw means; per-task arrays only exist
  in eval_wm stdout → EVAL_RAWDIR=volume persists them (committed).
- Final-won checkpoints store ck["value"] as a TRAINING-NODE path;
  snapshot-won ones embed it. Imports must repath to the persisted
  values/ copy — and only after probing with load_metric (embedding a
  value FILE's dict breaks the solver; repathing a legitimate snapshot
  embed silently swaps critics). Committed as the import fixup chain.
- tw ES-final actors are md=12 provenance (the FINAL-table config line
  says md=20; that applies to cube/reacher — [args-ok] enforces truth).
- Wiped-and-reused tags keep stale digest logs → health checks can read
  false FAIL while rawlogs are good; trust rawlogs + wandb.
