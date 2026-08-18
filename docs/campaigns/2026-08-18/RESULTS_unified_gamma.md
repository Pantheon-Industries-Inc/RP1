# Unified-γ cross-environment campaign — 2026-08-18

Question: can RLP's critic discount be unified across environments, and at
what cost per cell? The unified-config proposal argued γ=0.98/n=50 everywhere
("ordering-invariance; finite unreachable target"); TwoRoom falsified it
(§10/§10a of [RESULTS_tworoom_improvement.md](../2026-08-17/RESULTS_tworoom_improvement.md):
h100 collapses to 66–73 because V saturates at 1/(1−γ)=50 and grad V goes
flat). The revised proposal — γ as a horizon budget, γ ≈ 1 − 1.5/d_max, i.e.
**γ=0.99** with worst-case d_max≈150 — is the headline arm here.

## Design (user decisions pinned 2026-08-18)

- **Train everything ONCE at amax=2.5, max_delta=12** — no per-cell clip, no
  horizon-matched actor pairs. One actor per (γ × vnorm × seed), evaluated at
  both horizons from the same checkpoint. amax is adapted only at deployment
  (post-hoc eval pass with the checkpoint's top-level `amax` rewritten; the
  solver reads deploy amax exclusively from the checkpoint —
  `overlays/lip.py`, no config path exists).
- **Arms: γ ∈ {0.98, 0.99, 1.0} × vnorm ∈ {none, log}, n-step 50.** One γ
  point is each environment's bespoke value (TwoRoom 1.0; Cube/Reacher 0.98),
  so the fleet is a 3-point dose–response per environment with the bespoke
  value as an internal anchor.
- **Cells:** TwoRoom LeWM/LeJEPA (h25+h100, held-out SPLIT=1), OGBench Cube
  LeWM (h25+h100, held-out by construction), Reacher LeWM/LeJEPA (latched
  first-hit tau 0.1 and 0.05). Train seeds 0/1/2, report draws 42/43/44,
  50 episodes/cell.
- Everything else stays at each cell's established recipe (mw/lr anchors,
  steps/batch, teacher expectile/steps), so γ and the clip are the only
  deltas against banked per-cell numbers.

Infrastructure: `scripts/sky/unigamma/launch_unig_fleet.sh` (smoke → fleet),
`GRID=unig` in `tworoom_g98_rescue.yaml` (TwoRoom + Cube; `CU_GAMMA` threads
the cube discount — `TR_GAMMA` never reached cube), and
`reacher_gamma.yaml` (`RS_GAMMA`/`RS_VNORM` threading; the vnorm actor port
`overlays_reacher_vnorm/` is now wired). WM checkpoints come from the
volume stage (`stage_lfs.yaml` + `stage_probe.yaml`, `cp -rL`); the GitHub
LFS budget is exhausted and the W&B-artifact restore is fallback only.

## Banked context (already measured, TwoRoom LeWM/LeJEPA h100 @ a1.8/md20)

From jobs 7913–7919, 8022–8024 (collector: `collect_unig.yaml`,
TAG_GLOB `rlp-tw-g9*-20260818`):

| critic | raw E (ctrl) | vnorm=log |
|---|---|---|
| γ=1.0 / n50 (bespoke) | 88.67 (seed 3) | **97.33** (seed 3) |
| γ=0.98 / n50 (unified proposal) | 66.22 ± 21.7 | 81.78 ± 3.7 |
| γ=0.98 / n100 (≡ n200, see below) | 85.11 ± 3.3 | 87.33 ± 2.7 |
| γ=0.996 / n50 ("0.98 per block") | 87.11 ± 1.0 | 91.33 ± 2.4 |

Three observations that shape expectations for the fleet:

1. **The 0.98 collapse is a window problem as much as a discount problem.**
   Widening the exact-label window (n100) recovers +19 points and cuts seed
   sd 7×; moving the ceiling out via per-block discounting (γ=0.996, ceiling
   250 steps) recovers +21. Both rescue arms stay 6–10 points below the
   γ=1.0+vlog record — the discount still costs real accuracy at range.
2. **n=100 and n=200 trained bit-identical actors** (all 18 per-draw scores
   equal): TwoRoom's episode length caps sampled goal deltas at ~100
   primitive steps, so the n-step window saturates. n>100 is not a separate
   arm on this environment.
3. γ=0.99/n50 (ceiling 100) sits between the falsified 0.98 (ceiling 50) and
   the rescue 0.996 (ceiling 250). The horizon-budget rule predicts it
   should survive h100-with-drift; the §10 saturation mechanism predicts its
   d=100 slope (0.37/step) clears the critic noise floor that 0.98's
   0.13/step did not. This is exactly what the fleet measures.

PLDM matrix arms (banked, same protocol, PLDM base): γ=0.996 h100 ctrl
88–92 across seeds; γ=0.98/n100 ~85–87; γ=1.0 reference partial (job
cancelled mid-run; 88.0 on the completed seed). Not this campaign's target
cells but consistent with the LeJEPA ordering.

## Fleet (in flight)

18 jobs × H200:4 = 72 GPUs, p1: 9 TwoRoom (γ × seed sharded), 3 Cube
(γ, seeds packed), 6 Reacher (γ × vnorm, seeds in-job). Smoke canaries
(one per environment, γ=0.99) gate the fleet.

Tags: `rlp-tw-unig-{g98,g99,g100}-s{0,1,2}-20260818`,
`rlp-cu-unig-{g98,g99,g100}-20260818`,
`rlp-re-unig-{g98,g99,g100}-{none,log}-20260818`.

## Results

(pending — filled as jobs land)

### TwoRoom LeWM (h25 / h100)

### OGBench Cube LeWM (h25 / h100)

### Reacher LeWM (tau 0.1 / 0.05, latched first-hit)

### Deploy-amax pass

(post-hoc: checkpoint `amax` rewritten to {recipe, 3.0}; training amax fixed
at 2.5)
