# PushT diagnostics E1-E3 (2026-09-08)

Setup: config-B PushT actors (`pusht-v2l05-s{0,1,2}-20260902`, w4, K=8,
md20, acr 0.5, n-step 1), official LeWM base, held-out draws 42/43/44
(50 tasks each), one K=8 pass per decision. Conditions evaluated on the SAME
draws: `rlp` (zero init, K=8), `rlp_ceminit` (K=8 from a CEM plan on the
latent cost, 300x30, top-30), `rlp_ceminit_k0` (the CEM plan through the LIP
path, K=0: control), `rlp_k64` (64 iterations of the learned update),
`cem_latent`, `noop`. Code: `LIPSolver(init_mode="cem", iters_override=K)`,
per-iteration probes (`E_iters`, `lat_iters`), rollout recording
(`RECORD=1`), in-job analysis `scripts/pusht_diag/analyze.py`. Jobs
`pusht-diag3-s*-20260908` (full); the numbers below are from the salvaged
first launch (`pusht-diag2-*`, killed by the legacy volume's disk quota):
draw 42 for all three actors, draw 43 for actor 2.

## E1 -- oracle initialisation (success %, 50 tasks per cell)

| actor | draw | zero-init K=8 (`rlp`) | CEM plan, K=0 | **CEM plan + K=8** | CEM (`cem_latent`) | K=64 |
|---|---|---|---|---|---|---|
| s0 | 42 | 66 | 80 | **66** | 78 | 52 |
| s1 | 42 | 60 | 78 | **72** | 78 | 14 |
| s2 | 42 | 62 | 80 | **64** | 78 | 24 |
| s2 | 43 | 78 | 84 | **88** | 84 | 44 |

Paired, draw 42 (three actors): CEM-init refinement fixes 4 / 7 / 5 of the
10 / 11 / 10 mode-A episodes (RLP fails, CEM succeeds) but LOSES 10 / 7 / 9
episodes CEM had solved. The K=0 control agrees with `cem_latent` on 47 / 46 /
47 of 50 episodes (implementation check passed).

**Verdict: the refiner cannot hold a good plan.** Handed CEM's plan, eight
learned iterations lower the critic energy (E 11 -> 7.5-8.2, below the
zero-init endpoint of 8.7-9.1) while real success drops from ~79 to ~67. The
imagined objective improves, the outcome worsens: the gradient-based update
walks into critic/WM error that CEM's coarse population search never
reaches. Mode A is therefore not "the basin was never found"; it is
"the descent direction is exploitable".

## E2 -- refinement audit (critic energy E_k, t=0 replan, means per class)

| actor | class | E_0 | E_2 | E_4 | E_8 (deployed) | E_64 | imagined final latent distance at k=8 |
|---|---|---|---|---|---|---|---|
| s0 | RLP ok (n=33) | 21.1 | 16.3 | 12.8 | 8.7 | 12.4 | 5.4 |
| s0 | mode A (n=10) | 24.1 | 18.9 | 16.8 | 14.6 | 16.6 | 8.9 |
| s1 | RLP ok (n=30) | 20.0 | 15.5 | 12.1 | 7.8 | 19.0 | 4.7 |
| s1 | mode A (n=11) | 24.6 | 19.2 | 14.6 | 10.1 | 21.7 | 6.3 |
| s2 (2 draws) | RLP ok (n=70) | 21.5 | 16.7 | 12.9 | 9.1 | 15.4 | 5.3 |
| s2 (2 draws) | mode A (n=16) | 23.9 | 18.4 | 14.7 | 11.5 | 18.1 | 7.1 |

- Descent through k=8 is monotone (91-95 % of steps) and NOT converged:
  only 35-46 % of the total drop has happened by k=2 and E_7 -> E_8 still
  falls by ~0.7. K=8 is truncated descent.
- Continuing the learned update past its trained horizon diverges: by k=64
  the energy is back at 12-20 (monotone fraction 0.27-0.38) and success
  collapses to 14-52. The iterate is a K=8-unrolled network, not a fixed-point
  solver; more iterations of it are out of distribution. This is why the
  counterstrike K ladder saturated: K must be trained, not extended.
- Mode-A episodes start harder (E_0 24 vs 21) and end higher (E_8 14.6 /
  10.1 / 11.5 vs 8.7 / 7.8 / 9.1); their imagined final distance is 6-9 vs
  5, i.e. the planner's own scores flag them at t=0 (reproduces the
  counterstrike self-flagging result).

## E3 -- WM-error attribution
Pending (`pusht-diag3-*`): imagined-vs-real latent divergence over the first
plan under RLP's, CEM's and the expert's actions from the same starts.

## What this points to (before E3)
1. Do not extend K at deploy; if more descent is wanted it must be trained in
   (K=24 was flat in the counterstrike campaign, so this is not the lever).
2. The refiner's imagined objective is exploitable along its own path:
   candidates are (a) Dyna for PushT with the planner's own rollouts (the
   cube recipe: config-B collection, both offsets, two iterations) so the WM
   is correct where the refiner goes; (b) an anchor toward CEM/expert plans
   during training (`planner.bc_weight` exists, untested at B); (c) training
   the co-critic on the refiner's own final plans (value expansion is on at
   1.0 already, so the critic still ends up wrong there -- E3 decides between
   WM error and critic error).
