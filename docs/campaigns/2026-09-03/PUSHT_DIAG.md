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

## E1 -- oracle initialisation (all three actors, draws 42/43/44; success %, means over draws)

| actor | zero-init K=8 (`rlp`) | CEM plan, K=0 (control) | **CEM plan + K=8** | CEM (`cem_latent`) | K=64 |
|---|---|---|---|---|---|
| s0 | 68.0 | 80.0 | **68.7** | 79.3 | 38.7 |
| s1 | 64.0 | 79.3 | **68.7** | 78.7 | 9.3 |
| s2 | 64.7 | 79.3 | **69.3** | 79.3 | 30.0 |
| mean | 65.6 | 79.6 | **68.9** | 79.1 | 26.0 |

Paired over the 9 (actor, draw) cells: the K=0 control agrees with
`cem_latent` on 41-47 of 50 episodes per cell (implementation check passed).
Refinement FROM the CEM plan fixes 3-11 of the mode-A episodes per cell but
loses 2-15 episodes CEM had solved (sum over cells: fixes 55, loses 76).

**Verdict: the refiner cannot hold a good plan.** Handed CEM's plan, eight
learned iterations lower the critic energy (E 11 -> 7.5-8.2, below the
zero-init endpoint of 8.7-9.1) while real success drops from ~79 to ~69. The
imagined objective improves, the outcome worsens: the gradient-based update
walks into critic error that CEM's coarse population search never reaches.
Mode A is therefore not "the basin was never found"; it is "the descent
direction is exploitable".

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

## E3 -- WM-error attribution (actor s0, draws 42/43/44)

Imagined-vs-real latent divergence ||z_imag_t - z_real_t|| over the first plan
(5 blocks x 5 steps), rolling the frozen WM from the real start frames on the
EXECUTED actions of each arm (recorded rollouts) and on the expert's actions
from the same start rows (h5). Means over episodes; block 5 = plan end.

| actions | draw 42 | draw 43 | draw 44 | n per draw |
|---|---|---|---|---|
| expert (dataset) | 1.35 2.06 2.68 3.26 **3.74** | 1.14 1.65 2.19 2.64 **3.13** | 1.29 1.78 2.18 2.89 **3.64** | 50 |
| CEM (latent cost) | 1.45 2.76 4.52 5.55 **6.01** | 1.42 2.12 2.37 4.15 **5.32** | 2.60 3.57 4.22 5.31 **6.40** | 16 / 15 / 17 |
| RLP (K=8) | 1.93 3.85 5.16 5.46 **5.77** | 2.02 2.86 2.99 3.55 **4.30** | 1.93 3.46 4.73 5.61 **6.56** | 25 / 20 / 34 |

Caveat: the recorder drops episodes shorter than 25 steps and successful
episodes terminate early, so the planner rows are biased toward long
(failing) episodes; the expert rows are all 50 tasks.

Reading: the WM is ~1.7x worse off the expert distribution, and **equally so
under CEM's and RLP's actions**. The world model does not single out the
refiner's plans; both planners drive it into the same error band. So the
RLP-vs-CEM gap is not "RLP finds WM holes CEM avoids" -- it is the OBJECTIVE:
RLP descends the learned critic energy E, CEM optimises the raw latent L2
distance, and E1 showed E improving while reality worsens. The exploitable
component is the critic evaluated on imagined states (both planners' imagined
states are equally wrong; only the critic-driven descent turns that error into
a confident bad plan).

## What this points to
1. Do not extend K at deploy; if more descent is wanted it must be trained in
   (K=24 was flat in the counterstrike campaign, so this is not the lever).
2. E3 says the WM error is planner-agnostic, so PushT Dyna would lift CEM
   and RLP alike and is not the gap-closer. The exploitable part is the
   critic on imagined states. Decisive next test (E4, one eval-only job):
   CEM on the critic energy (`core/value=metric`, replacement mode) vs CEM on
   latent L2 on the same draws. If critic-CEM also drops to ~65, the critic
   is the defect (fix: critic training on planner-visited imagined states /
   a conservative critic); if critic-CEM stays ~79, the defect is the
   gradient descent path itself (fix: an anchor toward the latent objective
   or toward CEM/expert plans in the refiner's training, `planner.bc_weight`).
3. In either case the July "value exonerated" reading was for the old
   stack; under config B the objective is where the gap lives.

## Critic experiments launched 2026-09-08 (E4, E5, E6a, E7, E8)

- **E4 -- objective swap for the sampler** (`pusht-e4-s{0,1,2}`): CEM driven by
  the co-trained critic (`cem_value`), CEM driven by the offline TD teacher
  (`cem_tdvalue`), and `cem_latent` on the same draws. Critic-CEM ~65 => the
  critic ranks plans wrongly on its own; ~79 => only the gradient path is at
  fault.
- **E6a -- deploy-time critic swap for the refiner** (same jobs,
  `rlp_tdvalue`): the s0-s2 actors run with the OFFLINE teacher as the deploy
  critic instead of the co-trained one. Measures how much of RLP's outcome is
  the co-trained critic's drift (TwoRoom precedent: teacher swap +0.9).
- **E5 -- real-vs-imagined calibration** (`critic_probe.py`, POST_ONLY on
  `pusht-diag3-s0`): critic energy of the REAL window reached at the end of
  the first plan vs the IMAGINED window on the same executed actions, for
  RLP and CEM rollouts, co-trained and teacher critics; optimism gap by
  outcome; AUC of E_real / E_imag for predicting success. Separates "critic
  wrong on real states" from "critic wrong on imagined states".
- **E8 -- velocity channel** (same job, rlp probes): energy of the imagined
  4-frame window vs a static stack of the terminal frame. A mode-A-specific
  gap says the refiner's advantage is manufactured in the window's velocity
  components.
- **E7 -- epistemic disagreement** (`pusht-teacher-s10/s11`: two more TD
  teachers on the same cache, different seeds; analysis follows): std across
  teachers of the energy at RLP's imagined terminal states, mode-A vs ok.
  High disagreement on mode-A => the exploit sits in epistemic uncertainty;
  a pessimistic ensemble critic is then the fix candidate.
