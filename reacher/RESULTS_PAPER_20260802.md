# Reacher: LIP vs sampling planners — paper report

**2026-08-02 · branch `eval-sweep` · pod 212.247.220.107 (6×H200)**

Everything below is measured on this pod under one protocol. Where a number is
inherited from an earlier run it is marked. Where an arm is incomplete or
disputed it says so rather than being omitted.

---

## 1. Protocol

| item | value |
|---|---|
| environment | DM-Control Reacher (LeWM paper config), strict-success env |
| dataset | canonical `reacher.h5` — 10,000 episodes random play, 2,010,000 rows |
| observation | 224×224×3 pixels |
| action | 2-d, **action block = 5 primitive steps** ⇒ block width `a_dim = 10` |
| goal | `eval.goal_offset_steps=25` (goal is the state at t+25) — "h25" |
| evaluation pool | episodes **8000:10000** (`+eval.ep_range=8000:10000`) |
| episodes per cell | `eval.eval_budget=50` |
| replanning | `plan_config.receding_horizon=5` — the **whole** 5-block plan (25 primitive steps) is executed before replanning, for every solver |
| success metric | **held-at-end**: worst joint within τ of the goal *at the final step*, all joints. Headline τ = **0.1 rad**; τ = 0.05 rad also reported |
| reporting seeds | **42, 43, 44, 45, 46, 47** (6 env seeds) |
| selection seeds | **50, 51** — used for every hyperparameter choice; never reported |
| rendering | `MUJOCO_GL=egl`, one `MUJOCO_EGL_DEVICE_ID` pinned per GPU (osmesa costs ~7 pts and is never used) |

**Success convention.** "Held-at-end" is stricter than the latched convention
used in the LeWM paper's own code (which counts an episode successful if the arm
*ever* reaches the goal). Latched numbers from this campaign are not mixed into
the tables below.

---

## 2. World models (bases)

| base | checkpoint | latent | history | notes |
|---|---|---|---|---|
| **LeWM** | `lejepa_reacher` | 192-d | 3 | in-house replication of the paper's LeWM |
| **PLDM** | `pldm_reacher` | 192-d | 3 | |
| DINO-WM | `dinowmnp_reacher` | **75,264-d** | 3 | frozen stock DINOv2-small, 196 patch tokens × 384, **unpooled**. Not in the tables — see §7 |

All three convert from `reacher/{lewm,pldm,dinowm_noprop}.tar.zst` via
`scripts/plan/convert_reacher_bases.py`. World models are **frozen** everywhere;
nothing below fine-tunes them.

---

## 3. The learned value (critic)

One value definition is shared by **Value+CEM**, **Value+Adam**, and the
initialisation of both **LIP** and **PWM**.

| hyper | value |
|---|---|
| form | 3-frame **window** quasimetric, MRN head |
| window frames / lag | 3 / **5** (= action block; consecutive imagined latents are one block apart) |
| declared `latent_dim` | 3 × 192 = **576** (the eval hook infers `_m = 3` from this) |
| expectile | **0.05** |
| discount γ | **0.98** |
| n-step | 50 |
| steps / batch | 6000 / 1024 |
| sampler | `NStepGoalSampler`, `p_cross=0.3`, balanced |
| seed | 0 |
| data | cached fs1 latents, all 2,010,000 rows |

**Goal handling.** At *training* time the goal window is three genuine
consecutive frames (`_wrow(g_idx)`). At *deploy* only one goal frame exists, so
it is **tiled** three times (`_wpair`). `train_window.py` reports this gap
directly: `d(win → same-location tiled) = 7.02` against a random-pair baseline
of `82.19`.

**Critic data is cached-only.** `critic_step` samples exclusively from
`NStepGoalSampler(c_td, …)` over cached transitions, for both LIP and PWM.
LIP's value-expansion path exists but is **off** (`--expand-weight 0`). No arm
in this report touches the environment during training.

---

## 4. Planners

### 4.1 LIP (LIPv4) — the proposed method

**Shared across both bases** (these are the constrained knobs — TD/dataset
settings were required to be identical for the two bases):

| hyper | value |
|---|---|
| arch | `v4` |
| discount γ | **0.98** |
| expectile (critic co-training) | 0.1 → 0.03 (annealed) |
| critic-lr | 1e-3 → 1e-4 |
| replay-prob | **0.5** |
| expand-weight | **0** |
| n-step | 50 |
| steps | **1000** |
| batch | 128 |
| refinement iters | 8 |
| horizon | 5 blocks (= 25 primitive steps) |
| max-delta | 12 |
| λ schedule | uniform |
| context | `--pad-context` (1-frame deployment interface) |
| init-value | the §3 window value |

**Per-base** (only these three differ):

| | amax | actor-lr | mean-weight |
|---|---|---|---|
| **LeWM** | 2.2 | 1e-4 → 1e-5 | 0.3 |
| **PLDM** | 1.8 | 3e-4 → 3e-5 | 0.5 |

**Training seeds: 0–5 (6).** Each carded on 6 env seeds ⇒ **36 evals per base**.

`mean_weight` multiplies the mean of the terminal cost **across refinement
iterations k**, not across rollout timesteps — LIP has no per-timestep path
term.

Deployment cost: **~16 world-model rollouts per planning step.**

### 4.2 PWM — reactive policy baseline

`pi(z, z_g) → action block`: `h = cat[zp(z), gp(z_g − z)]` (two 192→256
projections), MLP 512×3 SiLU, `tanh(·)·amax`. Memoryless — no history, no
refinement. **Zero rollouts** at deploy (one encoder pass + one MLP pass; 5
imagined steps only to fill the plan tail under `receding_horizon=5`).

| hyper | value |
|---|---|
| objective | **terminal** (`−γ^H d̄(z_H, z_g)`) — matches LIP's |
| steps | **1000** — budget-matched to LIP |
| actor-lr | 5e-4 → 5e-5 |
| width / layers | 512 / 3 |
| EMA teacher τ | 0.005 |
| γ, n-step, horizon, max-delta, batch | 0.98, 50, 5, 12, 128 |
| expectile | 0.1 → 0.03 |
| amax | 2.2 (LeWM) / 1.8 (PLDM) |
| init-value | the §3 window value |
| seeds | 0, 1, 2 (3) ⇒ 18 evals per base |

### 4.3 Sampling / gradient planners

| solver | configuration | rollouts/step |
|---|---|---|
| **CEM** | 300 samples × 10 iterations, topk 30, var_scale 1.0 | 3000 |
| **Adam** | `GradientSolver`, AdamW lr 0.1, **300 samples × 10 steps** | 3000 |

Adam's config defaults (100 × 30) were **overridden to match CEM's budget
exactly**, so the CEM/Adam difference is planner quality, not compute.

- **Latent + CEM** — parameter-free L2 in 3-frame window space (`l2window3`, empty `state_dict`). This is the paper's method and the bar.
- **Value + CEM / Value + Adam** — the §3 learned value as the cost.

---

## 5. Main results — held-at-end @ 0.1 rad

Reporting seeds 42–47, episodes 8000:10000.

| arm | rollouts/step | **LeWM** | **PLDM** |
|---|---|---|---|
| **LIP** | **~16** | **98.2 ± 1.2** | **94.2 ± 4.1** |
| Latent + CEM *(bar)* | 3000 | 84.3 | 78.3 |
| PWM | 0 | 64.1 | 20.4 |
| Value + CEM | 3000 | 51.3 | 51.3 |
| Value + Adam | 3000 | 29.7 | 32.7 |

**LIP margin over the bar: +13.9 (LeWM), +15.9 (PLDM)** at ~1/200 the compute.

### held-at-end @ 0.05 rad

| arm | **LeWM** | **PLDM** |
|---|---|---|
| **LIP** | **66.3 ± 8.7** | **57.1 ± 12.5** |
| Latent + CEM | 44.7 | 39.3 |
| PWM | 35.0 | 10.3 |
| Value + CEM | 13.3 | 15.3 |
| Value + Adam | 9.0 | 9.3 |

± is the standard deviation across **training** seeds (LIP: 6). Sampling
planners have no training seed; their spread is across env seeds only.

**Per-seed LIP @0.1** — LeWM: 98.3 / 99.3 / 99.0 / 99.0 / 96.7 / 96.7.
PLDM: 98.7 / 96.0 / 87.0 / 92.7 / 97.0 / 94.0.

### Structural observations

1. **LIP beats both samplers at ~1/200 the compute.**
2. **Adam ≈ half of CEM at identical budget** — gradient planning through this
   world model is genuinely weaker than sampling.
3. **The learned value is the weakest component.** Value+CEM (51.3/51.3) is far
   below the *parameter-free* L2 cost (84.3/78.3) on the same planner and
   budget. LIP beats both — so its advantage comes from the amortised planner
   **in spite of** its critic, not because of it. This is the most robust
   structural finding in the campaign and survives every configuration tried.

---

## 6. Ablations

### 6.1 Discount γ (shared knob)

Selection seeds, @0.1, scored by the worst base's margin over its own bar:

| γ | LeWM | PLDM | min-margin |
|---|---|---|---|
| 0.95 | 81.7 | 86.7 | −2.6 |
| 0.97 | 95.0 | 89.0 | +10.7 |
| **0.98** | **99.3** | **93.0** | **+14.7** |
| 0.99 | 96.3 | 88.0 | +9.7 |
| 1.0 (undiscounted) | 89.0 | 90.4 | +4.7 |

A **sharp peak**, not a plateau — both neighbours are 4–5 points down. Before
this the entire stack was undiscounted (`gamma=1.0` was the default in both
`train_window.py` and `train_lip_ac.py` and was never overridden). Discounting
also improves the value on its own: **Value+CEM 38.3 → 51.3 (LeWM), 33.7 → 51.3
(PLDM)**.

### 6.2 Window value vs 1-frame value

| | LIP LeWM | LIP PLDM | Latent+CEM LeWM | Latent+CEM PLDM |
|---|---|---|---|---|
| **3-frame window** | **98.2** | **94.2** | **84.3** | **78.3** |
| 1-frame | 89.8 * | 72.4 | 73.0 | 70.7 |

The window is worth **8–22 points** and helps the parameter-free bar too, so it
is a property of the cost function rather than something only LIP exploits. It
stays, despite its tiled-goal bias and its plumbing cost.
\* LeWM 1-frame is 2 training seeds (12/18 evals) — one run hung at step 0 for
39 min on an otherwise idle machine and was killed.

### 6.3 Mean-weight (per-base, over refinement iterations)

| mw | 0.1 | 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|
| LeWM | 96.7 | **99.0** | 92.0 | 96.8 |
| PLDM | 93.1 | 91.0 | **94.1** | 91.0 |

Both optima are **interior**, not at a range edge.

### 6.4 Replay-prob × expand-weight (shared knob)

Selection seeds, @0.05:

| replay / expand | LeWM | PLDM | cross-base |
|---|---|---|---|
| **0.5 / 0** | 43.3 | 31.7 | 37.5 |
| 0 / 0 | 56.7 | 18.7 | 37.7 |
| 0 / 1.0 | 25.0 | 31.7 | 28.4 |
| 0.5 / 1.0 | 20.0 | 34.0 | 27.0 |

The bases disagree sharply on expansion (PLDM likes it, LeWM is destroyed by
it); `expand-weight 0` is the cross-base choice and its cost to PLDM is real.
Replay 0.5 was decided later by the full sweep, not this probe.

### 6.5 Value training steps (shared knob)

TD+CEM on values trained for {6000, 20000, 60000, 150000} steps: **flat across a
25× range** (6–13 on selection seeds, no trend). The TD loss does fall at 150k
(e.g. 1.5751 → 1.4537) but none of it reaches the task. The value's weakness is
**not** a training-budget problem.

### 6.6 PWM: objective × budget

| base | objective | steps | @0.1 |
|---|---|---|---|
| LeWM | dense | 1000 | 81.2 |
| LeWM | dense | 8000 | 85.9 |
| LeWM | **terminal** | **1000** | **64.1** ← table row |
| LeWM | terminal | 8000 | **0.3** |
| PLDM | dense | 1000 | 91.2 |
| PLDM | dense | 8000 | 99.0 |
| PLDM | **terminal** | **1000** | **20.4** ← table row |
| PLDM | terminal | 8000 | 1.8 |

**Terminal-only collapses with more optimisation** (64.1 → 0.3) while dense
improves — a clean demonstration of world-model exploitation: one scalar of
feedback at the most-extrapolated latent, and more steps simply drive harder
into critic error. LIP is also terminal w.r.t. the rollout and also degrades
with more steps (1000 beat 3000 in **all 12** pairings), which is why LIP's
1000-step setting is best understood as early stopping against this failure
mode rather than an arbitrary tuning choice.

*(An earlier PWM number of 99.0 on PLDM was obtained at 8000 steps with the
dense objective — 8× LIP's budget and a different objective. It is not a valid
baseline and is not the table row.)*

### 6.7 Value+CEM is refinement-limited, not value-limited

The Value+CEM row (51.3 / 51.3) was suspected of understating the learned value.
Three candidate causes were tested; only one mattered.

| variant | LeWM | PLDM |
|---|---|---|
| expectile 0.05, n_steps 10 *(table row)* | 51.3 | 51.3 |
| expectile 0.10, n_steps 10 | 49.0 | 50.0 |
| LIP's co-trained critic, n_steps 10 | 48.7 | 54.3 |
| **expectile 0.05, n_steps 30** | **63.3** | **57.3** |

**+12.0 / +6.0 from CEM refinement iterations alone.** The expectile did nothing
and the co-trained critic barely moved it — consistent with §3: that critic is
the same TD fit with more steps on the same cached data, having never seen
actor-visited states.

The interpretation is that a learned quasimetric needs **more refinement** than
a smooth L2 cost to be exploited: the same 300 samples, three times the
iterations. It does **not** enter the main table, because n_steps 30 is
300 × 30 = **9000 rollouts** against every other row's 3000. A matched
Latent+CEM at n_steps 30 is required before the comparison is fair; that run is
pending at the time of writing.

### 6.8 Dyna (world-model fine-tuning) — negative

Fine-tuning the WM on a 50/50 mix of expert (episodes 0:1500) and 3,600
on-policy episodes **helps the bar and hurts LIP**:

| arm on the Dyna WM | @0.05 | @0.1 |
|---|---|---|
| Latent + CEM | 53.0 (base 44.7) | 86.3 (base 84.3) |
| LIP, 5 seeds | 30.0 (base 49.8) | 75.1 (base 89.0) |

The margin **inverts**. Dyna is excluded from the headline. (It is also the only
part of the campaign that touched the environment.)

---

## 7. Validity caveats

**Training is entirely offline.** No environment interaction anywhere: the WM is
frozen, values train on cached latents, LIP's rollouts are *imagined*, and
`replay-prob 0.5` restarts the actor from its own previously **imagined**
windows. Dyna (§6.7, excluded) is the sole exception.

**The reported configuration is "leaked".** The value and actor train on the
full 0–9999 cache, so **20.0%** of every training batch is drawn from the
evaluation episodes (402,000 of 2,010,000 rows; neither trainer filters
episodes). A matched clean-split control was run — value and actor retrained on
episodes 0–7999 only, then carded on 8000:10000:

| | leaked (6 seeds) | **clean (6 seeds)** | leak |
|---|---|---|---|
| LeWM @0.1 | 98.2 ± 1.2 | **97.7 ± 1.6** | +0.5 |
| PLDM @0.1 | 94.2 ± 4.1 | **91.5 ± 6.0** | +2.7 |

The **margin** is robust regardless: Latent+CEM is measured on the same leaked
world model and its L2 cost is parameter-free, so it has no capacity to
memorise. The leak threatens absolute numbers, not the comparison.

**The world models themselves are leaked** — both were pretrained on all 10k
episodes. The clean split is therefore "value + actor clean, WM leaked", and is
not repairable without retraining the world models.

**DINO-WM is absent, not omitted.** Its latent is 75,264-d (unpooled patch
tokens), so a full cache is 605 GB against 389 GB free disk. The bar needs no
cache and was attempted at full width, but the 3-frame metric hook fails on this
base (`index -3 out of bounds for dimension 1 with size 2` — it supplies 2
frames). `cache_latents --compress rp1024` is the documented route for the value
and LIP; `rp` is preferred over mean-pooling because it preserves the L2
geometry the bar depends on. Not completed.

**Action convention was verified, not assumed** (this bit a parallel campaign).
For `dinowmnp` the checkpoint's `ext_mu/ext_std` are bit-identical to the
dataset stats `train_lip_ac.py` computes (`max|Δ| = 0.000e+00`), and a
ground-truth rollout comparison confirms the normalized convention is correct:

| convention | h=1 | h=4 |
|---|---|---|
| **normalized (z-scored)** | **0.2237** | **0.2335** |
| raw | 0.2488 | 0.2960 |
| zero actions | 0.2909 | 0.3590 |

---

## 8. Reproduction

Drivers and patches: `reacher/scripts_20260729/`. Key entry points —
`run_final6.sh` (clean 6-seed), `run_leak6.sh` (leaked 6-seed, the table),
`run_pwm.sh` + `run_pwm_ablate.sh`, `run_baselines.sh` (planner × cost),
`run_g098_grid.sh` / `run_phase2.sh` (γ and mean-weight), `run_window_ablate.sh`,
`run_valsteps.sh`, `run_cleansplit.sh`, `probe_dino_actions.py`.

**Two measurement defects were found and fixed during the campaign; earlier logs
may contain them.** (i) `meanof`'s `held10` parse returned the `10` from its own
key name, printing a constant `10.0` for every @0.1 mean. (ii) Concurrent
sampling-solver evals core-dump under GPU contention (five occurrences), and the
driver averaged a crashed arm to `0.0` — indistinguishable from a genuine total
failure. Every arm now reports how many seeds actually scored; treat any
unaudited `0.0` or `10.0` in older logs as a suspected defect.
