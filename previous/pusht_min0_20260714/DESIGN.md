# PushT min0 → 95%: TD redesign for the velocity-aliased latent

**Date:** 2026-07-14 · **Pod:** b (87.120.211.204:18099, 4×H100)
**Goal:** min0-style LIP-AC on PushT at ≥95 on the h25 protocol (= the official
"50-step budget" cell where the baselines page lists LeWM at 96 and our
latent+CEM replication drew 90/90/80 across draws).

## 1. Failure modes so far (what the TD change must actually fix)

**F1 — velocity aliasing (the root problem).** The official LeWM PushT latent is a
single-frame pooled ViT embedding: pose decodes at R²≈0.94–0.97 but velocity at
R²≈0.27 (`probe_vel.py`). Steps-to-go is NOT a function of a single frame, so the
TD regression target is ill-posed: two identical-looking states have different
cost-to-go. The deep sweep found the only workable corner under aliasing:
**n=1 backups** (don't compound the noise) + **τ=0.01 expectile** (resolve the
ambiguity maximally optimistically ≈ min over velocity-compatible states). That
teacher hit 88.0/52.7 with CEM — good, but structurally capped: it cannot rank
plans that differ mainly in arrival momentum, and every backup longer than one
step re-injects aliasing noise.

**F2 — the obvious fix was tried and failed for a *distribution* reason, not an
information reason.** `train_metric_hist.py` gave the value velocity:
V([z, Δz], [z_g, 0]), Δz = z_t − z_{t−5} (block-scale). Velocity probe R² rises
0.27 → 0.66. Planning got WORSE everywhere (118 vs 154 combined s42). Diagnosis
(writeup): at plan time the state-side input is the WM's **imagined** terminal
frame; imagined frames are smoothed, so imagined Δẑ is miscalibrated against the
real Δz the value was trained on. Differencing *amplifies* the imagined/real gap —
the velocity feature is exactly the component the WM gets most wrong.

**F3 — train/plan input mismatch exists even without Δz.** The value always
trains on real encoded latents and is queried on WM rollouts. Single-frame z is
close enough (pose dominates); Δz is not. `train_metric_imag.py` (drafted
2026-07-12 after the writeup, never run) fixes the mismatch Dyna-style: with
prob p_imag replace the state input z_t by a WM rollout (true dataset actions)
ending at t, keeping the TD target unchanged — "the imagined version of s costs
the same as s".

**F4 — multimodal contact landscape.** Push-left-around vs push-right-around:
local refinement (LIP one-shot, Adam) stalls at E≈2.4–2.75 where population CEM
reaches 0.89. Fixed at inference by 8 noisy restarts + argmin-V (~130 rollouts,
still ~70× cheaper than CEM). Cube's "no restarts on AC actors" lesson must be
re-probed here — PushT is the task where restarts earned their keep.

**F5 — the quasimetric head is load-bearing on PushT** (154 vs 48 for a plain
MLP head) — not via asymmetry (the learned function is de-facto symmetric) but
because its metric constraints regularize the distance field against
CEM-exploitable off-manifold minima. Keep it, unconditionally.

**F6 — selection noise.** n=50/draw → sd ≈5; 2-draw selection lied to us twice
(cube min0 "+7"). Deltas <~8 points need the 3-draw mean before belief. And for
tandem/min0 actors: E_final anti-correlates with success across input configs —
never select by it.

**F7 — what min0 changes and doesn't.** min0's actor input [A, ∇_A V, E] is
*representation-agnostic*: the actor interface doesn't change when the value's
input space does. All task information reaches the actor through the teacher's
gradient field and level — which is precisely the thing F1–F3 degrade. On cube,
tandem AC (schedamax) was worth +4 over sequential LIP against the same data;
on PushT tandem was drafted (`run_ac_pusht.sh`) but never run. So the two
untried levers compose: **a velocity-complete, imagination-calibrated teacher**
(better gradient field) × **tandem min0** (actor co-adapted to that field).

## 2. The TD change

State-side input becomes **[z, Δz] with Δz always drawn from the same generative
process the planner queries at plan time**:

- `train_metric_velimag.py` (Phase A, standalone): quasimetric head on 2×192-d
  input; state [z_t, Δz_t], bootstrap [z_{t+n}, Δz_{t+n}], goal [z_g, 0]
  ("arrive at rest" prior — success is pose-only, so goal momentum is
  irrelevant; HER reached-targets still teach d([z,Δz],[z,0])→small correctly).
  With prob **p_imag** the state input is replaced by [ẑ_t, ẑ_t − ẑ_{t−5}]
  where both frames come from a frozen-WM rollout of the TRUE dataset actions
  ending at t (depth 2–5 blocks, uniform — plan-time terminal queries sit at
  depth 5). Targets never change (n_eff/dist/real bootstrap). One knob, two
  fixes: velocity information (F1) delivered in the query distribution (F2/F3).
- Δ stride = 5 primitive steps = 1 action block = the spacing of imagined
  frames. Real-side Δz zeroed for the first 5 steps of an episode (as in hist).
- **Re-sweep n × τ in the new state space** (τ ∈ {0.01, 0.03, 0.1} ×
  n ∈ {1, 3, 5}, plus long-backup probes n=15 at τ ∈ {0.03, 0.1}): n1/τ0.01
  was the *aliasing-conditioned* optimum; if velocity de-aliases the state,
  longer backups should stop hurting (and help — they were what made
  cube/TwoRoom metrics strong; cube's optimum was n=50). Where the optimum
  lands is itself the diagnostic: if it stays at n1/τ0.01, velocity wasn't
  binding.
- Ablations that isolate the mechanism: `imag-only` (single-frame + p_imag —
  is input-matching alone the fix?) and `velimag p_imag=0` (= hist re-run —
  is velocity-without-matching still broken?). Together with the grid this
  gives the full 2×2.
- **State-mode comparison (user question: delta vs frame windows):** win3
  (`--win-frames 3`, the WM's own Markov window; state = 3 block-spaced
  frames, goal tiled) gets the IDENTICAL 11-cell sweep as delta; win5 gets 2
  probe cells, escalated only if it beats win3. Delta case: minimal
  velocity-complete input = smallest optimizer attack surface + cleanest
  quasimetric node semantics. Window case: exact depth-profile match to
  plan-time queries + contact/acceleration information + mild expert-window
  shape bias. Measured under identical selection, not argued.

**Selection:** CEM h25+h50 s42 combined (baseline: td2_e0.01_n1 = 154), same
harness as every prior number; winner confirmed on 3 draws. CEM selection is
planner-independent but CEM-rank ≠ gradient-rank (the PushT lesson that
motivated tandem AC) — hedged in Phase B: the runner-up teacher gets its own
min0 arm (`m0pi2`) whenever it is within selection noise (<8 combined) of the
winner. Everything downstream of Phase A is min0 (--drop-z0 --drop-zg); the
one full-input arm is a control, and headline numbers come from the min0
actor. The teacher's input mode is invisible to min0's [A, ∇V, E] interface.

## 3. min0 on top (Phase B)

`train_lip_ac.py` grows velocity support (auto-detected when the warm-start
metric's latent_dim = 2× cache dim, or `--vel-metric` for fresh critics):
critic TD pairs augmented on the fly ([z,Δz]/[z_g,0] via an index-returning
sampler), all teacher queries through a helper using the last two imagined
frames — ∇_A V then flows through BOTH frames, so the actor feels arrival
momentum in the gradient; the value-expansion tuple likewise ([z0, real block
delta from the fs5 context], [ẑ_H, ẑ_H−ẑ_{H−1}]). Optional `--p-imag` ports the
Dyna input-matching into the tandem critic so live TD steps don't drift the
imagined calibration away (the cube schedamax recipe re-trains the critic for
80% of the run — on real pairs only, that would slowly undo exactly the thing
Phase A bought).

Arms (all warm-started from the Phase A winner, n-step/expectile at the winner
corner, schedamax schedules, K=8, 8k steps, MLP min0 = --drop-z0 --drop-zg):
- `m0s` min0 schedamax (tandem, real-pair critic)
- `m0pi` min0 schedamax + --p-imag 0.5 (tandem, matched critic) ← expected best
- `m0f` min0 vs FROZEN teacher (freeze-critic-frac 0 — protects calibration by
  not moving the critic at all; sequential-min0 control)
- `full` champion-input schedamax (does dropping z0/zg cost anything on PushT?)

LIP-eval selection on s42 h25+h50 (beat: teacher 154, LIP2+R8 142), R8 restart
probe on the top two (F4 says restarts matter here; cube says they don't for AC
actors — measure, don't assume).

## 4. Headline (Phase C)

Winner → 3-draw h25 + h50, one-shot and R8; new teacher CEM 3-draw; plus
latent/teacher/min0 on the SAME protocol table for the writeup. Success bar:
h25 3-draw ≥95 is the stated goal; honest fallback is "highest PushT number at
~70× less plan compute than CEM, velocity fix quantified".

## 5. Cost

Phase A: 11 short TD trains + 22 CEM evals ≈ 3–4 h on 4 GPUs.
Phase B: 4 tandem trains + ~10 LIP evals ≈ 2–3 h. Phase C ≈ 3–4 h.
Setup (46 GB download, caches, baseline repro) ≈ 2 h. Fits one pod-day.
