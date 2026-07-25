# LIP-AC, MLP champion ("schedamax") — Method Notes

Cube / LeWM-v2WM. The current best cube planner: **v1 MLP actor architecture, trained
actor–critic with annealed schedules**. h25 mean **88.0** (88/96/80), one unswept tandem
run — ties/beats the fully swept sequential pipeline (87.3) and the published latent+CEM
(75.3). 2026-07-13.

## 1. Setting

Frozen world model W over pooled latents z ∈ R^192 (v2WM, 224px). Actions are blocks
a ∈ R^25 (5 primitive steps × 5 dims, z-scored). A plan is A = (A_1 … A_5): 5 blocks
= 25 primitive steps. The WM unrolls a plan autoregressively and differentiably:

    ẑ_t = W( ẑ[t−3:t−1], a[t−3:t−1] ),   t = 1 … 5   →   ẑ_1 … ẑ_5,  terminal ẑ_H

Two learned modules, trained in tandem:

| module | signature | role |
|---|---|---|
| critic V_φ | (z, z_g) → R≥0 | goal-conditioned temporal distance (quasimetric head: hidden 256, depth 2, embed 128); lower = closer |
| actor f_θ | (A, ∇V, E, z0, z_g) → (dA, gate) | learned refinement rule, applied K = 8 times, weight-tied |

## 2. Critic: n-step distance TD (expectile, HER)

Offline, stride-1 latent cache (2.01M frames — every frame; coarse caches hide grasp
events). Hindsight pairs: in-episode future goals (p = 0.7, balanced offsets) or
cross-episode (p = 0.3). n = 50 primitive-step backups, γ = 1:

    target   y = r·δ + (1−r)·( n_eff + V_target(z_{t+n}, z_g) )
    loss     L_TD(φ) = E[ w_τ(u) · Huber_1(u) ],   u = V_φ(z_t, z_g) − stop_grad(y)
    weight   w_τ(u) = 1−τ  if u > 0   |   τ  if u ≤ 0        (low τ → optimistic/min)

V_target = Polyak copy, ρ = 0.005. Warm-started from the sequential TD winner
(cf_dE_t003n50: τ 0.03, n 50, 6k steps).

**Schedules (the "schedamax" recipe — what beat the fixed-hyper arms):**

    expectile τ:  0.1 → 0.03   cosine over 8k steps
    critic lr:    1e-3 → 1e-4  cosine
    actor lr:     3e-4 → 3e-5  cosine
    plan clamp:   amax = 3.5   (expert action tails reach ~3.5σ; 2.5 clips them)

## 3. Actor: v1 MLP update rule

One flat network; no tokens, no attention. Inputs concatenated to a single vector:

    input  = [ A (125) | ∇_A V(ẑ_H, z_g) (125) | E (1) | z0 (192) | z_g (192) ]   635 dims
    net    = Linear(635→512) → ReLU → Linear(512→512) → ReLU → Linear(512→126)
    output = dA (5×25 full-plan update) + gate logit (1)

    A(k+1) = clip_±3.5( A(k) + σ(gate)·dA ),   k = 0…7,   A(0) = 0

E = V(ẑ_H, z_g) is the current plan's imagined steps-to-go; z0 = current encoded state.
Gradient and value features are detached (inputs to the rule, not training-gradient
paths). ~0.9M parameters. Default PyTorch init (the MLP tolerates it; see §6 of the
v3 notes for why the transformer didn't).

The single scalar gate multiplies the whole-plan update — a learned global step size
per iteration, and highway-style Jacobian damping through the 8 weight-tied
applications (the v3 no-gate ablation later confirmed the gate is worth ~5 points).

Explicitly, per refinement iteration:

    u  = [ vec(A) | vec(∇_A V) | E | z0 | z_g ]               ∈ R^635
    h1 = ReLU(W1·u + b1)          W1: 512×635
    h2 = ReLU(W2·h1 + b2)         W2: 512×512
    o  = W3·h2 + b3               W3: 126×512
    dA = reshape(o[0:125], 5×25),   g = σ(o[125])
    f_θ(A, ∇, E, z0, z_g) = clip_±3.5( A + g·dA )

    A(0) = 0
    for k = 0 … 7:                (weight-tied; no k input of any kind)
        ẑ_1…ẑ_5 = WM-unroll(history, A(k))       2 rollouts: gradient pass + value pass
        A(k+1)  = f_θ( A(k), ∇_A V(ẑ_5, z_g), V(ẑ_5, z_g), z0, z_g )
    execute A(8)

Two deliberate exclusions (both empirically validated): the actor never sees any
imagined latent — the trajectory reaches it only compressed through ∇_A V and E
(v2's direct end/traj feeds scored 82.0 vs 87.3) — and it never sees the iteration
index; stage-dependent behavior is inferred from the inputs themselves (E and the
gradient shrink as the plan improves).

## 4. Actor loss (pathwise, no RL, no BC)

    L_actor(θ) = E[ V_teacher(ẑ_H(A(K)), z_g) + 0.1 · mean_k V_teacher(ẑ_H(A(k)), z_g) ]

Backpropagated through all K refinement iterations and the frozen WM rollouts, into θ
only. Start/goal pairs from play data: offsets ≤ 10 blocks (50 primitives), 30%
cross-episode. Batch 128, 8000 steps, AdamW wd 1e-5, grad-clip 10.

## 5. Actor–critic coupling

- Interleaved 1:1 per step; strict gradient separation (actor loss never touches φ —
  else the critic learns "make plans look good" and collapses).
- The actor's teacher = EMA copy of the critic (ρ = 0.005), **frozen for the final 20%**
  of training (stable last-phase teacher).
- Value expansion (TD backups on the actor's imagined rollouts) tested at w = 0.3:
  preserved the teacher's CEM quality, did not help the student. Off in the champion.

**Dissociation result:** tandem training degraded every teacher as a CEM cost
(sequential teacher 82 → warm 68 / schedamax ~70s / fresh 64 at h25/s42) while the
students stayed at or above champion level. CEM-ranking quality and gradient-field
quality are separable axes of value quality; AC training optimizes the one the planner
consumes. (This is the cube-side mirror of the PushT finding, where the CEM-picked
teacher produced a bad LIP.)

## 6. Results (full protocol: 50 tasks/draw, budget 50, success = block within 4cm)

h25, seeds/draws 42/43/44:

| method | 42 | 43 | 44 | mean |
|---|---|---|---|---|
| random floor | 46 | 56 | 50 | 50.7 |
| latent + CEM (published planner) | 80 | 84 | 62 | 75.3 |
| TD(τ0.03/n50) + CEM (teacher, sequential) | 82 | 90 | 66 | 79.3 |
| sequential champion LIP v1 (swept) | 88 | 96 | 78 | 87.3 |
| **LIP-AC schedamax (one unswept run)** | **88** | **96** | **80** | **88.0** |

h50 s42: 74.0 — ties the h50-swept sequential winner (72–74 band) with zero h50-specific
tuning. Training-seed robustness (s42+s44 selection sums): seeds {0,1,2} = {168, 154, 168}.

Deployment: K = 8 iterations × 2 WM rollouts ≈ **16 rollout-equivalents per env step vs
9,000 for CEM** (300×30) — ~500× less planning compute, fully deterministic (no sampling,
no restarts; restarts actively hurt: the actor specializes to its zero-init refinement
path).

## 7. Artifacts & reproduction (pod 31.24.80.32:15419)

    actor   /workspace/actors/lip_ac90_schedamax.pt      (kind='lip', feed='none', amax 3.5)
    teacher /workspace/metrics/lip_ac90_schedamax_value.pt
    driver  /workspace/run_ac90b_ogbench.sh              (round B; winner confirm = win90_schedamax_*)
    trainer scripts/plan/train_lip_ac.py                 (arch=mlp default)

    train:  python scripts/plan/train_lip_ac.py --cache cube_full_fs5.pt --cache-td cube_full_fs1.pt \
              --h5 cube_single_expert.h5 --wm ogbench_cube_single_v2WM \
              --init-value metrics/cf_dE_t003n50.pt --horizon 5 --iters 8 --steps 8000 --n-step 50 \
              --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
              --actor-lr 3e-4 --actor-lr-final 3e-5 --amax 3.5 --seed 0
    eval:   python scripts/plan/eval_wm.py --config-name cube seed=42 ++bf16=true eval.img_size=224 \
              policy=<v2WM> solver=lip solver.actor_path=<actor.pt> eval.goal_offset_steps=25 \
              eval.eval_budget=50 eval.dataset_name=<h5>

## 8. Why the MLP is the champion (v3 transformer post-mortem, one paragraph)

The trajectory-aligned transformer (v3/v3.2: per-block tokens, aligned imagined latents,
global conditioning, small-init, feat-norm) was tested across five rounds — init (V/Z/G),
gate (H), optimization (S1) — best recipe mean 155 vs the MLP's 163.3 on the s42+s44
selection. It consistently underfits the actor objective (E_final ~3–4 vs 1.46), and
pushing it to fit further (12k steps) made evals WORSE (value exploitation), so the gap
is not closable by training harder. At H = 5 tokens, attention has little to coordinate
and its optimization cost buys nothing; the flat MLP is the better-matched inductive
bias at this plan length. (h50 probes of the best v3 arms — where attention should pay
rent — pending at time of writing.)

## 9. Minimal-input variant "min0" (input sweep winner)

An input-ablation square on the champion {full, −z_g, −z_0, −both} × 2 seeds found that
the actor needs neither raw input. The winner, **min0**, is the champion in every respect
(same MLP, same AC schedamax recipe) EXCEPT its input drops both the raw current state z0
and the raw goal z_g:

    champion:  u = [ vec(A) | vec(∇_A V) | E | z0 | z_g ]      ∈ R^635
    min0:      u = [ vec(A) | vec(∇_A V) | E ]                 ∈ R^251

    net    = Linear(251→512) → ReLU → Linear(512→512) → ReLU → Linear(512→126)
    f_θ(A, ∇_A V, E) = clip_±3.5( A + σ(gate)·dA )         (gate optional — see below)

with, as before, ∇_A V = ∇_A V(ẑ_H(A), z_g) and E = V(ẑ_H(A), z_g), both teacher-computed
and detached. The goal z_g and current state z0 now reach the actor **only** through these
two teacher signals — the actor never sees an absolute latent, only the value's level and
its gradient w.r.t. the plan.

Mathematically this makes f_θ a **pure learned optimizer** for the objective family
{V(·, z_g)}: a map (plan, ∇V, V) ↦ refined plan that is, by construction, invariant to the
latent representation and agnostic to which goal it is solving — it consumes only the local
geometry of the teacher's landscape. This is the cleanest possible statement of "the value
is the only teacher": it is also the only *input*.

Results (h25 s42+s44, champion full-input ref seeds 168/154 = mean 161):

| variant | input | 2-seed mean | 4-seed detail |
|---|---|---|---|
| full (champion) | [A,∇V,E,z0,z_g] | 161 | s0=168, s1=154 |
| −z_g | [A,∇V,E,z0] | 167 | 164, 170 |
| −z_0 | [A,∇V,E,z_g] | 149 | 160, 138 (noisy) |
| **min0 = −both** | [A,∇V,E] | **168 → 169** | 172/164/170/170 (spread 8) |
| min0, no gate | [A,∇V,E], A+dA | 166 | 166, 166 |

Per-draw, min0's four seeds average **s42 = 86.5, s44 = 82.5** (champion's confirmed seed:
s42 = 88, s44 = 80). min0 matches the champion at **lower seed variance** (spread 8 vs 14)
while being strictly leaner, and the gate can also be dropped (pure residual A + dA) at no
measured cost — unlike the transformer, where the gate was worth +5.

**3-draw confirm (s43 added; control: champion schedamax reproduced 96.0 exactly on s43):**

| min0 gated seed | s42 | s43 | s44 | 3-draw |
|---|---|---|---|---|
| s0 | 88 | 94 | 84 | 88.7 |
| s1 | 86 | 96 | 78 | 86.7 |
| s2 | 86 | 94 | 84 | 88.0 |
| s3 | 86 | 94 | 84 | 88.0 |
| **mean (4 seeds)** | | | | **87.8** |

Champion (single confirmed seed): 88/96/80 = **88.0**. min0 no-gate: 87.3 / 87.3.

**Verdict: min0 TIES the champion (87.8 vs 88.0 — 0.2, deep inside noise), it does not
beat it.** The 2-draw metric's apparent +7 was small-n noise, as flagged: s43 is the
champion's strong draw (96) where min0 averages ~94.5, slightly behind; the s42+s44 pair
excluded it. Corrected finding: min0 **matches** the champion with (a) strictly leaner
input (251 vs 635 dims — no raw state, no raw goal), (b) lower seed variance (4 seeds
within 2.0 points vs the champion's 14-pt spread on the selection metric), (c) gate
optional, (d) validated on 4 seeds vs the champion's single confirmed seed. It is the
better *design* — not a path past 88. E_final anti-correlated with success throughout
(min0 trained to E ≈ 3.8–4.0 vs champion 1.46 yet tied): removing raw z0/z_g removes the
actor's ability to exploit value idiosyncrasies — worse fit to the teacher, equal
real-world success. >90 remains open (see mean-weight sweep / failure autopsy).

    actors  /workspace/actors/lip_ac90s5_min0_s{0,1}.pt, lip_ac90s6_w_s{2,3}.pt (gated),
            lip_ac90s6_wng_s{0,1}.pt (no-gate)
    train   scripts/plan/train_lip_ac.py … --drop-z0 --drop-zg [--no-gate]  (else schedamax)
