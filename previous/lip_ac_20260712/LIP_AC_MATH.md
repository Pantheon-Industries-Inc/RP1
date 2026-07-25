# LIP-AC: Learned Iterative Planning, Actor–Critic Training — Method Notes

Cube / LeWM-v2WM instantiation. Version **v3.2** (slim tokens, global conditioning,
feat-norm, small-init, no gate) — 2026-07-13.

A learned plan-refinement rule (the actor) and a quasimetric temporal-distance value
(the critic) trained in tandem on offline data. No behavior cloning, no policy gradient,
no environment interaction — the value is the only teacher.

## 1. Setting

Frozen world model W over pooled latents z ∈ R^192 (v2WM, 224px). Actions are blocks
a ∈ R^25 (5 primitive steps × 5 dims, z-scored). A plan is A = (A_1 … A_H), H = 5 blocks
= 25 primitive steps.

Given the encoded 3-frame history, the WM unrolls autoregressively, differentiable in A:

    ẑ_t = W( ẑ[t−3:t−1], a[t−3:t−1] ),   t = 1 … H

giving the imagined trajectory ẑ_1…ẑ_H and terminal ẑ_H(A).

| module | signature | role |
|---|---|---|
| critic V_φ | (z, z_g) → R≥0 | goal-conditioned temporal distance (steps-to-go, quasimetric head); lower = closer |
| actor f_θ | (plan, context, k) → update | learned refinement rule, applied K = 8 times |

## 2. Critic: n-step distance TD (expectile, HER)

Offline, on a stride-1 latent cache (every frame). Pairs (z_t, z_g) by hindsight goals:
in-episode future (p = 0.7, balanced over offsets) or cross-episode (p = 0.3, stitching).
With n = 50 primitive-step backups, γ = 1, reached-flag r, actual distance δ:

    target   y = r·δ + (1−r)·( n_eff + V_target(z_{t+n}, z_g) )

    loss     L_TD(φ) = E[ w_τ(u) · Huber_β(u) ],    u = V_φ(z_t, z_g) − stop_grad(y)

    weight   w_τ(u) = 1−τ   if u > 0      (over-estimates punished hard)
                      τ     if u ≤ 0

Low τ (≪ 0.5) pushes predictions down → optimistic toward the min: shortest-path
semantics under noisy, multi-modal backups. V_target is a Polyak copy (ρ = 0.005).

Champion schedule ("schedamax"): τ annealed 0.1 → 0.03 cosine, critic-lr 1e-3 → 1e-4,
critic warm-started from the sequential TD winner.

## 3. Actor: what function is being learned

Not a policy, not a plan-generator: an update rule — a learned optimizer over plans.
Refinement from the zero plan A(0) = 0:

    A(k+1) = clip_±3.5( A(k) + dA(k) ),   k = 0 … K−1
    dA(k)  = f_θ( tokens(A(k)), E(k), k )        — pure residual, NO gate (v3.2)

The per-block sigmoid gate of v1–v3.1 (A + g·dA) is expressively redundant — g·dA is
absorbable into dA. Its real role was highway-style Jacobian damping through the K-step
weight-tied unroll (gates < 1 keep each factor of the product Jacobian near identity);
small-init now provides that at the start of training. Round H ablates the gate; round G
(gated) is the paired reference.

No algorithmic prior is baked into the rule — the value gradient enters only as an input
feature; what to do with it is fully learned (see §6). Deployed plan = A(K), executed
receding-horizon. No sampling, no restarts.

### Tokens (v3.2: slim tokens + global conditioning)

One token per plan block, t = 1 … 5:

    u_t = [ A_t (25) | ĝ_t (25) | ẑ_t (192) ]                          242 dims

    ĝ   = ∇_A V(ẑ_H, z_g) / RMS(∇_A V)          normalized terminal-value gradient
                                                 (per-sample RMS over all H×25 entries)

The two scalars enter through a dedicated global conditioning channel — a learned bias
added to every token embedding — instead of the token concat:

    x_t = W_in·u_t + pos_t + W_c·[ E/25 , k/(K−1) ]
    E   = V(ẑ_H, z_g)                            final value only, per-step values dropped

Rationale: as 2 of 244 concat dims beside a 192-dim latent, the scalars ride a sliver of
the input matrix; as a dedicated W_c (2 → 256) they receive gradient from all five tokens
every step. Per-token goal information flows through the gradient slice alone.

Network: token embed (242 → 256) + positional embedding + conditioning bias → LayerNorm
(feat-norm) → 2-layer transformer encoder (4 heads, FF 512, full bidirectional attention
over the 5 tokens) → per-token linear head → dA_t (25). ~1.6M parameters.

Key properties:

- The full aligned trajectory is fed, distributed across tokens: token t holds the
  post-state of block t (the state block t produces). Cross-block context flows through
  attention. The binding "state this block produces ↔ edit of this block" is structural,
  not learned (v2's failure mode was making the net learn it from a flat concat).
- The goal never enters raw — only through the learned metric: the value scalar and the
  terminal-value gradient. The actor is goal-agnostic by construction; raw latent
  geometry toward the goal (the weak planning signal) is never exposed.
- Gradient and value features are detached: inputs to the rule, not paths of the
  training gradient. The trajectory is re-imagined every iteration — the refiner watches
  its plan's predicted consequences move as it edits.

## 4. Actor loss (pathwise, no RL)

    L_actor(θ) = E[ V(ẑ_H(A(K)), z_g)  +  0.1 · mean_k V(ẑ_H(A(k)), z_g) ]

Backpropagated through the entire K-step refinement chain AND the frozen WM rollouts,
into θ only. Start/goal pairs from play data (offsets ≤ 50 primitive steps, 30%
cross-episode).

## 5. Actor–critic coupling (the tandem scheme)

- Strict gradient separation. φ is updated only by L_TD on dataset pairs; L_actor never
  touches φ — otherwise the critic learns "make plans look good" and collapses.
- Slow teacher. The actor optimizes against an EMA copy of the critic, frozen for the
  final 20% of training. Interleaved 1:1; actor-lr 3e-4 → 3e-5 cosine.
- Optional value expansion (off in champion): TD pairs harvested from the actor's
  imagined rollouts — kept the teacher CEM-usable, didn't help the student.

Key empirical fact (cube): tandem training degrades the critic as a CEM cost
(82 → 64–78 at h25/s42) while the student planner stays at or above champion level.
CEM-ranking quality and gradient-field quality are separable axes of value quality —
the AC loop optimizes the axis the planner consumes.

## 6. Initialization: the invariant, and the resolution

The refiner is residual and applied K times inside one unrolled graph, so initialization
decides whether early training compounds noise.

The invariant: any design that "starts as gradient descent but can learn more" is
necessarily GD + learned correction — an external −η·ĝ term, a preconditioner+residual
head (−softplus(p)·ĝ + r), a full learned preconditioner, attention with the gradient as
value stream: all reparameterizations of the same sum. A single mechanism whose zero
point is GD and whose span exceeds the gradient direction does not exist. The
composition is the price of the prior.

The resolution: don't pay it. The gradient is already an input feature; the network
learns its use (v1 did, from default init). Initialization alone fixes both real
pathologies, with no structural priors:

| scheme | head init | step-0 behavior | verdict |
|---|---|---|---|
| default (round V) | PyTorch | random updates compounded ×8 | slow convergence, E@1k ≈ 2× v1, 30-pt seed spread |
| exact-zero (round Z) | W, b = 0 | exact identity | encoder gradients dead until head grows |
| small-init (rounds G/H) | W × 0.01, b = 0 | near-identity, all gradients live | CHOSEN — one init scalar, zero mechanisms |
| precond head (shelf) | p-bias = softplus⁻¹(0.05) | ≈ normalized GD | canonical form IF a descent prior ever proves useful |

Supporting choices: iteration as plain scalar k/(K−1) (no embedding → nothing to patch
at init); feat-norm (ĝ instead of raw gradient, E/25, token LayerNorm) balances input
scales without touching the update rule.

## 7. Deployment & compute

A(0) = 0; each iteration costs 2 WM rollouts (one differentiable for the gradient, one
for features):

    ≈ 2K = 16 rollout-equivalents / env step   vs   9,000 for CEM (300 × 30)
    → ~500× less planning compute, deterministic

The v3 transformer adds < 1% wall-time — rollouts dominate (0.77 vs 0.79 s/step).

| method (h25, draws 42/43/44) | 42 | 43 | 44 | mean |
|---|---|---|---|---|
| random floor | 46 | 56 | 50 | 50.7 |
| latent + CEM (published planner) | 80 | 84 | 62 | 75.3 |
| sequential champion LIP v1 | 88 | 96 | 78 | 87.3 |
| AC schedamax (v1 arch) — current best | 88 | 96 | 80 | 88.0 |

Target: mean > 90.

## 8. Round map (v3 architecture, 4 seeds × schedamax teacher each)

| round | config | result (s42+s44 per seed; mean) |
|---|---|---|
| V | default init, token-cond, iter-emb | 164/134/138/142; 144.5 — LOSES (underfit: final E 3.5–7.5 vs v1's 1.46) |
| Z | exact-zero init | killed (~step 1500) — superseded by small-init |
| G | v3.2 gated: global-cond + feat-norm + small-init | running (E@500 = 6.4; v1 ref 5.2; round V was 10.2) |
| H | v3.2 + no-gate (pure residual) | chained after G — isolates the gate |

Selection: s42 + s44 at h25 per arm; incumbent best = schedamax 168 (3-seed mean 163.3).
Winner (if > 168) confirms on 3 draws + h50 s42. No restarts anywhere.
Leading indicator: E_final at steps 1k / 4k / 7.5k vs v1's reference curve 4.6 → 2.x → 1.46.
