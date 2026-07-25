# LIP-AC on Cube — Experiment Log

> 2026-07-14: the 88-plateau is causally explained — see **FAILURE_AUTOPSY.md** (two failure
> modes, per-episode video gallery in failure_gallery/, probe chain, and the negative-data
> collection for the WM re-run).

Cube single / LeWM-v2WM, full protocol (50 tasks/draw, budget 50, success = block ≤ 4cm,
draws = eval seeds 42/43/44). All numbers are success %. Selection metric throughout:
h25 s42+s44 sum per trained actor (max 200). Method details: LIP_AC_MLP_CHAMPION.md
(champion) and LIP_AC_MATH.md (v3 transformer line). 2026-07-12/13, pod 31.24.80.32.

## 0. Starting point (sequential era, for reference)

Pipeline: TD value (τ0.03, n=50, stride-1 cache) → selected by CEM → LIP v1 trained
against the frozen winner.

| h25 | 42 | 43 | 44 | mean |
|---|---|---|---|---|
| random | 46 | 56 | 50 | 50.7 |
| latent+CEM (published) | 80 | 84 | 62 | 75.3 |
| TD+CEM (teacher) | 82 | 90 | 66 | 79.3 |
| LIP v1 (swept) | 88 | 96 | 78 | 87.3 |

Motivating question: the TD is selected by CEM ranking, but LIP consumes the value's
gradient field — different axes (PushT showed the ordering can invert). Can training
critic and actor in tandem (actor–critic) remove the mismatch?

## 1. Round 1 — four tandem schemes (v1 arch, fixed hypers)

Fresh-pod bootstrap; anchors reproduced exactly (LIP 88.0, TD+CEM 82.0 on h25 s42).
Arms: critic from scratch (fresh), warm-started (warm), never-frozen teacher
(nofreeze), value expansion w=0.3 (expand03). All: v1 actor, expectile 0.03, K=8, 8k.

| arm | h25 s42 | h50 s42 | sum sel. |
|---|---|---|---|
| warm | 88 | 74 | 162 |
| nofreeze | 88 | 74 | 162 |
| expand03 | 86 | 68 | 154 |
| fresh | 84 | 68 | 152 |

Winner warm; 3-draw confirm 88/92/76 (85.3) — ties champion on s42, −2 elsewhere.
h50 74 with zero h50-specific tuning (sequential needed a dedicated sweep to reach 72–74).

**Key finding (dissociation):** every tandem teacher got WORSE as a CEM cost
(h25 s42: 68/74/78/64 vs sequential teacher's 82) while the students stayed at champion
level. CEM-ranking quality ≠ gradient-field quality; AC optimizes the axis the planner
uses. This answers the motivating question: the mismatch is real, and tandem training
exploits it — you can trade CEM quality you don't need for teachability you do.

## 2. Round 2 ("B") — schedules + amax (the current champion)

Also probed eval-time restarts on AC actors: they HURT (164 plain vs 160/158 with
R8 restarts) — the actor specializes to its zero-init refinement path. No restarts since.

Arms (on warm base): lr anneals (sched), max-delta 5 (md5), anneals + amax 3.5
(schedamax), amax alone (amax35).

| arm | s42+s44 |
|---|---|
| **schedamax** | **168** |
| md5 | 162 |
| sched | 158 |
| amax35 | 152 |
| (warm baseline) | 164 |

**Champion = schedamax**: expectile 0.1→0.03 cosine, critic-lr 1e-3→1e-4,
actor-lr 3e-4→3e-5, amax 3.5. Confirm: **h25 88/96/80 = 88.0 mean** (sequential champion
87.3, on 1 unswept run vs their full sweep); h50 s42 74. Training-seed robustness:
seeds {0,1,2} = {168, 154, 168}, mean 163.3. Reference E_final curve: 5.2 @500,
4.6 @1k, 1.46 @7.5k.

## 3. v3 transformer arc (trajectory-aligned refiner) — six rounds, closed

Motivation: v1's flat MLP has no locality (single scalar gate for the whole plan), no
iteration awareness, and v2's naive latent feeds hurt (82.0 vs 87.3) — misaligned
conditioning. v3: one token per block [A_t | grad_t | ẑ_t], transformer over 5 tokens,
per-token updates; v3.2 moved scalars (E, k) to a global conditioning bias and
normalized features.

| round | config | s42+s44 per seed | mean | verdict |
|---|---|---|---|---|
| V | vonly tokens, default init, iter-emb | 164/134/138/142 | 144.5 | underfits (E 3.5–7.5 vs 1.46); 30-pt seed spread |
| Z | + exact-zero init | killed @1.5k | — | dead-gradient flaw; superseded |
| G | v3.2: slim tokens, global cond, feat-norm, small-init ×0.01 | 156/152/158/154 | 155.0 | +10.5 vs V, spread 6 — fixes worked, still < MLP |
| H | G + no-gate | 146/150/148/156 | 150.0 | gate worth +5 on transformer |
| S1 | H + {hot-lr, 12k steps, w384, pre-LN} | best evals 132/130 | — | **lower train E → worse eval** (value exploitation); gap not closable by training harder |
| S2 | H + mean-weight {0, 0.3} vs 0.1 | 126/110 · 158/158 (vs 146/150) | — | clean monotone dose-response: intermediate-iterate loss term essential; 0.3 > 0.1 ≫ 0 |

h50 probes of the best v3 arms: G-best 68, H-best 70 vs MLP champion 74 — attention
doesn't pay rent at h50 either.

**Arc verdict:** at H=5 tokens the flat MLP is the better-matched inductive bias; the
transformer persistently underfits the actor objective, and pushing it to fit harder
trades into value exploitation rather than real plans. Champion stays v1-MLP.
Transferable finding: the actor loss NEEDS the intermediate-iterate term (mean_k V), and
more than the default 0.1 — testing 0.3/0.5 on the champion is the standing >90
candidate (round S4, currently shelved by the input-sweep directive).

## 4. Input sweep on the champion (COMPLETE 2026-07-13)

Question: which raw inputs does f_θ actually need? The champion consumes
[A, ∇V, E, z0, z_g]; the trajectory reaches it only through the teacher. Square:
{full, −z_g, −z_0, −both} × 2 seeds, then gate test on the square winner + champion.
Metric: h25 s42+s44 sum. Champion full-input ref: seeds 168/154 (mean 161).

| actor input | arm | seeds (s42+s44) | mean |
|---|---|---|---|
| full [A,∇V,E,z0,z_g] | (champion) | 168, 154 | 161 |
| −z_g [A,∇V,E,z0] | zg0 | 164, 170 | 167 |
| −z_0 [A,∇V,E,z_g] | z00 | 160, 138 | 149 (spread 22 — noisy) |
| **−both [A,∇V,E]** | **min0** | **172, 164** | **168 (winner)** |

S6 (winner = min0, +2 seeds +gate test):
- min0 all 4 seeds: 172 / 164 / 170 / 170 → **mean 169, spread 8** (champion spread 14).
- min0 no-gate: 166 / 166 — ties gated min0; **gate ~free to drop on the MLP** (vs +5 on
  the transformer, round H).

**Findings:**
1. The minimal learned-optimizer form [A, ∇V, E] — no raw state, no raw goal, goal/state
   reach the actor ONLY through the teacher's gradient+scalar — is the best and
   lowest-variance variant. Ties/beats the full champion and can also drop the gate.
2. E_final ANTI-correlated with eval: min0/zg0 had worse training loss (~4.0 vs champion
   1.46) but better success; champion-no-gate had near-champion E (~1.8) but worse
   success. Mechanism: removing raw z0/z_g removes the actor's ability to exploit value
   idiosyncrasies → forced onto the robust gradient/value signal → worse fit, better
   generalization. (Same value-exploitation axis as S1, seen from the input side.)
3. Non-monotonic: −z_0 ALONE hurts (149) but −both helps (168). Reading: raw z_g without
   z_0 is a misleading goal with no "where am I" anchor; remove both and the actor relies
   on the self-consistent teacher signals. Weakest cell (n=2, spread 22) — least reliable.

**3-draw confirm (s43 added; champion schedamax control reproduced 96.0 on s43 → harness
trustworthy):** min0 gated 4 seeds = 88.7 / 86.7 / 88.0 / 88.0 → **mean 87.8** vs champion
**88.0** — a tie. No-gate: 87.3 / 87.3. The 2-draw +7 was small-n noise (s42+s44 excluded
s43, the champion's strong draw = 96, where min0 ≈ 94.5). Corrected: min0 MATCHES the
champion with leaner input (251 vs 635), lower seed variance (2.0 vs 14 spread), gate
optional, on 4 seeds vs the champion's 1. Better design, not a win. >90 still open.

## 5. Method & ops lessons (paid for; don't re-buy)

- **Selection statistics:** eval seed drives the task draw (draw spread ~22 pts);
  n=50/cell → 1 episode = 2 pts, binomial sd ≈ 5. Two-cell sums still carry ±4 noise;
  ±14 seed spread on identical recipes observed. Compare recipe MEANS over paired seeds;
  treat +2 "wins" as ties.
- **E_final is a leading indicator, not a verdict** (S1: E 3.30 evaluated 132 while
  E 4.4 evaluated 146–156). Use it to kill clearly-broken runs, never to promote.
- **Teacher-freeze artifact:** td_loss prints nan after 80% of steps — expected
  (frozen-teacher phase), present in champion runs too.
- **Restarts hurt AC actors** (zero-init specialization). Deterministic K=8 deploy.
- **Value expansion** (TD on planner rollouts) preserves teacher CEM quality but doesn't
  help the student.
- Fresh-pod replication of the sequential anchors was EXACT (88.0 / 82.0) — caches,
  renderer, action-stats, eval harness all verified before any new conclusion was drawn.
- **Driver-generation bug (cost ~35 min):** generating round scripts by string
  substitution silently no-op'd an arm block (replace source spelled a post-rename
  name); the round trained the previous round's arms. Fix + rule: verify the actual
  train-call lines AND the live process cmdlines (`ps` showing --drop-zg etc.), not
  summary variables.
- ssh/pkill self-match traps (3×): a kill pattern that appears in plain form anywhere in
  the same remote command kills the session itself. Kill by PID list or bracketed
  patterns, verify in a separate session.

## 6. Current standings (h25, 3-draw mean where confirmed)

| rank | planner | mean | note |
|---|---|---|---|
| 1 | **LIP-AC schedamax (v1 MLP, full input)** | **88.0** (88/96/80) | 1 confirmed seed |
| 1= | **min0 (v1 MLP, input [A,∇V,E])** | **87.8** (4-seed 3-draw) | TIE; leaner + lower-variance + gate-optional |
| 3 | sequential LIP v1 | 87.3 (88/96/78) | full sweep behind it |
| 4 | v3.2 best (G / S2-mw03) | ~155–158 on 2-cell sum | never confirmed 3-draw |
| — | latent+CEM (published) | 75.3 | 9,000 rollouts/step vs LIP's 16 |

Open items, in order of expected value: (1) **confirm min0 on 3 draws** (add s43 × its 4
seeds) — the leaner/lower-variance minimal-input actor [A,∇V,E], candidate to replace the
champion; (2) mean-weight 0.3/0.5 on the champion (shelved S4 — standing >90 candidate),
now testable stacked on min0; (3) failure autopsy of the ~6 failing episodes per draw (is
>90 an actor problem at all?); (4) sequential-version input square, if wanted.

Sweep artifacts (pod, idle): actors /workspace/actors/lip_ac90{s3_zg0,s5_z00,s5_min0,
s6_w,s6_wng}_*.pt; drivers run_ac90s{3,5,6}_ogbench.sh; summary /workspace/results/
summary.csv (rows lipac90s{3,5,6}_*). Flags: PlannerNet(use_zg/use_z0/use_gate),
train_lip_ac.py --drop-zg/--drop-z0/--no-gate; ck.get defaults True (legacy-compatible).
