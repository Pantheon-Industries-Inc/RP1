# Spec: the pretrained LM as world model — porting LeWM / TRM / LIP to language

**Date:** 2026-07-28. **Status:** design only, nothing built.
**Scope:** a pretrained base LM becomes the world model for an agentic coding loop,
with a learned goal-conditioned value and a learned planner. **No RL anywhere** —
no reward model, no policy gradient, no environment interaction during training.

Read alongside `stable-worldmodel/docs/lip_results.md` (the cube/PLDM card this
ports), `Dyna/HANDOFF_20260728.md` (amax + Dyna), `Dyna/DATA_SPLIT_POLICY.md`.

---

## 0. The claim, and the one thing to get straight first

The cube stack is three learned pieces stacked on frozen self-supervised
representations, and **none of them is RL**:

1. a **world model** — latent dynamics `f(z_{t-k..t}, a) → ẑ_{t+1}`, trained by
   self-supervised latent prediction on offline traces;
2. a **value** — goal-conditioned *temporal distance* `d(z, g)`, trained by TD
   regression whose only supervision is **how many steps apart two states were in
   a trajectory**;
3. a **planner** — an actor trained by *backpropagating the value through the
   frozen world model*.

Piece 2 needs no reward: time is the label. Piece 3 needs no policy gradient: the
simulator is differentiable, so you descend the value directly. That is why this
whole thing ports to a pretrained base with no RL on top — and it is the property
every design choice below has to preserve.

**The deep disanalogy, stated up front.** Cube dynamics are smooth, local, and
low-dimensional. Program semantics are not: "does this test pass after this patch"
is, in general, *run the program*. A latent one-step predictor cannot be an
interpreter, and any spec that pretends otherwise will fail.

The resolution is architectural, and it is already how the cube stack deploys:

> **The world model ranks. The environment verifies.**

The WM never has to be right in general — only good enough to *order* the handful
of candidate actions a planner proposes, over a horizon of 3–8 actions. Ground
truth comes from actually running the tests, at which point you replan. This is
exactly the receding-horizon loop of §6.3, and it is the reason short horizons are
not a limitation here but the design.

---

## 1. Component map

| cube stack | language / coding | new work |
|---|---|---|
| pixel obs 224px | serialized repo+session state (§3.1) | serializer, pinned |
| ViT-tiny encoder, frozen | frozen base LM + read-out (§3.2) | read-out head |
| 3-frame window, 192-d latent | k-step obs history, token read-out (§3.2) | — |
| predictor depth-6 → `ẑ_{t+1}` | latent dynamics over LM read-outs (§3.3) | small transformer |
| SIGReg / VICReg anti-collapse | same (§3.4) | — |
| 25-d action block = 5×5-d primitives | `H × d_a` continuous action codes (§4.2) | action autoencoder |
| `amax` clamp on plan (σ units) | norm clamp on action codes (§4.5) | — |
| goal = expert frame N steps ahead | goal = spec / failing tests / target state (§5.1) | — |
| `QuasimetricHead` (MRN) | unchanged (§5.3) | — |
| TD, n-step, low expectile τ0.03 | unchanged + **non-unit action cost** (§5.5) | cost-weighted target |
| success = block within 4cm | success = test suite passes, **at rest** (§5.6) | — |
| LIPv4 actor `[A, ∇_A V, V]` | **unchanged** (§6.2) | — |
| CEM 9,000 rollouts | LM proposes N, value ranks (§6.1) | — |
| Dyna WM fine-tune, +4.7 | unchanged (§7 stage 5) | — |

---

## 2. What is the world model

### 2.1 State and observation

The observation at step `t` is everything the agent could see after its last
action. For coding, that is two different things and they must be kept separate:

- **world state** `w_t` — the repo: file contents, git status, build artifacts.
  Changed only by `patch` / `write` / shell side effects.
- **agent knowledge** `b_t` — what has been read, grepped, and observed. Changed
  by *every* action, including read-only ones.

Robotics has no equivalent of `b_t`; the camera always sees. This makes the
setting genuinely partially observed with information-gathering actions. §5.2
shows the temporal-distance value handles it for free, which is the single most
elegant part of this port — do not replace it with a hand-designed reward.

`obs_t = serialize(w_t restricted to what's open, b_t, last tool result)`.

### 2.2 Encoder — the frozen base LM

`z_t = R(LM(obs_t))`, base LM frozen, `R` a learned read-out over the final hidden
states.

Two corrections to the obvious guesses, both from reading the actual code:

**The workhorse latent is a single 192-d vector, pooled hard.**
`lewm.py:31` takes `output.last_hidden_state[:, 0]` — **the CLS token only** — then
a `MLP(192→2048→192)` projector. All the cube/PLDM results in `lip_results.md` run
through that bottleneck. Only the `wm/dinowm/tokens.py` family keeps patch tokens
flat (196×384 = 75,264), and `trm/io.py:92` `CompressedMetric` exists specifically
to squeeze those back down (`mean` / `rp<D>` / `spatial<k>`) before the value head
sees them.

That is *encouraging* for this port, not a constraint: 192 dimensions over 3
frames were enough to plan cube manipulation. So start with a **single pooled
read-out vector, 512–1024-d**, and only escalate to `M` query tokens if the §2.5
probe says the bottleneck is losing `tests_pass` / `n_failing`. Do not begin with
a 64-token read-out on the theory that more is safer — the pairwise feature map is
`4·D` wide (`head.py:37`) and a wide latent makes the value head enormous for no
demonstrated gain.

**In the main line the encoder is *not* a frozen foundation model.** LeWM's ViT is
`pretrained: false` and is trained from scratch *by the WM objective itself*
(`scripts/train/lewm.py:167-181`); it is frozen only downstream, for the value and
actor (`train_lip.py:65-66`). So "frozen pretrained base LM as encoder" is a real
departure. The precedent for it is the *other* line — the pasted-pretrained-bases
work (`tworoom-bases-campaign`, `wm/prejepa/`, `wm/dinowm/`), where LIPv4 was run
on top of frozen external encoders and worked. Cite that line, not the cube line,
when claiming this is proven.

What *is* non-negotiable: the encoder is frozen for value and actor training, and
`rollout` detaches the initial latent (`lewm.py:81`, "to avoid backprop in
encoder"). Otherwise the actor drags the representation into its own optimism.
Note PLDM omits both the detach and the encode cache (`pldm.py:81`) — gradient can
reach its encoder. Do not copy PLDM here.

### 2.3 Latent dynamics — two designs, build both

**Design A — latent WM on a frozen LM (LeWM-faithful, recommended).**
A small transformer `f(z_{t-k..t}, a_{t..t+H}) → ẑ_{t+H}`, `k`≈2 (3 observations,
mirroring the 3-frame window). Rollouts cost `H` forwards of a *small* net, so
hundreds of candidate plans are affordable, which is what makes planner
comparison possible at all.

**Design B — LM-as-dynamics (baseline / upper bound).**
No new dynamics params: `z_{t+1}` is the LM's own state after appending the
action's soft embeddings. Exactly differentiable, zero new parameters, but one
full LM forward per rollout step — so it can *only* be driven by an amortized
planner (§6.1 arm 3), never by search.

Build A as the system and B as the honest ceiling. Note the prediction from the
cube card: **LIP's advantage over search is largest where the WM's native cost is
weakest** (1.93× over-floor ratio on PLDM vs 1.51× on LeWM). A text WM's native
cost will be weak. That predicts the learned-value + learned-planner stack should
pay *more* here than on cube, not less.

**How the action enters the predictor — copy this, it is not the obvious choice.**
The action is **not a token in the sequence**. `Predictor.forward(x, c)`
(`wm/lewm/module.py:285-294`) passes the action embedding as `c`, and every
`ConditionalBlock` consumes it as **AdaLN-zero conditioning**:
`adaLN_modulation(c).chunk(6, -1)` → shift/scale/gate for both attention and MLP,
zero-initialized (`module.py:88-89`). Attention over the state stream is causal.

Two reasons to keep this for language rather than concatenating action tokens:
zero-init means the predictor starts as a pure autoregressive state model and
*learns* to be action-conditioned, and FiLM-style modulation keeps the action out
of the attention pattern, so a 3-observation window stays a 3-token sequence
regardless of how long the action text is. The action encoder itself is an
`Embedder` — `Conv1d(k=1) → Linear → SiLU → Linear → 192` (`module.py:178-209`) —
which takes the action code (§3) to the latent width.

Rollout windowing detail worth preserving: `LeWM.rollout` *grows* the attention
window, `lo = max(0, H + t − HS)` (`lewm.py:97`), so with one observed frame at
eval the window is 1, 2, 3, 3, … Any reimplementation must match this or the
deploy/train conditioning silently diverges — this is what `rollout_compat` in
`solver/lip.yaml` guards.

### 2.4 Objective

Latent prediction loss `||ẑ_{t+H} − sg[z_{t+H}]||²` plus an anti-collapse term.
Both cube WMs needed one (LeWM: SIGReg; PLDM: VICReg + temporal + IDM), and an
IDM auxiliary — predict `a_t` from `(z_t, z_{t+1})` — is worth keeping because it
directly forces the latent to retain *action-relevant* structure, which for coding
means "which file changed and how", the thing most at risk of being washed out.

### 2.5 The probe that decides whether any of this works — run it first

Port the grasp-miss probe verbatim, because it is the cheapest experiment here and
it can kill the project in a week.

1. Train a **linear** readout from a real encoded observation `z` to ground-truth
   quantities: `tests_pass ∈ {0,1}`, `n_failing`, `compiles`, `files_touched`.
   On cube the analogous probe hit **R² = 0.99**, proving the representation was
   fine and the failure was elsewhere. If this probe is weak, the read-out `R` or
   the serializer is wrong — fix that before anything else.
2. Apply the *same frozen probe* to **imagined** `ẑ_H` after rolling out a real
   action sequence, and compare against what actually happened.

The gap between (1) and (2) is imagination optimism, and it is the number that
governs everything downstream. On cube it was catastrophic and invisible until
probed: the WM imagined a successful lift on **100% of real grasp misses**.
Expect the text analog — *the WM will imagine the tests passing* — because agent
traces are curated toward success and contain almost no confidently-wrong patches.

Knowing this in advance, the fix is already specified (§8.1): failure data in the
WM fine-tune mix at ≈10%, never failure-only.

---

## 3. Actions, and how to make them differentiable

### 3.1 Requirements

The planner descends `∇_A V(f(z, A), g)`, so the action must be (i) a continuous
tensor, (ii) bounded, with a norm that means something, (iii) decodable to a
concrete tool call, (iv) `A = 0` must be a sane initialization — LIPv4 starts
every plan at zero and *degrades from any non-zero start*.

### 3.2 Option A — continuous action code + LM decoder (recommended)

Train an **action autoencoder** on the real actions in agent traces:

- encoder `q(z_a | action_text, ctx)` — small transformer, output `z_a ∈ R^{d_a}`,
  `d_a` ≈ 32–64;
- decoder `p(action_text | z_a, ctx)` — the **frozen base LM**, with `z_a` mapped
  by a linear layer to a handful of soft prefix tokens; trained with NLL on the
  action tokens only.

A horizon-`H` plan is then `A ∈ R^{H × d_a}` — structurally identical to the
cube's `5 primitives × 5-d = 25-d` block. Planning is entirely in code space;
decode happens **once**, at execute time.

Two properties must be *measured*, not assumed, because the planner's gradients
are lies if they fail:

- **Cycle consistency.** `q(decode(z_a)) ≈ z_a`. Train with a denoising /
  VAE-style prior so the code space is locally smooth; without it the decoder has
  dead zones where gradients point nowhere.
- **Interpolation sanity.** Walk a line between two real action codes and read the
  decoded actions. If they jump discontinuously between unrelated tool calls, the
  differentiable relaxation is fictitious and you should fall back to Option C.

### 3.3 Option B — soft tokens / straight-through

`a_t = softmax(logits/τ) @ E_vocab` over `L` positions. No new parameters,
differentiable immediately — good as a one-week probe. But the interior of the
simplex is entirely off-manifold for a real LM; this is the textual form of the
"broad false basin" the cube autopsy found. The honest test is the **hard-decode
gap**: does the argmax action achieve the value the soft action was scored at? If
not, discard. Temperature + an entropy penalty play the `amax` role here.

### 3.4 Option C — factored skill codebook

Action = (discrete head over ~10 tool types via Gumbel straight-through) ⊕
(continuous parameter code). Most debuggable, most controllable, and closest to
the cube's factored primitive block. Note this is *factorization, not hierarchy* —
the HWM campaign found **no hierarchy win at any horizon**, so do not layer a
subgoal actor on top of it.

Sequence: B as a cheap sanity probe → A as the real system → C if A's decoder is
too noisy.

### 3.5 `amax` — port the single biggest empirical win

The largest controlled effect in the entire cube line was clamping the plan:
**amax 3.5 → 1.6 was +7.1 points (79.3 → 86.4), and it collapsed seed spread
20.7 → 4.7**, eliminating a catastrophic-seed mode entirely. The variance was the
story more than the mean.

Port: clamp `||z_a||_∞ ≤ amax · σ` in normalized code space, with `σ` from the
pinned action statistics. The mechanism transfers exactly — the clamp keeps the
planned action inside the region where **both** the decoder and the WM are
in-distribution. It matters *more* in language, because an off-manifold `z_a`
decodes to a syntactically plausible, semantically insane tool call.

Sweep `amax` early. It is cheap, it is the highest-yield knob known, and the
plateau (1.4–2.2 on cube) means you only need a coarse grid.

Also port the two deploy-time diagnostics already in `solver/lip.yaml`:
`plan_scale` and `plan_clip`, added because **LIP under-actuates** — plan rms 0.32
vs CEM's 0.58 vs the data's 1.00. A learned refiner picks a limp member of a broad
minimizer set. Check for the same effect in code-space (are proposed diffs
systematically smaller than real ones?).

### 3.6 Action-space pinning

One shared normalization statistics file across WM, value, actor, and decoder.
The puzzle campaign needed exactly this ("ONE shared action pin for LIP
compatibility"); a silent stats mismatch between components is undetectable at
train time and fatal at deploy.

---

## 4. Value: cost-to-go on coding tasks

### 4.1 Goal representation

`g` is encoded by the **same frozen encoder** into the same latent space: the
failing-test identity plus the issue text, optionally the target diff. Because
goals are just encoded observations, hindsight relabeling works — which is the
whole engine of §4.2.

### 4.2 The supervision is free: temporal distance + HER

Any trajectory gives, for every `i < j`:

    d(z_i, z_j) ≤ (number of steps from i to j)

Take state `j` as the goal for state `i`. That is hindsight relabeling and it is
valid on **every trace, successful or not** — a failed run still teaches "from
here, that state is 6 steps away." No reward model, no human labels, no RL.

Keep the cube sampler's structure: balanced full-horizon pairs plus ~30%
cross-episode negatives (unreachable pairs → large distance), which is what stops
the metric collapsing to "everything is close."

**Why this handles information-gathering actions for free:** reading the right
file genuinely shortens the remaining path to green tests, so it gets a lower
distance, with no bespoke curiosity term. Reading an irrelevant file does not.
This is the payoff for keeping `b_t` in the observation (§2.1).

### 4.3 The head: MRN quasimetric, unchanged

`stable_worldmodel/trm/head.py:107` ports as-is:

    d(z_i → z_j) = ||u(z_i) − u(z_j)||₂ + max_k ReLU( v(z_j)_k − v(z_i)_k )

The triangle inequality is what lets the value **stitch** long distances out of
short observed transitions — the mechanism behind flat LIPv4's implicit subgoals.

Asymmetry should pay *more* here than on cube. In code the directional gap is
large and structural: deleting a function is one action, restoring it without
knowing what it was is many. The PushT card noted "QRL asymmetry real but
unpaid"; coding is where it gets paid. Worth an explicit ablation
(`sym_frac = 1.0` vs `0.5`) precisely because it's a domain-motivated prediction.

Feed the head a pooled 256–512-d projection of the `M`-token read-out, learned
jointly with the value (`pair_features` is `4·D` wide, so the full read-out is
impractical).

### 4.4 TD target, and the τ trap

The target (`trm/learners/td.py:6-13`, implemented at `:87-97`):

    reached within n_eff steps  →  target = δ                    (Monte-Carlo, exact)
    else                        →  target = c(n_eff) + γ^n_eff · d_target(z_{t+n}, z_g)
    with c(n_eff) = n_eff  for γ=1,  else (1−γ^n_eff)/(1−γ)

    loss = expectile_Huber( d(z_t, z_g) − stopgrad(target) )

`γ = 1.0` by default: undiscounted steps-to-go, with an exact MC label whenever the
hindsight goal falls inside the n-step window.

**`expectile` and `tau` are two different parameters and this codebase overloads
the name.** In `TDConfig`: `expectile = 0.7` (`td.py:48`) is the asymmetric loss
weight; `tau = 0.005` (`td.py:47`) is the **Polyak/EMA rate of the target
network** (`td.py:101-103`). But `train_lip_ac.py:344` passes a variable *named*
`tau` as the expectile, which is where "τ0.03" in `lip_results.md` comes from. Fix
the naming before porting anything, or the sweep will tune the wrong knob.

The value to carry over is **expectile ≈ 0.03–0.1**. Mechanically
(`td.py:56-59`), `weight = where(diff > 0, 1−expectile, expectile)` with
`diff = pred − target`, so a *low* expectile weights overestimation at `1−e ≈ 0.97`
and underestimation at `≈ 0.03` — it pushes the predicted distance **down**, i.e.
optimistic toward the min, which is what you want when many action sequences reach
the goal and the value should track the best one. Cube: 0.03 beat 0.1 (78 vs 72).

Note the in-code comment `# >0.5 optimistic (shortest-path)` at `td.py:48` and the
`0.7` default are **the bug** the standing lesson records
(`trm-online-expectile`: "TD expectile was backwards"). Do not port the default;
port the swept value.

n-step is load-bearing and domain-specific — a clear inverted-U on cube
(n25=70, n50=78, n100=60), but **n=1 on PushT** because of velocity aliasing. The
coding analog of aliasing is that actions differ wildly in how much they change the
observation (a `read` barely moves the world, a `run_tests` moves the observation
and not the world), so sweep it rather than inheriting 50.

**Do not decorate `cost()` with `@torch.no_grad()`.** Both heads carry a comment
about this (`head.py:89`, `head.py:131`) because it silently killed every
TD+gradient run and presented as a solver bug. The gradient-based planner
backprops through the terminal cost; that is the entire mechanism.

### 4.5 The one required extension: non-unit action cost

Cube steps are all one unit of time. Coding actions are not: a grep is free, a
full CI run is three minutes, a `read_file` costs tokens.

The change localizes to exactly two lines. In `td.py:89-94`, `c(n_eff) = n_eff`
becomes a **sum of per-action costs over the n-step window**, and the MC branch's
`dist = δ` becomes the summed cost over those δ steps:

    c   = cost[t : t+n_eff].sum(-1)          # was: ne
    tgt = reached * cum_cost_to_goal + (1 − reached) * (c + disc * d_next)

`cost[·]` is a scalar per action (wall-clock, tokens, or a blend) that traces
already record. `NStepGoalSampler` already returns row indices
(`samplers.py:203-205`, added so `dwell` could look up ground-truth state), so the
per-step cost column is fetchable with no sampler change.

This is the most important single adaptation in this document — without it the
planner is indifferent between grepping and rebuilding. Cost: it partly breaks the
pure trajectory-index supervision of §4.2, so keep an unweighted head as a control.

### 4.6 Terminal predicate, "at rest", and the dwell head

The goal predicate here is **exact and cheap**, unlike a 4cm threshold: the test
suite passes. Take the at-rest semantics seriously (`969a770`): success should mean
*tests pass and the agent has stopped editing*, not "tests transiently passed while
mid-refactor." Record it as a **per-step** label from the environment's own check
rather than from episode termination — that distinction is load-bearing and was the
subject of the Dyna collection bug (§7.7).

**There is already a head for exactly this, and it fits coding better than cube.**
`trm/learners/dwell.py` is deliberately *not* a quasimetric: it learns a discounted
**time-at-goal** value `V(z,g) = E[Σ γ^k r_{t+k}]` with `r = 1{state within tol of
g}`, then serves the planner via `cost = v_max − V` (`dwell.py:92-94`,
`v_max = 1/(1−γ)`). It uses `PairwiseMetricHead(softplus=False)` because a
discounted return is signed, plain Huber with **no expectile**, and — uniquely —
it reads **ground-truth state, not latents**, for its reward (`dwell.py:131-147`,
raises without `cache.state`).

Port: `r = 1{tests pass}`. That is a cleaner reward than the joint-tolerance ball
ever was, and the ground-truth-state requirement is trivially satisfied because the
test result *is* recorded state. Run dwell as a second head alongside the
quasimetric — the quasimetric answers "how far to green", dwell answers "will it
*stay* green", and the second question is the one that distinguishes a real fix
from a test-shaped patch.

### 4.7 What this deliberately does not have

No reward model, no preference data, no human labels, no RLHF-shaped anything.
The value's only teacher is elapsed cost between states that a real trace visited.

---

## 5. Planner

### 5.1 Three arms, and they must all be built

Mirroring the cube protocol exactly, because it is what makes the result
interpretable:

1. **Native cost + search.** Latent MSE from `ẑ_H` to `g` under the WM, searched
   over sampled action codes. The "does the learned value buy anything" control.
2. **Value + search.** Same search, learned `d(ẑ_H, g)` as the cost. Isolates
   value quality from search paradigm. **In language this is where the base LM's
   prior belongs: the LM proposes N candidate actions, they are re-encoded to
   codes, the WM rolls them out, the value ranks.** Likely the strongest practical
   configuration early on, and unlike cube's CEM it starts from a genuinely good
   proposal distribution.
3. **LIP actor.** Amortized learned optimizer, §5.2.

### 5.2 The LIP actor transfers unchanged — and that is the interesting part

LIPv4 is a 2×512 MLP applied `K` times as a learned refinement rule over the plan.
Its **only** input is

    [ vec(A), vec(∇_A V), V ]

the current plan, the gradient of the value through the frozen WM, and the value.
No raw state. No raw goal. No gate. *"The value is the only teacher, and the only
input."*

That makes the actor **domain-agnostic by construction**: its input dimension
depends only on the plan size. Cube's `5×25 = 125` plan gives a 251-d input; a
`5 × 64` code plan gives 641-d. Same architecture, same width, same `K`. There is
a real chance the cube actor's *recipe* transfers with nothing but a dimension
change — the cube→PLDM transfer already showed architecture transfers while
schedules need re-tuning.

Training: descend `V(f(z, A), g)` through the **frozen** WM and a frozen-or-EMA
critic. Supervised learning through a differentiable simulator; no policy gradient,
no behavior cloning — pathwise only. The value-gradient ablation says this is where
the signal is: `[A, ∇V, E]` 87.6 → `[A, E]` 54.0 — **the gradient carries ~91%**.

Two implementation details that are easy to get wrong (`train_lip.py:128-144`):

- **`gA` is detached.** The gradient feature is an *input to the learned rule*, not
  a training path. Only `theta` receives gradient; `E_feat` and `traj_f` are
  detached too. Getting this wrong turns the actor into a second-order method by
  accident.
- **The loss is terminal-plus-path**: `loss = e_path[-1] + 0.1 * mean(e_path)`,
  i.e. the final value plus a small weight on the mean value across all K
  iterations. `mean_weight = 0.1` (0.3 was mid-pack); `AdamW(3e-4, wd 1e-5)`,
  `clip_grad_norm_(10.0)`, K=8, 8000 steps, batch 128. 6k steps ≈ 8k, so take the
  25% saving.

Training-time context sampling (`train_lip.py:88-104`): 3-frame latent history, 2
real action blocks, HER goal with `p_cross=0.3`, else a future state at
`delta ~ U[1, 10]`.

Cost: ~16 rollout-equivalents per replan vs ~9,000 for CEM. On cube the fair
matrix showed LIP *ties* search on a strong encoder and trails ~4 points on a weak
one at ~450× less compute — "LIP beats CEM" was retracted. Claim the compute
argument, not a quality argument.

### 5.3 Deploy loop

    A ← 0
    repeat K times:  A ← A − actor([vec(A), vec(∇_A V), V])
    clamp ||A||_∞ ≤ amax·σ
    decode A[0] → tool call;  EXECUTE for real;  observe;  re-encode;  replan

Receding horizon, execute the first block only. Three constraints from the cube
line:

- **Plan short: 3–8 actions.** `h200` was a *universal* cliff — latent, all TD,
  and LIP all scored 0.0 — because it crossed the data-support boundary. Robust
  planning extends to roughly ⅓ of an episode span, no further. Do not try to
  imagine a whole refactor.
- **Replanning more often was strictly worse** (60/48 and 58/56 vs 68/72). Each
  replan queries the actor from a *real* agent-visited state that is off the
  training manifold. This is the known unfixed failure (§8.6).
- **Conditioning parity is not optional.** WM, value, and actor must see the same
  history depth (`use_frame_history`, `rollout_compat` in `solver/lip.yaml` exist
  for exactly this). The earlier "LIP beats CEM" claim died when conditioning was
  equalized.

---

## 6. Build order, with a kill gate at each stage

| stage | build | gate before proceeding |
|---|---|---|
| 0 | trace corpus + serializer + per-step outcome labels | ≥10k trajectories; **deliberately collected failures** (§8.1); split policy written *first* (§8.3) |
| 1 | action autoencoder | cycle consistency; interpolation sanity; decode-under-clamp always yields a valid tool call |
| 2 | read-out + latent WM | **probe (§2.5): linear `z → tests_pass` R² high on real obs**; imagined-vs-real optimism gap measured and reported |
| 3 | value (MRN, TD, low τ, cost-weighted) | value+search beats native-cost+search |
| 4 | LIP actor | beats both search arms at *matched* conditioning and matched `amax` |
| 5 | Dyna round | +Δ with 3 seeds, all positive |

Stage 5 is still not RL: it fine-tunes the **world model** (not a policy) on
self-collected on-policy traces with the same supervised latent-prediction loss.
On cube, properly controlled, that was **+4.7, all three seeds positive, p≈0.02**,
and it collapsed seed spread 2.7 → 0.7. Plan for it from day one, because §8.6
says offline-only hits a wall.

---

## 7. Ported failure modes — the checklist

Each of these cost weeks on cube. They are all cheaper to prevent than to
diagnose.

**7.1 Off-manifold / imagination optimism — the #1 risk.**
Cube: WM imagined a lift on 100% of real grasp misses; the probe proved the latent
*could* read block height (R²=0.99), so it was a **data** failure, not a
representation failure — zero attempted-and-missed grasps in 2M expert frames.
Language: agent traces are curated toward success, so the WM will imagine tests
passing. Fix, known in advance: failure data at **≈10%** of the WM fine-tune mix
(the 90/10 mix broke the cube plateau); **failure-only destroys the gradient
field** (−20 points). So: collect failures deliberately — mutate a working repo,
run a weak agent, record the flailing — and keep the mix interior.

**7.2 `amax`.** §3.5. Biggest single win, mostly through variance. Sweep early.

**7.3 Observation-format parity — the sleeper.**
`MUJOCO_GL=osmesa` vs `egl` was worth **+7.3 points** and made every eval silently
out-of-domain. The port: deploy-time serialization must be *byte-identical* to
training — same truncation limits, same diff format, same line-number style, same
tokenizer, same tool-output rendering. A different `git diff` context width is
your osmesa. Hash the serializer config and assert it at eval.

**7.4 Conditioning parity.** §5.3. Never compare across history depths.

**7.5 Horizon / data support.** §5.3. Plan 3–8 actions.

**7.6 The replan boundary — known unfixed.**
Cube's transport failures localized *exactly* at the replan boundary: the actor is
queried from its own mid-execution state, off the expert manifold. Search tolerates
this (coarse ranking suffices); the gradient refiner does not. An offline fix
(training on *imagined* boundary states) was tried and **bounded**: 1/30 cells
flipped. Imagination does not stand in for real visited states. The two live routes
are on-policy data (stage 5) and WM fine-tune on failures (§7.1). Expect this in
language too, and expect it to be worse, because the first real tool result puts
you in a state no rollout visited.

**7.7 On-policy collection: don't let the filter eat your successes.**
Cube: with `terminate_at_goal=True`, successes are exactly the episodes that end
early, so the 25-frame window filter **discarded the successes and kept the
failures** — 407 of 1800 episodes, duplication factor K=87, and a false null
(+2.7, insignificant, negative on one seed). Collecting with
`terminate_at_goal=False` gave 1800/1800, K=18, and the clean +4.7. Port: when
collecting traces, keep rolling past the first green test run, and always check
the duplication factor.

**7.8 Selection statistics.** Single-seed leads under ~10 points are not real —
paid for four separate times. Only seed-replicated, multi-draw means get promoted.
And keep the **anchor gate**: on any new machine, reproduce a known number exactly
before trusting anything.

**7.9 Train/eval contamination.** The standing caveat on the whole cube stack:
WM, value, actor, and Dyna all trained on the same episodes eval draws from, so
every number is an upper bound. This is *worse* in code, where the fix is often in
the repository's own git history. Hold out **repositories**, not just tasks. Write
the split policy before collecting, and write a checker.

---

## 8. What genuinely differs and needs new design

1. **Non-uniform action cost** → §4.5. The one mandatory change.
2. **Information-gathering actions** — a `read_file` changes `b_t` but not `w_t`.
   Handled for free by temporal distance (§4.2); do not hand-design around it.
3. **Exact goal oracle** — tests pass, versus a 4cm threshold. A real advantage:
   the terminal signal is not noisy.
4. **Irreversibility** — `rm -rf`, force-push, a published release. The cube resets.
   This is the one place the spec is genuinely incomplete: it needs action masking
   and/or a separate risk head, and neither is derivable from temporal distance.
   Treat irreversible actions as outside the planner's action space for now.
5. **A strong prior already exists.** Cube had an expert dataset and no prior; here
   the base LM is already a decent policy. That argues the highest-value early
   configuration is arm 2 — **LM proposes, value ranks, WM rolls out** — with the
   LIP actor as the compute-efficiency play once the value is trustworthy.

---

## 9. Minimal first experiment

Two weeks, and it either kills the idea or justifies the rest:

1. Corpus of agent traces on a held-out set of repos, with per-step outcome labels.
2. Read-out + linear probe: can a frozen-LM latent read `tests_pass` / `n_failing`?
   (Cube's analog: R²=0.99. If this fails, stop and fix the serializer.)
3. Train the value **only** — MRN head, hindsight temporal distance, low τ. No WM,
   no actor.
4. Test it as a *re-ranker* over the base LM's own proposed next actions, scored on
   real executed outcomes. Does `d(z', g)` order candidate actions better than the
   LM's own logprob?

Step 4 is the whole thesis in miniature: **a value learned from nothing but
elapsed time in offline traces beats the LM's own preference over its own
proposals.** If that holds, the world model and the differentiable planner are
worth building. If it doesn't, no amount of planning machinery will save it.
