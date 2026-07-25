# Small-Init: Initialization as the Whole Answer

*Why `--head-scale 0.01` replaces both the zero-init trick and the GD-prior machinery for the v3 refiner. 2026-07-13.*

## 1. The setting: a residual rule unrolled K times

The v3 planner is a residual update rule applied K=8 times inside **one** training graph:

$$A^{k+1} = \text{clip}_{\pm 3.5}\big(A^k + g^k \odot \Delta A^k\big), \qquad (\Delta A^k, g^k) = f_\theta(\text{tokens}(A^k, \hat z^k, V, k))$$

Training backprops the terminal value through all 8 applications of $f_\theta$ *and* 8 WM unrolls.
Whatever $f_\theta$ does at initialization is therefore applied — and compounded — eight times
before the loss ever measures it. Initialization is not a detail here; it decides what kind of
dynamical system early training has to control.

## 2. Two failure modes, one axis

**Default init → compounding noise.** At PyTorch-default init, $\Delta A$ is a random function
with O(1) outputs and the gate sits at $\sigma(0) = 0.5$. Each iteration applies a ~0.5-scale
random edit; across 8 iterations the plan performs a random walk that early training must first
unlearn. This is not hypothetical: round V's training curve runs ~2× behind the v1 MLP's at the
same step (E_final at step 1000: 7.6–11.2 across seeds vs 4.6 for v1) — the transformer pays
more for the noisy start than the shallow MLP ever did.

**Exact-zero init → dead gradients.** Zeroing the head ($W, b = 0$) makes the refiner the exact
identity — clean forward, but the backward path dies: $\partial\,\text{out}/\partial\,\text{features} = W = 0$,
so the encoder, input projection, and positional parameters receive **zero gradient** at step 0.
Worse, the gate column is doubly dead ($\partial/\partial W_\text{gate} \propto \Delta A = 0$).
Only the dA-half of the head learns at first; everything upstream trains blind until it grows.

The key observation: **both are scale problems, not structure problems.** Default init has the
right derivatives and the wrong output magnitude; exact-zero has the right output magnitude and
no derivatives. Those are the two ends of one dial.

## 3. The mechanism: ε instead of 0

Scale the default head init by $\varepsilon = 0.01$ (biases zeroed):

$$W_\text{head} \leftarrow \varepsilon \, W_\text{head}^\text{default}, \qquad b_\text{head} = 0$$

**Forward at init — near-identity.** Encoder features are O(1) (post-LN transformer), so per-entry
updates are $O(\varepsilon)$: measured max drift < 0.02 z-scored action units per iteration
(vs $a_\text{max} = 3.5$). Eight iterations move the plan by ≲ 0.1 total — the unrolled chain is
effectively linearized around $A = 0$; there is no random walk to unlearn.

**Backward at init — everything lives.** $\partial\,\text{out}/\partial h = W \neq 0$: every upstream
parameter gets gradient from step 0 (verified: all 24 encoder parameter tensors have nonzero
grads on the first backward). And the *head's own* learning signal is not ε-suppressed —
$\partial L/\partial W_\text{head} = h \cdot \delta$ with $h = O(1)$ is full-size — so the head grows at
normal speed, and as it grows, upstream gradients scale up with it. Small-init thus behaves like
an **implicit warmup schedule**: the network turns itself on smoothly, no lr-warmup hyper needed.

This is established practice under several names — GPT-2's residual-projection scaling,
Fixup's final-layer scaling, muP's output-layer treatment, and ControlNet's zero-conv taken to
ε instead of 0 precisely to keep the upstream tap open. We are applying the standard trick at
the one place the unrolled structure makes it critical: the output head of a K-times-composed
residual function.

## 4. Why "starts with no algorithm" is acceptable

The argument for baking gradient descent into the update rule (the η·ĝ bolt-on, or the
preconditioner head $\Delta A = -\text{softplus}(p) \odot \hat g + r$) was that zero-init otherwise starts
as "no algorithm." But the gradient is already an **input feature** — a 25-dim slice of every
token. The network doesn't need descent wired into its output algebra to use descent; it needs
weights that copy and transform that input feature, which is a few gradient steps of learning,
not a structural necessity. The empirical precedent is v1 itself: its champion (88/96/80) learned
excellent gradient use from *plain default init*, with no prior, ever.

Every "starts as GD but can learn more" design is necessarily $\text{GD} + \text{learned
correction}$ — a sum of two mechanisms, however it's dressed (external η term, preconditioner +
residual, full learned preconditioner, attention over gradient values). The composition is the
price of the prior. Small-init refuses to pay it: the prior stays in the *features*, the
architecture stays a single free update rule, and initialization alone delivers the training
stability the prior was supposed to buy. If a descent prior ever earns its structure, that's an
empirical claim the shelved precond head can test in one arm.

## 5. What small-init does *not* fix

- **Input scale imbalance.** Tokens still mix raw scales (value ~O(25), WM gradient of arbitrary
  magnitude, raw latents). That's the orthogonal `--feat-norm` lever, deliberately not bundled.
- **Anything structural.** If v3's gap to v1 is not an init artifact — wrong token design,
  wrong capacity, wrong horizon conditioning — ε changes nothing. That's exactly what makes
  round G a clean experiment: it isolates the init axis.
- **ε is still a choice**, though a forgiving one: anything in ~[0.001, 0.1] should land in the
  same basin (near-identity forward, live backward). 0.01 is the conventional default.

## 6. How to read the results

Four init regimes, one architecture (vonly tokens, scalar iteration, gate head):

| regime | step-0 forward | step-0 backward | run |
|---|---|---|---|
| default | random ~0.5-scale edits ×8 | all live | round V |
| exact-zero | identity | head-dA only; encoder dead | round Z |
| **small-init ε=0.01** | **near-identity (<0.02)** | **all live** | **round G (proposed)** |
| precond (GD prior) | ≈ normalized GD, step 0.05 | all live | shelf |

Predictions if init noise was the binding constraint: round G's E_final curve tracks or beats
v1's from early training (≤ ~5 at step 1000), smaller seed spread than round V, eval ≥ round V's
cells. If G ≈ V on evals, init wasn't the problem — attention moves to token design/capacity.
If Z matches G, gradient liveness didn't matter and identity-start alone sufficed (would be
surprising given the blocked-encoder analysis).

## 7. Config

```bash
python scripts/plan/train_lip_ac.py ... \
  --arch traj --goal-mode vonly --iter-mode scalar \
  --head-scale 0.01          # the entire change
```

One scalar. No new parameters, no new terms in the update rule, no second head, no η, no p0.
Composes with everything except `--zero-init` (which would re-zero the head; the two are
alternatives on the same dial). Checkpoints carry `head_scale` for provenance only — loading is
unaffected since the trained weights are what's saved.
