# DATA SPLIT POLICY — read before running any Dyna round

**Rule: the episodes used for on-policy Dyna collection (and therefore for the WM
fine-tune) MUST be disjoint from the episodes the evaluation draws its tasks from.**

This is currently **NOT enforced**, and the round-1/round-2 results in this repo were
produced **without** it. Any number measured before this policy was added should be
treated as an upper bound, not a generalization estimate.

---

## What went wrong

The stack's inherited contract is `train-ds == eval-stats-ds == eval-draw-ds`: v2WM, the
latent cache, the TD teacher and the LIP actor are all trained on the same 10,000-episode
expert set (`quentinll/lewm-cube`), and `eval_wm.py` then draws its 50 tasks from that
same set — with the goal being a frame 25 steps later *in the same demonstration*.

For the base LeWM/DINO-WM numbers that is the published protocol, so it is at least
comparable to prior work. **For the Dyna claim it is not good enough**, because Dyna adds
a second, avoidable leak:

```
collect_r1.sh   ->  eval_wm.py with seeds 1000+   ->  draws start states from ALL 10k episodes
                                                       (~120 calls x 50 envs = broad coverage)
build_*_dataset ->  fine-tune WM1 on those rollouts
eval            ->  eval_wm.py with seeds 42/43/44 ->  draws tasks from the SAME 10k episodes
```

So WM₁ was fine-tuned on rollouts starting at or near the very states it is later scored
on.

## How bad is it, per stage — calibrate before panicking

There is no separate "eval set". There is **one** lance; the eval draw is a
150-task **subsample of** LIP's own training pool (~2M valid (state, goal) pairs).
That distinguishes two very different failure modes, and only the second is live:

| failure mode | risk | live here? |
|---|---|---|
| trained *on* the eval items, overweighted | severe — memorization | **no** — 150 of ~2M pairs, unweighted, and the critic is a 192-d MLP; it cannot have memorized them |
| no holdout at all | can't separate generalization from in-distribution fit; **model selection happens on test** | **yes** |

So the correct worry is *not* critic memorization. An earlier version of this file said
"the leak lands in the load-bearing component" (the value gradient carries ~91% of LIP,
per the [A,gradV,E]→[A,E] ablation 87.6→54.0) as though that were evidence the leak is
biting. It is not — it says only where a leak *would* bite if one existed. Retracted.

The two real bites, in order:

1. **`amax` selection is on test.** The winner was chosen by its score on those same 150
   tasks. With n=150 binary trials SE ≈ 2.9 pts, and a max over ~6 candidates buys
   roughly 3 pts of optimism ⇒ **the winning amax's headline is ~3 pts inflated.** It
   cannot explain seed 0's 68 → 86.7 swing, which is far too large for selection noise,
   so amax genuinely helps; only its reported value is biased.
2. **Dyna's exposure is orders of magnitude above LIP's** — see below. This is the one
   that needs a redesigned control, not just a caveat.

## Why "it's closed-loop control" is not a defence

Closed-loop control does make memorisation harder: the policy must act under real
dynamics, and remembering pixels does not move the block. But that is a statement about
difficulty, not a licence to skip held-out evaluation. The RL literature is explicit that
agents overfit their training environments:

- Cobbe et al., *Quantifying Generalization in RL* (arXiv 1812.02341) — training and
  testing on the same environments "offers relatively little insight into an agent's
  ability to generalize"; agents memorise even randomly generated training levels.
- Zhang et al., *A Study on Overfitting in Deep RL* (arXiv 1804.06893); *Assessing
  Generalization in Deep RL* (arXiv 1810.12282) — same finding; held-out task sets are
  the prescribed remedy.
- OGBench itself (arXiv 2410.20092), whose cube environment we use, is designed to
  "require multi-goal generalization".

## The policy

Canonical split over the 10,000 expert episodes (see `dyna_harness/episode_split.py`):

| range | use |
|---|---|
| episodes **0 – 7999** | on-policy Dyna collection; WM fine-tune data |
| episodes **8000 – 9999** | evaluation tasks ONLY — never collected from, never fine-tuned on |

Rules:
1. **Collection** must restrict its start states to the COLLECT range.
2. **Fine-tune datasets** must contain no episode from the EVAL range — including the
   expert slice that gets mixed in (`build_arm_dataset.py`, `build_r2_dataset.py`,
   `build_redo_dataset.py` all sample expert episodes and must respect the split).
3. **Evaluation** must draw tasks only from the EVAL range.
4. Every reported Dyna number states which split it used. Numbers produced under the
   old no-split protocol are labelled as such.

Unavoidable residual: **v2WM itself was pre-trained on all 10k episodes**, so the world
model has seen the eval episodes' pixels regardless. Fixing that would require retraining
the base WM on a subset. The split above isolates the *Dyna* contribution, which is what
the campaign is actually claiming.

---

# REQUIRED CHANGES

## 1. The Dyna control MUST be changed — blocking for the Dyna claim

The Dyna result as it stands (83.0 → 94.4 at h25, 6 seeds, complete separation) is **not
a valid controlled comparison** and must not be reported as one. Two independent defects,
either of which alone invalidates it:

**(a) Episode exposure, not task exposure.** LIP's leak is 150 pairs in ~2M — negligible.
Dyna's is not remotely comparable: collection ran 40 calls × 3 actors → 1,743 episodes /
80,746 steps, and *every one of those steps* entered the WM fine-tune at 25× duplication
in the winning 50/50 arm. If a collected episode is also an eval episode, WM₁ was trained
heavily and repeatedly on the exact demonstration it is scored against. Run
`episode_split.py overlap` to get the measured number — but the fix is the split, not the
measurement.

**(b) The pre/post arms use different `amax`.** Pre-Dyna is now at amax 2.2/1.8; post-Dyna
was trained at 3.5. The +11.4 therefore confounds the Dyna fine-tune with a boundary-clip
change that is independently worth ~17 pts on seed 0. **This is the larger defect of the
two** and it needs no new protocol to fix — just re-run the post-Dyna side at the same
amax as the pre-Dyna side.

Required design:

```
collect   from episodes 0-7999 only          (needs eval_wm.py episode-range filter)
fine-tune on episodes 0-7999 only            (expert slice AND on-policy)
eval      on episodes 8000-9999 only         (both arms)
pre/post  identical amax, identical seeds, identical everything but the WM
```

Until that has run, describe the Dyna number as *uncontrolled* — not as "+11.4".

## 2. Held-out model selection for `amax` — free, do it now

Do **not** select and report on the same draws. The sweep already computes per-draw cells
in `results/summary_amaxsweep.csv`, so this costs zero extra compute:

- **select** the winning amax on draw **42** alone;
- **report** that amax's mean on draws **43 + 44** only.

This converts tuning-on-test into honest held-out selection. It is weaker than a real
episode split (43/44 are still inside LIP's training pool) but it removes the selection
bias, which is the part that actually inflates the headline.

## 3. Enforcement — `eval.ep_range`, and the two bugs that made it a no-op

`eval_wm.py` **does** have an episode-range filter, `+eval.ep_range=LO:HI`, added by the
reacher campaign in `8fe3d04`. Earlier revisions of this file said it didn't exist; wrong.
But it did not work, for two independent reasons:

1. **KeyError on every intended use.** `ep_range` shrinks `ep_indices` but not the
   dataset, and `max_start_per_row` looked up *every* row's episode in a dict built only
   from the kept ones. Any row outside the range → `KeyError`. **Fixed**: rows outside the
   range are masked out instead. With no `ep_range` the mask is all-True, so the
   no-filter path is byte-identical (verified: same tasks drawn).
2. **The reacher ranges are inert on a 1,024-episode file.** The collector passes
   `+eval.ep_range=0:8000`, but every reacher episode id is < 8000, so *all* 1,024 are
   kept — the filter is a no-op and collection still covers the whole file. The eval side
   passes no `ep_range` at all (grep: `ep_range` appears in exactly one script). Had it
   passed `8000:10000`, `_keep` would be empty and the `num_eval` assert would fire.
   ⇒ **The reacher "held-out draws" in `reacher_dyna_20260724/RESULTS_reacher_dyna.md`
   are not held out**, and the revised headlines there (LeWM 76.9→78.2, PLDM 78.8→89.2)
   still carry the leak they were meant to remove. Reacher needs ranges sized to its
   actual episode count, e.g. collect `0:820` / eval `820:1024`.

### Correct usage (cube, 10k episodes — the ranges here are real)

Keep the **canonical eval seeds 42/43/44**; change only the *pool* they sample from:

```bash
# collection — episodes 0-7999
eval_wm.py ... seed=$((1000+i)) +eval.ep_range=0:8000
# evaluation — episodes 8000-9999, standard seeds
eval_wm.py ... seed=42 +eval.ep_range=8000:10000
```

Same seed + different pool = different tasks, so these numbers are **not** comparable to
the historical 42/43/44 cards; they form a new baseline that both Dyna arms must share.
