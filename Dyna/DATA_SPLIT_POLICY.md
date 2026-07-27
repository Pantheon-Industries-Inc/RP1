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
on. Exact (episode, start-step) collisions are unlikely — 50 rows drawn from ~1.7M valid
start rows — but **episode-level overlap is expected to be large**, and the Dyna gain is
the novel claim, so it is the one that most needs the control.

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

## Status

- `episode_split.py` — canonical ranges + `assert_disjoint()` + an overlap checker that
  reports, for a given collection lance and eval seed, how many episodes are shared.
- `eval_wm.py` has **no episode-range filter yet**; adding one (e.g.
  `eval.episode_range=[8000,10000]`) is required before rule 3 can be enforced.
- Until then: run the checker and report the measured overlap alongside any Dyna result.
