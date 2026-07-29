# Dyna on LeWM cube-single — campaign results

**2026-07-22 → 07-29 · pod 157.66.254.11:15788 (4×H100) · branch `eval-sweep`**
Read with: [`HANDOFF_20260728.md`](HANDOFF_20260728.md) (running log),
[`HYPERPARAMS.md`](HYPERPARAMS.md) (recipe), [`DATA_SPLIT_POLICY.md`](DATA_SPLIT_POLICY.md),
[`RESULTS_amax_sweep.md`](RESULTS_amax_sweep.md).

## TL;DR

**Dyna works on LeWM: +3.6 points, six seeds, all positive, p≈0.002 — and the
effect replicated across three independently built fine-tune datasets (+4.2 to
+4.6 each).** The other genuine win of the campaign is `amax 1.6` (+7.1 over
the old default, and it kills the catastrophic-seed mode). Everything else we
tried on top — more failure data, more distinct data, deeper off-manifold
coverage, harder optimization — is a **measured null**: the recipe saturates
at ≈92 on this eval, and the eval itself cannot resolve refinements under
~3–4 points. The campaign also built reusable machinery (offline outcome
relabeling, outcome-stratified mixing, failure-state chaining) and closed with
an unusually complete set of retractions.

| arm (held-out eps 8000–9999, amax 1.6, egl, 3 draws × 50 tasks) | s0 | s1 | s2 | s3 | s4 | s5 | mean |
|---|---|---|---|---|---|---|---|
| PRE (LIPv4 on frozen v2WM) | 88.7 | 86.0 | 88.7 | 87.3 | 86.7 | 85.3 | **87.1** |
| **POST-full** (Dyna φ=0.08, K=18) | 92.7 | 92.0 | 92.7 | 89.3 | 90.0 | 87.3 | **90.7** |
| POST-thin (K=87, control arm) | 92.7 | 90.7 | 88.0 | — | — | — | 90.4 |
| f13 (Dyna φ=0.13) | 92.0 | 90.7 | 93.3 | — | — | — | 92.0 |
| pool (Dyna φ=0.08, combined pool, K_f=5) | 90.0 | 92.0 | 94.7 | — | — | — | 92.2 |
| lrsw_it16 / s12k / alr (planner-side arms) | | | | | | | 87.8 / 88.7 / 85.6 |

---

## 1. Headline: Dyna is real — +3.6 at 6 seeds, three replications

Protocol: episode-disjoint (train/collect on episodes 0–7999, eval draws from
8000–9999 only, latent caches filtered too), matched amax 1.6, shared PRE arm,
`terminate_at_goal=False` collection, 50/50 expert:on-policy fine-tune of v2WM
(lr 1e-5, 2 epochs, epoch 1 pre-registered), then fresh LIPv4 actors on the
fine-tuned WM.

- **6-seed paired delta: +3.56 ± 1.50, t = 5.80, df = 5, p ≈ 0.002**, every
  seed positive (+4.0/+6.0/+4.0/+2.0/+3.3/+2.0).
- **Three independent fine-tune datasets reproduce it**: POST-full (+4.6 at
  s0–2), f13 (+4.2), pool (+4.4). Different mixtures, different pools, same
  ~+4.
- The 3-seed point estimate (+4.7) was ~1 pt lucky; seeds 3–5 traded point
  estimate for an order of magnitude in significance. The 3-seed claim that
  Dyna collapses seed spread to 0.7 **did not survive** (6-seed spread 5.4).

## 2. The other real win: amax 1.6

45-eval sweep, 3 seeds × 3 draws, full-pool: flat plateau 1.4–2.2
(85.6/86.4/86.2/85.8), both tails fall off. **The variance is the story**:
seed spread 20.7 → 4.7 and the catastrophic-seed mode (seed 0 = 68.0 at
amax 3.5) disappears. Honest-selection check (select on draw 42, report on
43/44) picks 1.4 and reports 86.3 vs the naive 86.4 — the tuned-on-test bias
was small *because the plateau is flat*. Full-pool headline LIP number:
**86.5** (86.4 episode-split-era osmesa; egl and osmesa measure equal on cube).

## 3. Why the effect was almost missed: the duplication control

The WM needs a 25-frame window, so episodes shorter than 25 steps are dropped.
With `terminate_at_goal=True` and an ~84%-success actor, **the successes are
exactly the episodes that end early** — the filter discards them:

| | thin | full |
|---|---|---|
| episodes kept | 407/1800 (22%) | 1800/1800 |
| on-policy rows | 18,500 | 90,000 |
| duplication K for 50/50 | 87 | 18 |
| Dyna delta (3 seeds) | +2.7, p≈0.25, seed 2 *negative* | **+4.7, p≈0.02** |

Both arms rolled identical start states with identical seeds, so full − thin
(+2.0) isolates the data regime. Shipped alone, arm 1 would have concluded
"Dyna does nothing." Row-level truth (established later): the two arms hold
the **same ~14.5k failure rows**; full adds ~75k success rows and cuts K —
nothing was ever de-biased. Which of {distinctness, failure share} carried the
+2.0 was then answered by the nulls below: distinctness.

## 4. The null trilogy: the recipe is saturated at ≈92

All three follow-ups came back clean nulls, which together close the recipe
class:

1. **Failure fraction (f13):** raising the failure share of the mix
   0.08 → 0.13 (+62% failure mass, identical rollouts, same duplication
   regime): **−0.44 ± 1.02 paired** (t=−0.76).
2. **Pool quality (pool):** +47% distinct on-policy rows (90k → 132.5k), K
   18 → ~5, plus 39k rows of chained deep-failure coverage, at fixed φ=0.08:
   **−0.22 ± 2.34** (t=−0.16).
3. **Optimization arms (lrsw):** at amax 1.6 — `iters 16` = **87.8, PRE
   exactly** (the same knob lost 8 pts at amax 3.5: the clip really did wall
   off action exploitation; there was just nothing to gain); `steps 12k` +0.9
   (noise); `actor-lr/3` −2.2.

Conclusion: at this recipe, {mixture ratio, data volume, duplication,
off-manifold coverage, inner iters, training length, actor lr} all have **zero
measurable headroom**. Dyna's +4 is the plateau, ≈92 is the ceiling, and with
n=150/arm-seed (paired SE ~1.4–2.4) the eval cannot resolve anything finer.
The next real moves are architectural or a **harder eval** (draw 43 is
saturated at ~94; draw 44 does all the discriminating).

A per-registered probe of "is 92 recipe-bound or eval-bound" is in flight as
of this writeup: the full Dyna loop on the s12k base (`dyna_s12k_loop.sh`,
PRE = 88.7). POST above 92.4 ⇒ recipe-bound; POST ≈ 92 ⇒ eval ceiling.

## 5. Machinery built (reusable beyond this campaign)

- **Offline outcome relabeling** (`relabel_onpolicy.py`): recovers per-step
  success labels for already-collected rollouts with no replay and no env —
  the task draw is deterministic in the collection seed, zero-drop full-budget
  collection makes recorded-episode↔task mapping exact, and the goal is read
  from expert `privileged_block_0_pos` at (episode, start+25). Three gates,
  evaluated before anything is written. Live run: 13 s, **1,511/1,800
  successes vs ~1,507 predicted**, 36/36 per-call floors clean.
- **Recorder outcome column** (`world.py`, `SWM_RECORD_OUTCOME`): on-policy
  collection is born labeled; sourced from the env's own per-step success
  flag, NOT `terminateds` (which never fires under `terminate_at_goal=False` —
  the naive label would silently mark every full-budget episode a failure).
- **Outcome-stratified mixing** (`build_dyna_mix.py --failure-frac`):
  duplicates failure/success pools at separate rates so failure share and
  on-policy share move independently; per-lance keep-sets (recorded ids
  restart per lance — a union keep-set leaks cross-lance); refuses
  K > 30 (`--max-dup`); hard post-write composition self-check.
- **Failure-state chaining** (`build_chain_dataset.py` + `chain_collect.sh`):
  packages mid-failure states (t ∈ {15,25,35}) as 26-frame pseudo-task
  episodes the *unmodified* eval path consumes — one valid start per episode,
  so 51-episode `ep_range` slices give deterministic coverage with zero
  cross-call duplicates. Result: 850 chains, **8.1% recovery** (even
  cross-actor — mid-failure states are objectively lost under this policy
  class), 781 new distinct failure episodes (pool 3.7×). The chained coverage
  did not move the planner through the WM (null #2), but the recovery-rate
  measurement stands on its own.

## 6. Retractions and corrections (in chronological order)

1. **"+11.4 from Dyna" — retracted.** Confounded: amax change worth ~7 pts +
   on-policy data collected from the eval pool. Controlled effect: +4.7 → +3.6.
2. **"Needs 3.3 h re-collection for outcome labels" — wrong.** Relabeling is
   13 s of arithmetic (see §5); re-collecting would also have *added* a
   data-reshuffle confound.
3. **"Collection costs ~3.3 h" — wrong.** Measured 77–86 min: the vectorized
   runner is paced by cap-length failures in both arms;
   `terminate_at_goal=False` cost +9 min, not 2.3×.
4. **K-table shift.** 18/22/44 belong to φ=0.08/0.10/0.20, not
   0.08/0.20/0.40 — caught by the guard it motivated (φ=0.20 would have been
   K_f=45, half-way back to thin's regime).
5. **"full de-biased thin's failure-heavy data" — wrong framing.** Full kept
   thin's failures bit-identically and added successes (§3).
6. **"Dyna collapses seed spread to 0.7" — 3-seed artifact** (6-seed: 5.4).
7. **Handoff wording "authors' exact checkpoint" for v2WM** — long-standing,
   wrong (in-house replication; see `lewm-84-replication` note).

## 7. Validity

- **Split**: train/collect 0–7999, eval 8000–9999, strict (fs1/fs5 caches
  filtered; collection asserts `ep_range` applied per call; eval refuses to
  record without it). Verified arithmetically (1,608,000 = 8000×201 rows;
  352,000 = 2000×176 valid starts).
- **Residual leak, must be stated in any publication**: v2WM itself was
  pre-trained on all 10k episodes. Fixing it needs a base-WM retrain on the
  split. Empirically the critic-side leak measured *negligible* (LIP 87.8
  held-out vs 86.5 in-pool — better on unseen episodes).
- **There is no eval set** — the expert lance is a task generator: starts are
  expert states (teleported qpos/qvel), goals are t+25 expert configurations,
  2× budget, success = block within 4 cm. Every start is on the expert
  manifold; recovery from off-expert states is never tested. Draw seed = task
  draw; 42/43/44 spread ~20 pts; 43 saturated.
- TD teacher is a single seed shared by every LIP run (reported spreads
  understate true variance). amax was originally selected on the eval draws
  (bias measured small, §2).

## 8. Ops (measured times + the expensive lessons)

Stage times (H100, egl): eval cell 1.9 min · collection 1800 eps ≈ 80 min ·
mix ≈ 2 min · WM fine-tune 3h20–4h40 · caches 22 min · TD 2 min · LIPv4
6k-step 47 min (12k: ~1h50 wave). Training (LIP/FT/caches) has no EGL and
co-resides safely across GPUs; **renders never overlap anything** (3-way
SIGABRT, 2-way silent corruption, historical 8-pt error).

New gotchas this campaign: per-lance keep-sets (union bug shipped a 0.609
mix; caught by the achieved-composition gate in 15 min); arch-config
validation by `cmp` against the copy source, never by content grep
(`load_state_dict` is a *loader* key — grepping for it killed a finished
fine-tune); `pgrep -f "pat[t]ern"` matches your own ssh wrapper's cmdline —
launch with pattern-free wrappers; the PLDM checkpoint keys are
transformers-version-specific (old↔new ViT layout, `invert_pldm_keys.py`).

## 9. Assets

Pod: WMs `dyna_{dsp,full,f13,pool}_5050` + `dyna_s12k_5050` (in flight);
actors `lip4_{dsp_pre,rep_thin,rep_full,f13,pool,lrsw_*,s12kpost}_s*`;
labeled lances `onpolicy_full_a*_lab`, `chained_a*`, `onpolicy_s12k_a*`;
chain tasks `chain_tasks.lance`; caches/TDs per WM; all scores in
`results/summary_dynasplit.csv`. Repo: everything under `Dyna/dyna_harness/`,
fixture-tested (relabel, mixer, chain builder each have end-to-end tests).

## 10. Open threads

1. **s12k Dyna loop** (running): the recipe-bound-vs-eval-ceiling probe.
2. **PLDM base bring-up** (running): frozen PLDM + canonical LIPv4 + CEM/TD
   references under this split — the cross-base comparison arm.
3. Harder eval: drop draw 43, add draws / longer horizons; nothing below
   ~3–4 pts is measurable at n=150.
4. Base-WM retrain on 0–7999 to close the last leak.
5. Weighted sampler to replace physical duplication (hygiene, not expected
   to move numbers — K was measured a null down to K≈5).
6. Mode-stratified mixing (classifier over labeled qpos trajectories exists
   in outline; φ was a null so only worth it behind a harder eval).
