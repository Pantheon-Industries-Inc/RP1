# Reacher · window costs, held-at-end, LIP optimization — results 2026-07-29/30

Reconstructed from session records 2026-07-30 while both pods were unreachable
(pod-side CSVs to be committed under `reacher/results_pod/` when a pod answers;
every number below is copied from driver-log cards, not recomputed).

## Protocol (final, user-directed)

- **Setting:** lejepa (LeWM) · canonical `reacher.h5` · eval episodes 8000:10000 ·
  h25 (goal offset 25, budget 50) · CEM 300×10 · EGL · 1-frame policy
  conditioning · deterministic per-task resets · **plain LIP deploy (no restarts)**
- **Primary metric: HELD-at-end** — all joints within 0.05 rad at the final step
  (user call 07-29, supersedes latched). Latched (authors' convention) kept as a
  comparability column; both are printed by every run.
- **LIP training seeds fixed at {0,1,2}**; card = 3 train seeds × 6 eval seeds
  (42–47), n=50 each. Single-training-seed screens are banned (misled twice).
- **No privileged objectives** for headline LIP (act-penalty results archived
  as an explicit-prior reference, not headline).

## Headline tables

### Window cost (3-frame, goal tiled) — held-at-end, n=6/actor

| arm | h25 | h50 | h100 |
|---|---|---|---|
| **LIP · window, plain (pooled s0-2)** | **44.2** (35.0/54.0/43.7) | **59.8** (44.3/67.3/67.7) | **63.0** (50.7/68.3/70.0) |
| Latent+CEM · window (unlearned L2) | 42.7 | 42.3 | 38.0 |
| TD+CEM · window (learned quasimetric) | 11.7 | 13.7 | 12.3 |

### Same arms — latched (authors' metric), for comparability

| arm | h25 | h50 | h100 |
|---|---|---|---|
| LIP · window plain | 90.6 | 97.9 | 97.4 |
| Latent+CEM · window | 89.7 | 92.3 | 79.7 |
| TD+CEM · window | 86.7 | 98.0 | 99.0 |

### Terminal-frame (1-frame) references — h25, n=6

| arm | held | latched |
|---|---|---|
| LIP · terminal, pad-context (best s0 / pooled) | **57.0 / 48.6** | 85.3 / 82.9 |
| Latent+CEM terminal (= the paper's own method) | 41.7 | 82.7 |
| TD+CEM terminal (same critic LIP distils) | 35.7 | 81.3 |

Paper-exact anchor (latched, n=6): Latent+CEM 82.7 lejepa / 75.3 pldm
(n=3: 86.7/77.3) vs published 86/78 — reproduced. dinowmnp (pixels-only
conversion) 65.3 vs published 79 (variant mismatch: theirs had proprio).

## Findings (each at n=6 eval seeds unless noted)

1. **Value-vs-planner: the planner is the strong part.** Fixed critic, swap the
   search: TD+CEM 35.7 → LIP 48.6 pooled (+12.9; +21.3 best actor). Meanwhile
   every deployable cost is near-chance at settling (stopped-vs-moving at goal:
   1-frame quasimetric AUC 0.505, window3 0.610, window-L2 0.631; achievable
   0.886 at an undeployable 1-step lag), and the random-play data contains
   almost no holding (0.4% of at-goal states still at goal 25 steps on).
2. **The 3-frame window is a latched-only win.** Latched it beats every terminal
   cost (+7 unlearned); held it is a wash for Latent+CEM (42.7 vs 41.7) and
   catastrophic for TD+CEM (35.7 → 11.7). Attribution (unlearned window-L2 89.7
   latched vs learned 86.7): the latched gain is information access, not metric
   learning.
3. **Horizon story:** latched — learned costs climb with horizon (TD+CEM-w 99.0
   at h100) while unlearned window-L2 collapses (79.7); held — LIP-window grows
   to 63.0 pooled at h100 (best held numbers of the campaign) while both CEM
   arms fall. LIP is top-or-joint-top at every horizon under both metrics, at
   ~16 rollouts vs CEM's 3000.
4. **TD hypers are not the lever.** Full τ∈{0.05,0.1,0.2} × n∈{5,25,50,100}
   grid at 6 eval seeds: best arm +5.6 over canonical τ0.1/n50 — inside the
   ~5.7 noise band. (3-seed screens of the same grid were wildly non-monotone;
   regression-to-mean confirmed on extension.)
5. **LIPv4 hypers around the centre mostly hurt.** 10-arm OFAT (latched-era,
   1 train seed, held-rescored): every arm ≤ centre on held. amax 1.8's apparent
   win was training-seed noise (3-seed: 33.4 vs centre 44.2 held). Combo
   amax1.8+iters6 lost to amax1.8 alone (latched 84.0 vs 88.3, 1 seed).
6. **Act-penalty (λ·mean(A[:,-1]²), PRIVILEGED — archived, not headline):** the
   only lever that reproducibly moved held: +6.7 at matched amax (44.0 → 50.7,
   both train seeds), best config amax1.8/λ0.05 = 53.8 pooled (2 seeds).
   Screening cards before the pivot: a1.4/λ0.2 56.7, a2.0/λ0.2 56.0,
   a1.8/λ0.05 56.3 (1 train seed each). Excluded from headline because the
   baselines get no equivalent settling prior; fair variants: add the same term
   to CEM's objective, or the implicit zero-action-tail "hold test" (proposed,
   unimplemented).
7. **Fixed on the way:** authors' latched success is their code (PR #158);
   eval nondeterminism root-caused (unseeded per-reset target ball, ±7-14/cell)
   → per-task reset seeds; Dyna r1 mix leaked 400k eval-range rows (10.1%) →
   rebuilt clean (3.2M rows, on-policy frac 0.500, expert 0:8000 only);
   `cache_latents` compress-meta bug; checkpoint-ambiguity normalisation.

## In flight at write time

- **Wave A (new pod, driver confirmed alive pre-outage):** window-held LIP,
  non-privileged arms {mean-weight 0.5, steps 16k, replay 0.3, bc 0.1} × seeds
  {0,1,2}, vs centre 44.2 / target ≥ ~51.
- **Dyna round 2 (old pod, driver confirmed alive pre-outage):** clean-mix
  fine-tune done; resumes at caches → window value on FT latents → LIP s0-2 →
  held cards vs 44.2/42.7. Hypothesis: on-policy data contains holding, so Dyna
  should move held specifically.
- Both pods went unreachable ~09:30 UTC 07-30 (network/MooseFS trouble);
  drivers are setsid-detached and CSV-idempotent. Volume `3cv5zzezm9` (187G)
  holds everything old-pod; a replacement pod can attach it directly.

## Reproducibility

Every driver/patch from this campaign is committed under
`reacher/scripts_20260729/` (pod copies live in `/workspace/` on both pods):
window value + L2 control (`train_window.py`, `make_l2window.py`), vframes=3
LIP plumbing (`patch_vframes3.py`), pad-context + act-penalty + param-free
critic trainer patches, detseed, clean-mix rebuild, the sweep/eval drivers, and
the value-vs-planner diagnostic (`diag_value_vs_planner.py`).
