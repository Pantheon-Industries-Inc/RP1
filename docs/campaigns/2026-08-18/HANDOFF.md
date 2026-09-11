# HANDOFF — unified-γ cross-environment campaign (2026-08-18/19)

Results doc: [RESULTS_unified_gamma.md](RESULTS_unified_gamma.md) (tables +
mechanisms; this file is operational state). Branch
`codex/eval-sweep-refactor`, everything committed. Infra:
`scripts/sky/unigamma/`.

## The question and the pinned design

Can RLP's critic discount be unified across environments, and at what cost?
User decisions that define the protocol:

- **Train everything ONCE at amax=2.5, max_delta=12** (later md=20 arms — see
  in-flight). One actor per arm, evaluated at h25+h100 (TwoRoom/Cube) or
  tau 0.1/0.05 (Reacher) from the same checkpoint. **amax is adapted only at
  deployment** (checkpoint top-level `amax` rewrite — the solver has no
  deploy-clip config path).
- Arms: γ ∈ {0.98, 0.99, 1.0} × vnorm {none, log}, n-step 50. Cells: TwoRoom
  LeWM/lejepa (SPLIT=1 held-out), Cube LeWM (held-out by construction),
  Reacher LeWM/lejepa. Train seeds 0/1/2, report draws 42/43/44, 50 eps/cell.
- Everything else stays at each cell's campaign anchors (mw/lr, steps/batch,
  teacher spec), so γ and the clip are the only deltas vs banked recipes.

## State: DONE

1. **Legacy-target γ dose–response, all 3 envs** (fleet, 18 jobs):
   - TwoRoom h100: **γ-flat** (88.0 / 88.0 / 87.8 raw-E); h25 ceiling. The
     2026-08-17 γ=0.98 falsification (−39) was a **γ×clip interaction** —
     collapse exists only at a1.8. vlog: banked +8.7 was clip-specific; at
     a2.5 it is neutral (γ=1) or harmful (γ=0.98: 74.9).
   - Cube: γ-flat (h100: 81.1 / 78.4 / 80.0), no γ=1 collapse. Unified cost
     vs banked tuned: −2/−1.
   - Reacher (held convention!): **bespoke γ=0.98 wins outright**
     (94.0/55.1 vs 85.3/40.9 and 87.6/46.7), seed variance explodes at
     γ≥0.99. Mechanism: cross-episode pairs (30%) can never receive exact
     labels (sampler labels same-episode only) and everything in reacher is
     ≲50 steps apart — γ=0.98's ceiling 1/(1−γ)=50 ≈ env diameter is the
     ANCHOR for those pairs; γ↑ unanchors them → embedding churn → τ0.05 and
     seed stability degrade first. γ is a horizon budget, applied downward.
   - TwoRoom's h100 sd~9 on raw-E arms = one md12 actor extrapolating outside
     its trained E-band at h100; γ=1 extrapolates linearly (sd 3.9), vlog
     conditions it away (γ=0.99+vlog sd 0.8).
2. **TD-seam test (user observation)**: target is non-monotone at the n-step
   window for γ<1 (V(50)=50 vs V(51)≈32.2 at 0.98/n50). Implemented
   `--boundary {legacy,smooth,disc}` in both teachers + both co-train
   trainers (`overlays_boundary/`). **Verdict: behaviorally cosmetic; no fix
   beats legacy anywhere; smooth mildly harmful. Keep legacy.** Also
   discharges the γ-correlated-confound worry and pins reacher's γ story on
   anchoring, not the cliff.
3. Banked context: TwoRoom a1.8 rescue arms — 0.98/n100 85.1±3.3 (n100 ≡
   n200 bit-identical: episode length caps the window), 0.996/n50 87.1±1.0
   (vlog 91.3), vs γ=1+vlog record 97.3.

## State: CAMPAIGN COMPLETE v2 (2026-08-21) — final protocol & table

Final protocol (user-approved): **uniform per-run early stopping** (snapshots
every 2k steps, val-argmax on draws 50/51), **n=6 seeds**, one config
(γ=0.98/n50/none/legacy/a2.5-train/md20/ema0.005), reporting **median
primary** + mean±sd + IQM + per-seed appendix; no cross-seed selection
(restarts-as-recipe explicitly rejected by user). Final table in
RESULTS_unified_gamma.md "FINAL protocol table". Highlights: Reacher beats
the paper on all four columns as a plain mean; Cube-PLDM h100 +4.3;
TwoRoom-LeWM par on median (94.0 vs 94.2); deficits Cube-LeWM ~−1.5 and
TwoRoom-PLDM −6.6. Seed-variance mechanism pinned by elimination: the
actor's out-of-band refinement rule is a per-run lottery — healthy training
curves, teacher-swap null, critic stabilizers re-roll it; ema_tau 0.002
helps 5/6 cells but Cube-LeWM vetoes (80.9→75.0), so it ships as analysis,
not config. mujoco<3.12 pinned (3.12.0 is sdist-only on PyPI).

## Earlier close-out (2026-08-19; superseded above)

Both closing waves landed — see RESULTS_unified_gamma.md "md20 wave",
"Deploy-amax pass" and "Conclusion". Outcomes: md20 push-up hypothesis
refuted on TwoRoom (87.1 ≈ 88.0) but md20 adopted anyway (free on Cube,
+1.8 τ0.1 and 4× τ0.05 seed-sd cut on Reacher); deploy clip is a clean
null in [1.6, 3.0] — the h100 spread is per-seed actor quality (weak seed
74–78 at every clip, strong seeds 95–96 at deploy 1.8). Final unified
config: γ=0.98, n=50, vnorm=none, boundary=legacy, amax_train=2.5
(deploy free), md=20. Open frontier: seed-to-seed stability of the raw-E
actor's out-of-band extrapolation at γ<1.

## Original in-flight notes (for provenance; jobs 8565–8583)

1. **md20 wave** (`*-g98md20*`, 9 jobs): the push-up hypothesis. TwoRoom's −9
   vs banked 97.3 lives in the band×conditioning interaction (md20 alone +0,
   vlog alone +0, together +8.7 — all at a1.8). vlog is vetoed as compressor
   (reacher γ≥0.99; tworoom 0.98/md12), so the candidate unified recipe is
   **γ=0.98 / raw E / md20 / a2.5** — the discount caps E at 50, inside the
   trained band by construction. Jobs: tw ctrl+vlog ×3 seeds
   (`rlp-tw-unig-g98md20-sN-20260819`), tw smooth-boundary probe ×3
   (`g98md20sm`; carried because the md20 band straddles the n=50 seam),
   cube packed (`rlp-cu-unig-g98md20-20260819`), reacher none+log
   (`rlp-re-unig-g98md20-{none,log}-20260819`).
   - Reads: tw-md20-ctrl ≫ 88 ⇒ discount-as-compressor confirmed, unified
     recipe = 0.98/md20; ≈ 88 ⇒ the interaction needs vlog specifically
     (check the vlog rows); md20 hurts reacher ⇒ band must be stated as
     "min(horizon-covering, ~2× env diameter)" rather than one constant.
2. **deploy-amax wave** (`*-g98da{1p8,1p6,2p2,3p0}*`, 10 eval-only jobs):
   frozen γ=0.98 fleet actors, ck['amax'] rewritten to {recipe, 3.0}.
   Resolves the clip share of remaining gaps + the user's "adapt amax at
   deployment". Uses `ACTOR_IMPORT_TAG` + `REUSE_ONLY=1` + `DEPLOY_AMAX`
   (hook rewrites top-level amax only; train_args untouched so [args-ok]
   passes).

## How to collect

- TwoRoom/Cube: `sky jobs launch scripts/sky/unigamma/collect_unig.yaml -n
  <name> -y --async --env TAG_GLOB="rlp-tw-unig-g98md20*-20260819"` (etc.).
  Aggregates per (tag, arm, offset); the OLD collector pools same-named arms
  across tags — never quote its aggregate section.
- Reacher: summary.csv per tag —
  `/checkpoints/armin@pantheon.inc/<tag>/results/summary.csv`, rows
  `rs_x0_r0.5_s<seed>_final_e<draw>,held05=..,held10=..`. A ready collector
  yaml pattern is in the session scratchpad (`collect_re_bnd.yaml`); it's a
  5-line volume-mounted cat loop.
- **Reacher convention caveat**: GRID=cross scores HELD-at-end (held10 ≈
  τ0.1 sweep, held05 = ball residency ≈ dm_control 0.05). The latched
  first-hit passes are gated to the latched grids. In-fleet contrasts are
  clean; do not compare against latched tables.

## Decision on landing (the final table)

Unified single config the data supports so far: **γ=0.98, n=50, vnorm=none,
boundary=legacy, amax_train=2.5, band = md20 if the md20 wave confirms**
(else md12 and accept TwoRoom h100 ≈ 88). Deploy-amax per the damax wave.
Quote unified-vs-bespoke deltas per cell, not a silently retuned table.
Remaining writeup work: md20 + damax sections, final conclusion, and the
"γ as horizon budget, stated via d_max" framing note.

## Infra map + traps (all fixed versions committed)

- **WM checkpoints**: GitHub LFS budget exhausted. Volume stage
  (`stage_lfs.yaml`→`/checkpoints/<user>/lfs_stage/`, verify with
  `stage_probe.yaml`) serves TwoRoom AND Cube; the rescue yaml prefers it,
  W&B artifact is fallback. **`cp -rL` always** — plain `cp -r` copies
  SkyPilot file mounts as symlinks that dangle when the staging cluster dies.
- **Env threading**: `TR_GAMMA` (tworoom), `CU_GAMMA` (cube — TR_GAMMA never
  reached cube; the 08-18 "cube-vlog γ=1.0" jobs silently ran 0.98),
  `RS_GAMMA`/`RS_VNORM`/`RS_MD` (reacher, `reacher_gamma.yaml`), `UNIG_MD`,
  `BOUNDARY`/`RS_BOUNDARY`, `DEPLOY_AMAX`. Teacher lives in CACHE_ROOT keyed
  by `CACHE_VERSION` — a γ- or boundary-arm MUST use a matching fresh
  CACHE_VERSION or it silently reuses another arm's teacher.
- **Watchdog**: pgrep now matches the trainer only
  (`train_lip_ac\.py.*<cell>.pt`). The bare-filename pattern killed cube's
  node-serialized queued evals (first cube wave).
- **YAML heredocs**: python heredoc content must sit at the literal-block
  indent (YAML strips it) — column-0 content ends the block; the prior
  session's final yaml revision did not parse for exactly this reason.
- **The Bash tool here is zsh**: unquoted `$VAR` does NOT word-split; launch
  via `bash script.sh` (`launch_unig_fleet.sh` stages:
  smoke|tw|cu|re|fleet|bnd|bnd99|md20|damax).
- Reacher clone must be `GIT_LFS_SKIP_SMUDGE=1` (needs no LFS objects; WM
  from W&B artifact `reacher_base_*`, data from HF).
- Monitors: fleet terminal-state watcher was a persistent `sky jobs queue`
  poll (10 min); stall detection is in-job (watchdog + 4 h trainer timeout).
