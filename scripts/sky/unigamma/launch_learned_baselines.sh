#!/usr/bin/env bash
# L2O-MPC and DMPO under the uniJ protocol: same world model, same held-out split, same report draws, and the SAME
# CRITIC as the cell's locked RLP pair -- unified teacher recipe (gamma per stack, n-step 1, expectile 0.03, depth 2)
# trained for exactly the locked teacher BUDGET. The TD trainer has no step-dependent schedule, so training N steps
# reproduces the RLP row's step-N teacher snapshot (same seed, same latent cache).
# Both baselines TRAIN an inner-loop policy, so they run TRAIN_SEEDS seeds (default 3, vs n=6 for the eval-only arms).
#   scripts/sky/unigamma/launch_learned_baselines.sh l2o|dmpo [reacher|tworoom|cube|all]
set -uo pipefail
cd "$(dirname "$0")/../../.."; export PATH="$HOME/.sky/bin:$PATH"
eval "$(python3 scripts/sky/unigamma/recipes/recipe_env.py scripts/sky/unigamma/recipes/uniJ.yaml)"
METHOD=${1:?l2o|dmpo}; WHICH=${2:-all}; DATE=${DATE:-20260924}; USERV=armin@pantheon.inc; PRIO=${PRIO:-p1}
SEEDS=${TRAIN_SEEDS:-"0 1 2"}
# PAPER-STANDARD budgets (user decision 2026-09-24):
#  DMPO (Sacks et al., ICRA 2024, Sec. V-A): "up to 1000 iterations of PPO" -> OBJECTIVE=ppo, PPO_ITERATIONS=1000
#    (our earlier in-house arm used the pathwise objective at 2000 steps; the paper's headline is the PPO loop).
#  L2O (Sacks & Boots, ICRA 2022, Sec. IV): DAgger structure is matched exactly (20 rounds, beta_k=0.8^k, bootstrap +
#    128 rollouts/round, expert N >> learner M). Its literal optimizer budget -- 1000 epochs per round over the
#    aggregated set at batch 8 -- is ~10^6 gradient steps/seed; in their setup a step is a tiny MLP update, in ours it
#    is a world-model unroll (64 particles x 4 iterations + periodic 512-particle expert query), so matching the step
#    COUNT would cost days per seed without matching the compute the paper spends. We train to convergence instead:
#    L2O_POLICY_STEPS (default 50000 = 25x the original port, 4x the 12k actor budget) and report the plateau.
DMPO_PPO_ITERS=${DMPO_PPO_ITERS:-1000}
# Budget + SELECTION (2026-09-24). RLP deploys a snapshot CHOSEN on validation draws 48-51; a fixed
# final iterate is not the same treatment, and both baselines' training objectives say that iterate is
# arbitrary -- L2O's is flat from ~step 1,000 on Reacher/TwoRoom and rising (worse) on Cube, DMPO's PPO
# return is a trendless walk in all three envs (the paper itself says "up to 1000 iterations").
# So: snapshot periodically and select, exactly as the RLP rows do -- one shared step per cell by the
# mean validation score over the training seeds, ties to the earlier step.
# L2O's ladder is 3k..18k, the same six-point ladder the RLP teacher uses.
L2O_POLICY_STEPS=${L2O_POLICY_STEPS:-18000}; L2O_SAVE_EVERY=${L2O_SAVE_EVERY:-3000}
DMPO_SAVE_EVERY=${DMPO_SAVE_EVERY:-200}
SELECT=${SELECT:-1}
YAML=scripts/sky/${METHOD}_campaign.yaml
CELLS=""
case $WHICH in
  # cell:base:stack:LOCKED TEACHER BUDGET (from RESULTS_uniJ_shared_stop.md)
  reacher) CELLS="reacher:lejepa:25:18000 reacher:pldm:25:18000" ;;
  tworoom) CELLS="tworoom:lejepa:25:15000 tworoom:lejepa:100:6000 tworoom:pldm:25:3000 tworoom:pldm:100:3000" ;;
  cube)    CELLS="cube:lewm:25:3000 cube:lewm:100:9000 cube:pldm:25:15000 cube:pldm:100:12000" ;;
  all)     CELLS="reacher:lejepa:25:18000 reacher:pldm:25:18000 tworoom:lejepa:25:15000 tworoom:lejepa:100:6000 tworoom:pldm:25:3000 tworoom:pldm:100:3000 cube:lewm:25:3000 cube:lewm:100:9000 cube:pldm:25:15000 cube:pldm:100:12000" ;;
  *) echo "unknown $WHICH" >&2; exit 2 ;;
esac
for c in $CELLS; do
  IFS=: read -r ENVN BASE HZ TSTEP <<< "$c"
  # ONLY_BASE / ONLY_HZ: relaunch a single cell without re-touching its siblings
  [ -n "${ONLY_BASE:-}" ] && [ "$BASE" != "$ONLY_BASE" ] && continue
  [ -n "${ONLY_HZ:-}" ] && [ "$HZ" != "$ONLY_HZ" ] && continue
  P=""; [ "$BASE" = pldm ] && P="-p"
  if [ "$HZ" = 25 ]; then GAMMA=$UNI_GAMMA_H25; else GAMMA=$UNI_GAMMA_H100; fi
  VEX="value.gamma=$GAMMA value.expectile=$UNI_TD_EXPECTILE value.n_step=$UNI_NSTEP value.depth=$UNI_CRIT_DEPTH"
  VF=1; [ "$ENVN" = reacher ] && { VF=$UNI_WINDOW_REACHER; VEX="$VEX +value.window_frames=$VF +value.window_lag=5"; }
  PFX=tw; [ "$ENVN" = cube ] && PFX=cu; [ "$ENVN" = reacher ] && PFX=re
  G=${GPUS:-3}   # one GPU per training seed (both campaign yamls train the seeds concurrently)
  # goal band: uniJ md=20 everywhere. TwoRoom fs5 episodes are only 20 blocks (100 primitive steps), which used to
  # trip WindowSampler's "episode longer than max_delta + 4" guard -- an over-strict check, since goals and reference
  # blocks are clamped to the episode end. Relaxed 2026-09-24 in src/rlp/train/windows.py, exactly as the RLP actor
  # trainer had already relaxed it (scripts/sky/overlays/train_lip_ac.py), so the baselines see the band RLP saw.
  MD=${MD_OVERRIDE:-$UNI_MD}
  # Reacher MuJoCo evals abort (core dump) when many run at once; 2-wide carried Reacher/PLDM but still
  # core-dumped Reacher/LeJEPA after the first rollout, so that cell needs REACHER_EVAL_PAR=1.
  EVAL_PAR_ARG=""; [ "$ENVN" = reacher ] && EVAL_PAR_ARG="--env EVAL_PAR=${REACHER_EVAL_PAR:-2}"
  NAME=rlp-${PFX}-${METHOD}-h${HZ}${P}; TAG=${NAME}-${DATE}
  # A relaunch under a NEW date needs a fresh EXPERIMENT_TAG (the yamls cache trained policies and eval results per
  # tag, so reusing the tag would silently return the old run's numbers) but should NOT re-encode the latent caches,
  # which cost more than the training. CACHE_DATE points $CD at the previous run's own cache dir -- per cell, so no
  # two jobs ever write the same cache files. Unset, it equals $DATE and nothing changes.
  CTAG=${NAME}-${CACHE_DATE:-$DATE}
  echo "==> $METHOD $ENVN/$BASE h$HZ gamma $GAMMA teacher ${TSTEP} steps seeds [$SEEDS] ($NAME, ${G} GPU, md $MD, caches $CTAG)"
  sky jobs launch $YAML -n "$NAME" --priority $PRIO --gpus H200:$G -y --async \
    --env ENVNAME=$ENVN --env BASE=$BASE --env VALUE_FRAMES=$VF --env AMAX=$UNI_AMAX \
    --env VALUE_EXTRA_OVERRIDE="$VEX" --env VALUE_STEPS_OVERRIDE=$TSTEP --env HORIZONS_OVERRIDE=$HZ \
    --env MAX_DELTA_OVERRIDE=$MD --env TRAIN_SEEDS="$SEEDS" --env EVAL_SEEDS="$UNI_REPORT_DRAWS" \
    $([ "$METHOD" = l2o ] && echo "--env L2O_STEPS=$L2O_POLICY_STEPS --env SAVE_EVERY=$L2O_SAVE_EVERY" \
        || echo "--env OBJECTIVE=ppo --env PPO_ITERATIONS=$DMPO_PPO_ITERS --env SAVE_EVERY=$DMPO_SAVE_EVERY") \
    --env SELECT=$SELECT --env VAL_SEEDS="$UNI_VAL_DRAWS" \
    --env SMOKE=0 --env EXPERIMENT_TAG="$TAG" --env CACHE_TAG="$CTAG" $EVAL_PAR_ARG \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer --env PANTHEON_USER=$USERV 2>&1 | { grep -E "Submitted|sky jobs logs [0-9]+|rror" || true; } | tail -1
done
