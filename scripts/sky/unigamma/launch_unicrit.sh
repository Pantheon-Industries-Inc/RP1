#!/usr/bin/env bash
# UNIFIED CRITIC + UNIFIED ACTOR BAND on top of config B, all seven cells
# (TwoRoom/Cube/Reacher x LeWM/PLDM, PushT/LeWM). Actor band rule (2026-09-20):
# max_delta = longest deployment horizon in fs5 blocks + 1 (tw/cube 21,
# reacher/pusht 6), offsets drawn as a MIXTURE over the deployment-horizon
# bands (tw/cube 6,21: half the goals within 6 blocks, half within 21; a
# single-horizon env is just uniform 1..6) so one actor serves h25 and h100,
# p_cross 0.1 everywhere.
# Rest of the actor is config B (amax 2.5, ac 0.5, K 8, ES on 48-51,
# per-env budgets/anchors/replay).
#
# The critic recipe (= PushT's W1NEAR_FRZ teacher, made the rule everywhere):
#   * offline TD teacher only, deployed verbatim: freeze_critic_frac 0, so the
#     co-critic never runs and every planner.* TD knob is inert (see
#     BEST_CONFIGS_20260914.md section 7a)
#   * teacher steps = ACTOR steps (ratio 1.0; PushT's ladder 3k -7.4 / 6k best /
#     12k / 24k -7.4 is the only measured curve): tw 8000, cube 6000,
#     reacher 6000, pusht 6000. Batch 1024 everywhere.
#   * n-step 1, gamma 0.98, expectile 0.03 (tworoom was 0.1), depth 2,
#     p_cross 0.3, max_delta = full episode, near-goal oversampling 0.3 of
#     in-episode goals at 1..3 primitive steps (fs1 cache everywhere),
#     one teacher per training seed (cube/reacher shared seed 0 before).
#   * window frames stay the environment property (1 / 1 / 2 / 1).
#   * selection draws 48-51 in every cell (PushT was 50/51), report 42/43/44.
#   * PushT base = the OFFICIAL quentinll/lewm-pusht release (user decision
#     2026-09-20), not the in-house epoch-20 WM.
#
# Prerequisite: GIT_TOKEN secret refreshed (tworoom/reacher yamls clone
# Value_Metric_LeWM@eval-sweep; every such job failed on it 2026-09-09 and
# 2026-09-20). Run `sky workspace use armin` first.
#
#   bash scripts/sky/unigamma/launch_unicrit.sh            # all 7 cells, seeds 0-2
#   CELLS="cube-pldm pusht" SEEDS="0 1 2 3 4 5" bash ...   # subset
#   DRY=1 bash ...                                          # print only
#   CRITIC=plaintd bash ...                                 # plain-TD control (mlp head, expectile 0.5)
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="$HOME/.sky/bin:$PATH"
DATE=${DATE:-20260920}
SEEDS=${SEEDS:-"0 1 2"}
PRIO=${PRIO:-p1}
# CRITIC=mrn (default): quasimetric MRN head + expectile 0.03 (config B / PushT's teacher).
# CRITIC=plaintd: the "normal TD" control -- plain pairwise-MLP V(z,g), expectile 0.5
# (symmetric Huber), same sampler / gamma / n-step / budget; tags get -plaintd.
CRITIC=${CRITIC:-mrn}
case $CRITIC in
  mrn)     TDHEAD=quasimetric; TDEXP=0.03; CSUF="";;
  plaintd) TDHEAD=mlp;         TDEXP=0.5;  CSUF="-plaintd";;
  *) echo "CRITIC must be mrn|plaintd" >&2; exit 1;;
esac
USERV=armin@pantheon.inc
CELLS=${CELLS:-"tworoom-lejepa tworoom-pldm cube-lewm cube-pldm reacher-lejepa reacher-pldm pusht"}
COMMON=(--env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer --env PANTHEON_USER=$USERV --env MAXPAR=4)
run() { if [ "${DRY:-0}" = 1 ]; then printf '%q ' "$@"; echo; else "$@" 2>&1 | grep -Ev "^warning: the SkyPilot|curl -fsSL" | tail -1; fi; }

for CELL in $CELLS; do
  ENV=${CELL%%-*}; BASE=${CELL#*-}; B1=$(echo $BASE | cut -c1)
  case $ENV in
  tworoom)
    NAME="rlp-tw-unicrit${CSUF}-${B1}-${DATE}"
    run sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$NAME" --priority $PRIO -y --async \
      --env ENVNAME=tworoom --env BASE=$BASE --env GRID=unig --env ONLYCFG=unig_ctrl_a2.5 \
      --env SPLIT=1 --env SMOKE=0 --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
      --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
      --env TR_GAMMA=0.98 --env TR_NSTEP=1 --env CRIT_DEPTH=2 --env UNIG_LAYERS=2 --env UNIG_MD=21 --env UNIG_K=8 \
      --env ACR_LAM=0.5 --env TW_REPLAY=0 --env ACTOR_BANDS="6,21" --env ACTOR_PC=0.1 \
      --env UNIG_SCHED=frz --env TD_STEPS=8000 --env TW_TD_EXPECTILE=$TDEXP --env TD_HEAD=$TDHEAD --env UNIG_PC=0.3 \
      --env NEAR_FRAC=0.3 --env NEAR_MAX=3 --env TD_PER_SEED=1 \
      --env STEPS=8000 --env BATCH=128 --env RH=5 \
      --env CKPT_SELECT=1 --env CKPT_EVERY=2000 --env CKPT_VAL_SEEDS="48 49 50 51" \
      --env EVAL_SEEDS="42 43 44" --env UNIG_OFF="25,100" --env EVAL_RAWDIR=volume \
      --env TRAIN_SEEDS="$SEEDS" --env CACHE_VERSION="tw-unicrit${CSUF}-${B1}-v1" --env EXPERIMENT_TAG="$NAME" "${COMMON[@]}"
    ;;
  cube)
    NAME="rlp-cu-unicrit${CSUF}-${B1}-${DATE}"
    run sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$NAME" --priority $PRIO -y --async \
      --env ENVNAME=cube --env BASE=$BASE --env GRID=unig --env ONLYCFG=unig_ctrl_a2.5 \
      --env SPLIT=0 --env SMOKE=0 --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
      --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
      --env CU_DYNA=0 --env CU_DYNA_WM="" --env CU_EXPAND=1.0 --env CU_REPLAY=0.5 --env CU_GAMMA=0.98 --env TR_NSTEP=1 \
      --env CRIT_DEPTH=2 --env UNIG_LAYERS=2 --env UNIG_MD=21 --env UNIG_K=8 --env ACR_LAM=0.5 --env ACTOR_BANDS="6,21" --env ACTOR_PC=0.1 \
      --env UNIG_SCHED=frz --env TD_STEPS=6000 --env CU_TD_EXPECTILE=$TDEXP --env TD_HEAD=$TDHEAD --env UNIG_PC=0.3 \
      --env NEAR_FRAC=0.3 --env NEAR_MAX=3 --env TD_PER_SEED=1 \
      --env STEPS=6000 --env BATCH=256 --env RH=5 \
      --env CKPT_SELECT=1 --env CKPT_EVERY=2000 --env CKPT_VAL_SEEDS="48 49 50 51" \
      --env EVAL_SEEDS="42 43 44" --env UNIG_OFF="25,100" --env EVAL_RAWDIR=volume \
      --env TRAIN_SEEDS="$SEEDS" --env CACHE_VERSION="cu-unicrit${CSUF}-${B1}-v1" --env EXPERIMENT_TAG="$NAME" "${COMMON[@]}"
    ;;
  reacher)
    NAME="rlp-re-unicrit${CSUF}-${B1}-${DATE}"
    run sky jobs launch scripts/sky/unigamma/reacher_gamma.yaml -n "$NAME" --priority $PRIO -y --async \
      --env BASE=$BASE --env GRID=cross --env AMFIX=2.5 --env CROSS_EXPANDS="0" --env CROSS_REPLAYS="0.5" \
      --env SMOKE=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 --env ACTOR_IMPORT_TAG="" \
      --env RS_GAMMA=0.98 --env RS_NSTEP=1 --env CRIT_DEPTH=2 --env RS_LAYERS=2 --env RS_MD=6 --env RS_K=8 --env RS_ACTOR_BANDS="6" --env RS_ACTOR_PC=0.1 \
      --env ACR_LAM=0.5 --env VALUE_FRAMES=2 --env VALUE_EXPECTILE=$TDEXP --env RS_TD_HEAD=$TDHEAD \
      --env RS_FREEZE=0 --env RS_TD_STEPS=6000 --env RS_TD_PC=0.3 --env RS_TD_NEAR_FRAC=0.3 --env RS_TD_NEAR_MAX=3 --env RS_TD_PER_SEED=1 \
      --env STEPS=6000 --env BATCH=256 --env CKPT_EVERY=2000 --env RS_CKPT_SELECT=1 --env RS_CKPT_VAL_SEEDS="48 49 50 51" \
      --env RS_LATCHED=1 --env FINAL_ONLY=1 --env HELD_AT_END=0 --env REPORT_DRAWS="42 43 44" \
      --env SUCCESS_THRESHOLDS="0.1 0.05" --env PLANNER_RH=5 \
      --env TRAIN_SEEDS="$SEEDS" --env EXPERIMENT_TAG="$NAME" "${COMMON[@]}"
    ;;
  pusht)
    # PushT already IS this critic recipe (W1NEAR_FRZ + value.steps=6000 = actor
    # steps); the actor side is the 78.7 es1k recipe. Re-run only for a same-date tag.
    for S in $SEEDS; do
      # official quentinll/lewm-pusht base (CACHE_TAG=counterstrike, no WM_DIR);
      # selection draws 48-51 like every other cell (E41: extra draws are free).
      DRY=${DRY:-0} DATE=$DATE REPLAY=0 MAX_DELTA_OVERRIDE=6 CKPT_VAL_SEEDS="48 49 50 51" \
        EXTRA_OVR="planner.p_cross=0.1 planner.band_mix=6 value.steps=6000 planner.ckpt_every=1000 value.head=$TDHEAD value.expectile=$TDEXP" TAGSUF=-unicrit${CSUF} \
        scripts/sky/launch_pusht_critic_arms.sh W1NEAR_FRZ $S
    done
    ;;
  *) echo "unknown cell $CELL" >&2; exit 1;;
  esac
  echo "-> $CELL"
done
echo "Collect: python3 scripts/wandb_collect_rlp.py 'unicrit${CSUF}' --reacher"
