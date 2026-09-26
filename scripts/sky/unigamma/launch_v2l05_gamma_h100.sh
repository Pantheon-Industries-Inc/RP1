#!/usr/bin/env bash
# Config-B gamma ladder for the long horizon (2026-09-21, user-approved:
# "open to redoing all h100 (cube + tworoom) across the two bases with higher gamma").
#
# Hypothesis: with n-step 1 the gamma=0.98 cost-to-go saturates at 1/(1-gamma)=50
# steps, so the h100 planner's 25-step plans are ranked at 75 steps-to-go where
# the critic is flat. gamma 0.99 (horizon 100) and 0.995 (horizon 200 = the 2h
# episode budget) test that directly. gamma is applied to BOTH the offline
# teacher and the co-trained critic (TR_GAMMA / CU_GAMMA).
#
# Everything else is config B verbatim (launch_v2l05_pldm_s345.sh), including
# the selection rule (snapshot every 2000 steps on draws 48-51, mean over both
# horizons) so the existing config-B seeds 0-2 are the exact gamma=0.98 control.
# Both horizons are evaluated: the h100 column is the question, the h25 column
# says whether one gamma could serve both. Selection of a per-horizon gamma is
# to be made on eval/ckv5/rh5_h100_s48..51 (selection draws), not on 42-44.
#
#   bash scripts/sky/unigamma/launch_v2l05_gamma_h100.sh
#   GAMMAS="0.995" ENVS="tworoom" bash scripts/sky/unigamma/launch_v2l05_gamma_h100.sh
set -euo pipefail
export PATH="$HOME/.sky/bin:$PATH"
DATE=${DATE:-20260921}
SEEDS=${SEEDS:-"0 1 2"}
GAMMAS=${GAMMAS:-"0.99 0.995"}
ENVS=${ENVS:-"tworoom cube"}
PRIO=${PRIO:-p1}
USERV=armin@pantheon.inc
COMMON=(--env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer --env PANTHEON_USER=$USERV
        --env TRAIN_SEEDS="$SEEDS" --env MAXPAR=4)

gtag(){ echo "g${1#0.}"; }   # 0.99 -> g99, 0.995 -> g995

for G in $GAMMAS; do
  GT=$(gtag "$G")
  for ENVN in $ENVS; do
    if [ "$ENVN" = tworoom ]; then BASES="lejepa pldm"; else BASES="lewm pldm"; fi
    for BASE in $BASES; do
      P=""; [ "$BASE" = pldm ] && P="-p"
      if [ "$ENVN" = tworoom ]; then
        NAME="rlp-tw-v2l05-${GT}${P}"; TAG="rlp-tw-v2l05-${GT}${P}-${DATE}"; CV="tw-v2${P}-n1s2-v1"
        echo "==> tworoom / $BASE / gamma $G  ($NAME)"
        sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$NAME" --priority $PRIO -y --async \
          --env ENVNAME=tworoom --env BASE=$BASE --env GRID=unig --env ONLYCFG=unig_ctrl_a2.5 \
          --env SPLIT=1 --env SMOKE=0 --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
          --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
          --env TR_GAMMA=$G --env TR_NSTEP=1 --env CRIT_DEPTH=2 --env UNIG_LAYERS=2 --env UNIG_MD=20 --env UNIG_K=8 \
          --env ACR_LAM=0.5 --env UNIG_SCHED=tw --env TW_REPLAY=0 \
          --env STEPS=8000 --env BATCH=128 --env TD_STEPS=6000 --env RH=5 \
          --env CKPT_SELECT=1 --env CKPT_EVERY=2000 --env CKPT_VAL_SEEDS="48 49 50 51" \
          --env EVAL_SEEDS="42 43 44" --env UNIG_OFF="25,100" --env EVAL_RAWDIR=volume \
          --env CACHE_VERSION="$CV" --env EXPERIMENT_TAG="$TAG" \
          "${COMMON[@]}" 2>&1 | tail -1
      else
        NAME="rlp-cu-v2l05-${GT}${P}"; TAG="rlp-cu-v2l05-${GT}${P}-${DATE}"; CV="cu-v2${P}-n1s2-v1"
        echo "==> cube / $BASE / gamma $G  ($NAME)"
        sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$NAME" --priority $PRIO -y --async \
          --env ENVNAME=cube --env BASE=$BASE --env GRID=unig --env ONLYCFG=unig_ctrl_a2.5 \
          --env SPLIT=0 --env SMOKE=0 --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
          --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
          --env CU_DYNA=0 --env CU_DYNA_WM="" --env CU_EXPAND=1.0 --env CU_GAMMA=$G --env TR_NSTEP=1 \
          --env CRIT_DEPTH=2 --env UNIG_LAYERS=2 --env UNIG_MD=20 --env UNIG_K=8 --env ACR_LAM=0.5 \
          --env STEPS=6000 --env BATCH=256 --env TD_STEPS=12000 --env RH=5 \
          --env CKPT_SELECT=1 --env CKPT_EVERY=2000 --env CKPT_VAL_SEEDS="48 49 50 51" \
          --env EVAL_SEEDS="42 43 44" --env UNIG_OFF="25,100" --env EVAL_RAWDIR=volume \
          --env CACHE_VERSION="$CV" --env EXPERIMENT_TAG="$TAG" \
          "${COMMON[@]}" 2>&1 | tail -1
      fi
    done
  done
done
echo "collect: python3 scripts/wandb_collect_rlp.py 'v2l05-g99'"
