#!/usr/bin/env bash
# PushT critic arms on the config-B recipe (pusht-v2l05-s*-20260902, unchanged
# otherwise) -- see docs/campaigns/2026-09-03/PUSHT_DIAG.md (E12).
#
#   scripts/sky/launch_pusht_critic_arms.sh E01 0 1 2    # expectile 0.1 (teacher + co-critic, flat)
#   scripts/sky/launch_pusht_critic_arms.sh NEAR 0 1 2   # near-goal oversampling 0.3 of goals at 1..3 steps
#   scripts/sky/launch_pusht_critic_arms.sh TOL 0        # success-tolerance relabeling
#   scripts/sky/launch_pusht_critic_arms.sh NEARTOL 0    # both
#   DRY=1 ... prints the command
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.sky/bin:$PATH"

ARM=${1:?arm E01|NEAR|TOL|NEARTOL}; shift
SEEDS=${*:-0}
DATE=${DATE:-20260909}
BASE_OVR="value.window_frames=4 value.window_lag=5 planner.ac_weight=0.5 planner.ckpt_every=2000"
VEXP=0.03; EXTRA=""; TOL=0; NEAR=0; VNSTEP=1; EXPN=""; NW=0
case $ARM in
  E01)     VEXP=0.1; EXTRA="planner.expectile=0.1 planner.expectile_final=0.1";;
  NEAR)    NEAR=${NEAR_FRAC:-0.3};;
  TOL)     TOL=1;;
  NEARTOL) NEAR=${NEAR_FRAC:-0.3}; TOL=1;;
  EXPN)    EXPN=${EXPECTILE_NEAR:-0.5};;          # neutral expectile inside the last NEAR_STEPS steps
  N5)      VNSTEP=5;;                              # exact MC targets for teacher goals within 5 steps
  NW)      NW=${NEAR_WEIGHT:-1.0};;                # loss weight (1+d)^-1
  COMBO)   TOL=1; NEAR=${NEAR_FRAC:-0.3}; EXPN=${EXPECTILE_NEAR:-0.5}; VNSTEP=5;;
  *) echo "unknown arm $ARM" >&2; exit 1;;
esac
for S in $SEEDS; do
  TAG="pusht-${ARM,,}-s${S}-${DATE}"
  cmd=(sky jobs launch scripts/sky/counterstrike_pusht.yaml
    -n "rlp-$TAG" --priority p1 -y --async
    --env EXPERIMENT_TAG="$TAG"
    --env CACHE_TAG=counterstrike --env WAIT_CACHE_MIN=0
    --env SEED="$S" --env TD_MODE=cube --env ITERS=8
    --env VALUE_GAMMA=0.98 --env VALUE_NSTEP="$VNSTEP" --env VALUE_EXPECTILE="$VEXP"
    --env MAX_DELTA=20 --env MEAN_WEIGHT=0.1 --env AMAX=2.5
    --env CKPT_SELECT=1 --env CKPT_VAL_SEEDS="50 51"
    --env TRAIN_OVERRIDES="$BASE_OVR $EXTRA"
    --env TOL_RELABEL="$TOL" --env NEAR_FRAC="$NEAR" --env NEAR_MAX="${NEAR_MAX:-3}"
    --env EXPECTILE_NEAR="$EXPN" --env NEAR_STEPS="${NEAR_STEPS:-3}" --env NEAR_WEIGHT="$NW"
    --env EVAL_SEEDS="42 43 44" --env EVAL_CONDS="rlp cem_value cem_tdvalue" --env SMOKE=0
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer
    --env PANTHEON_USER=armin@pantheon.inc)
  if [ "${DRY:-0}" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}" 2>&1 | grep -c "Submitted\|Check logs" || true; fi
  echo "-> $TAG (arm $ARM: value.expectile=$VEXP nstep=$VNSTEP tol=$TOL near=$NEAR expectile_near=${EXPN:-off} near_weight=$NW)"
done
