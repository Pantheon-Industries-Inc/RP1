#!/usr/bin/env bash
# PushT critic arms on the config-B recipe (pusht-v2l05-s*-20260902, unchanged
# otherwise) -- see docs/campaigns/2026-09-03/PUSHT_DIAG.md.
#
# Only the arms reachable under the shipped code remain. The 2026-09-08/10
# arms for agent augmentation, tolerance relabeling, distance-dependent
# expectile, near-goal loss weighting, actor near-goal problems, goal tiling,
# imagination-MC, randomized smoothing and pessimistic expansion all closed
# negative and their knobs were deleted with the scaffolding; PUSHT_DIAG.md
# keeps the numbers.
#
#   scripts/sky/launch_pusht_critic_arms.sh BASE 0 1 2       # config B as shipped
#   scripts/sky/launch_pusht_critic_arms.sh NEAR 0 1 2       # + near-goal oversampling (+4)
#   scripts/sky/launch_pusht_critic_arms.sh W1NEAR_FRZ 0 1 2 # best recipe: median 70.7
#   DRY=1 ... prints the command
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.sky/bin:$PATH"

ARM=${1:?arm BASE|LATENT|E01|NEAR|NEAR5|NEARM5|NEARN5|N5|W1NEAR|W2NEAR|W1NEAR_FRZ|W1N5NEAR_FRZ|N5NEAR_FRZ|NEAR_NOEXP|N5NEAR_NOEXP|W1NEAR_FRZ_D3}; shift
SEEDS=${*:-0}
DATE=${DATE:-20260911}
WF=4   # critic window frames (config B: 4); W1NEAR / W2NEAR retest narrower windows with near-goal training
VEXP=0.03; EXTRA=""; NEAR=0; VNSTEP=1; NMAX=${NEAR_MAX:-3}; EXPAND=1.0
case $ARM in
  BASE)    ;;                                      # config B exactly, no critic change
  # The control that separates "the critic is a bad objective" from "the refiner is a bad
  # optimizer": the deployed critic IS the single-frame latent distance, i.e. exactly the
  # objective CEM scores 79.3 with. Reach ~79 and RLP matches CEM at ~1000x less compute;
  # stay ~70 and the deficit is optimizer-generic and no critic will fix it.
  LATENT)  WF=1; EXTRA="value.learner=l2 planner.freeze_critic_frac=0"; CONDS=${CONDS:-"rlp cem_latent cem_value"};;
  E01)     VEXP=0.1; EXTRA="planner.expectile=0.1 planner.expectile_final=0.1";;
  NEAR)    NEAR=${NEAR_FRAC:-0.3};;                # near-goal oversampling 0.3 of goals at 1..3 steps
  NEAR5)   NEAR=0.5;;                              # dose: half of the goals near
  NEARM5)  NEAR=0.3; NMAX=5;;                      # wider near band 1..5 steps
  NEARN5)  NEAR=0.3; VNSTEP=5;;                    # near goals get exact MC targets (teacher n-step 5)
  N5)      VNSTEP=5;;                              # exact MC targets for teacher goals within 5 steps
  W1NEAR)  WF=1; NEAR=0.3;;                        # single-frame critic + near-goal oversampling (no window handicap)
  W2NEAR)  WF=2; NEAR=0.3;;                        # two-frame (position + velocity) critic + near-goal oversampling
  # W1NEAR's offline teacher reached 78.0 as a CEM objective (= latent L2 78.9) while its co-trained
  # critic fell to 68.7 and the planner to 61.3 -> train the refiner against that teacher, frozen.
  # This is the campaign's best recipe: 72.0 / 70.7 / 67.3, median 70.7 vs config-B base 64.7.
  W1NEAR_FRZ)    WF=1; NEAR=0.3; EXTRA="planner.freeze_critic_frac=0";;
  W1N5NEAR_FRZ)  WF=1; NEAR=0.3; VNSTEP=5; EXTRA="planner.freeze_critic_frac=0";;
  N5NEAR_FRZ)    VNSTEP=5; NEAR=0.3; EXTRA="planner.freeze_critic_frac=0";;
  W1NEAR_FRZ_D3) WF=1; NEAR=0.3; EXTRA="planner.freeze_critic_frac=0 value.depth=3";;
  NEAR_NOEXP)    NEAR=0.3; EXPAND=0;;              # co-train without value expansion on imagined rollouts
  N5NEAR_NOEXP)  VNSTEP=5; NEAR=0.3; EXPAND=0;;
  *) echo "unknown arm $ARM" >&2; exit 1;;
esac
BASE_OVR="value.window_frames=$WF value.window_lag=5 planner.ac_weight=0.5 planner.ckpt_every=2000"
# DATA=N: data-volume ladder. Caps BOTH learners at episodes [0, N) of the shared
# 16k cache -- no re-encode, and the held-out eval range (16000+) is untouched.
SUF=""
if [ -n "${DATA:-}" ]; then
  BASE_OVR="$BASE_OVR value.max_episodes=$DATA planner.max_episodes=$DATA"
  SUF="-d${DATA}"
fi
for S in $SEEDS; do
  TAG="pusht-${ARM,,}${SUF}-s${S}-${DATE}"
  cmd=(sky jobs launch scripts/sky/counterstrike_pusht.yaml
    -n "rlp-$TAG" --priority p1 -y --async
    --env EXPERIMENT_TAG="$TAG"
    --env CACHE_TAG=counterstrike --env WAIT_CACHE_MIN=0 --env EXPAND="$EXPAND"
    --env SEED="$S" --env TD_MODE=cube --env ITERS=8
    --env VALUE_GAMMA=0.98 --env VALUE_NSTEP="$VNSTEP" --env VALUE_EXPECTILE="$VEXP"
    --env MAX_DELTA=20 --env MEAN_WEIGHT=0.1 --env AMAX="${AMAX_OVERRIDE:-2.5}"
    --env CKPT_SELECT=1 --env CKPT_VAL_SEEDS="50 51"
    --env TRAIN_OVERRIDES="$BASE_OVR $EXTRA"
    --env NEAR_FRAC="$NEAR" --env NEAR_MAX="$NMAX"
    --env EVAL_SEEDS="42 43 44" --env EVAL_CONDS="${CONDS:-rlp cem_value cem_tdvalue}" --env SMOKE=0
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer
    --env PANTHEON_USER=armin@pantheon.inc)
  if [ "${DRY:-0}" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}" 2>&1 | grep -c "Submitted\|Check logs" || true; fi
  echo "-> $TAG (arm $ARM: wf=$WF nstep=$VNSTEP near=$NEAR/$NMAX expand=$EXPAND extra='$EXTRA')"
done
