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
WF=4   # critic window frames (config B: 4); W1NEAR / W2NEAR retest narrower windows with near-goal training
VEXP=0.03; EXTRA=""; TOL=0; NEAR=0; VNSTEP=1; EXPN=""; NW=0; NMAX=${NEAR_MAX:-3}; ANEAR=0; TILE=0; EXPAND=1.0; IMAG=0
SMOOTH=0; SMOOTH_STD=0.1; EXPAND_MODE=""; TD_WEIGHT=""
case $ARM in
  # --- refiner anti-exploitation (2026-09-09), on the frozen single-frame near-goal teacher (W1NEAR_FRZ 72.0 / 70.7)
  W1NEAR_FRZ_SM)   WF=1; NEAR=0.3; EXTRA="planner.freeze_critic_frac=0"; SMOOTH=4; SMOOTH_STD=0.1;;
  W1NEAR_FRZ_SM3)  WF=1; NEAR=0.3; EXTRA="planner.freeze_critic_frac=0"; SMOOTH=4; SMOOTH_STD=0.3;;
  # --- pessimistic imagination for the co-critic (w1 near teacher, co-training kept)
  W1NEAR_PESS)     WF=1; NEAR=0.3; EXPAND_MODE=pessimistic;;
  W1NEAR_PESSONLY) WF=1; NEAR=0.3; EXPAND_MODE=pessimistic; TD_WEIGHT=0;;
  # --- imagination-MC (2026-09-09): train the co-critic on WM-imagined latents along the data's actions
  #     with exact labels -- the critic is read on imagined latents, where E5 showed it optimistic
  NEAR_IMAG)          NEAR=0.3; IMAG=1.0;;
  NEAR_IMAG_NOEXP)    NEAR=0.3; IMAG=1.0; EXPAND=0;;
  N5NEAR_IMAG_NOEXP)  VNSTEP=5; NEAR=0.3; IMAG=1.0; EXPAND=0;;
  # --- teacher-vs-co-critic split (2026-09-09): every arm that sharpened the offline teacher as a CEM
  #     objective (N5 76.7, NEARTILE 76.0, NW 73.3 vs base 67.3) LOST on the planner while the co-critic
  #     got worse -> (a) freeze the critic at the sharpened teacher (no co-training), (b) co-train without
  #     value expansion on imagined rollouts
  N5NEARTILE_FRZ) VNSTEP=5; NEAR=0.3; TILE=1; EXTRA="planner.freeze_critic_frac=0";;
  N5NEAR_FRZ)     VNSTEP=5; NEAR=0.3; EXTRA="planner.freeze_critic_frac=0";;
  NEAR_NOEXP)     NEAR=0.3; EXPAND=0;;
  N5NEAR_NOEXP)   VNSTEP=5; NEAR=0.3; EXPAND=0;;
  TILE)    TILE=1;;                                # deploy-matched tiled goal frames for the window critics
  NEARTILE) NEAR=0.3; TILE=1;;
  W1NEAR)  WF=1; NEAR=0.3;;                        # single-frame critic + near-goal oversampling (no window handicap)
  W2NEAR)  WF=2; NEAR=0.3;;                        # two-frame (position + velocity) critic + near-goal oversampling
  # W1NEAR's offline teacher reached 78.0 as a CEM objective (= latent L2 78.9) while its co-trained
  # critic fell to 68.7 and the planner to 61.3 -> train the refiner against that teacher, frozen
  W1NEAR_FRZ)   WF=1; NEAR=0.3; EXTRA="planner.freeze_critic_frac=0";;
  W1N5NEAR_FRZ) WF=1; NEAR=0.3; VNSTEP=5; EXTRA="planner.freeze_critic_frac=0";;
  E01)     VEXP=0.1; EXTRA="planner.expectile=0.1 planner.expectile_final=0.1";;
  NEAR)    NEAR=${NEAR_FRAC:-0.3};;
  NEAR5)   NEAR=0.5;;                              # dose: half of the goals near
  NEARM5)  NEAR=0.3; NMAX=5;;                      # wider near band 1..5 steps
  NEARN5)  NEAR=0.3; VNSTEP=5;;                    # near goals get exact MC targets (teacher n-step 5)
  NEARA)   NEAR=0.3; ANEAR=${ACTOR_NEAR_FRAC:-0.3};;  # + actor-side near-goal problems (1..2 blocks)
  TOL)     TOL=1;;
  NEARTOL) NEAR=${NEAR_FRAC:-0.3}; TOL=1;;
  EXPN)    EXPN=${EXPECTILE_NEAR:-0.5};;          # neutral expectile inside the last NEAR_STEPS steps
  N5)      VNSTEP=5;;                              # exact MC targets for teacher goals within 5 steps
  NW)      NW=${NEAR_WEIGHT:-1.0};;                # loss weight (1+d)^-1
  COMBO)   TOL=1; NEAR=${NEAR_FRAC:-0.3}; EXPN=${EXPECTILE_NEAR:-0.5}; VNSTEP=5;;
  *) echo "unknown arm $ARM" >&2; exit 1;;
esac
BASE_OVR="value.window_frames=$WF value.window_lag=5 planner.ac_weight=0.5 planner.ckpt_every=2000"
for S in $SEEDS; do
  TAG="pusht-${ARM,,}-s${S}-${DATE}"
  cmd=(sky jobs launch scripts/sky/counterstrike_pusht.yaml
    -n "rlp-$TAG" --priority p1 -y --async
    --env EXPERIMENT_TAG="$TAG"
    --env CACHE_TAG=counterstrike --env WAIT_CACHE_MIN=0 --env EXPAND="$EXPAND"
    --env SEED="$S" --env TD_MODE=cube --env ITERS=8
    --env VALUE_GAMMA=0.98 --env VALUE_NSTEP="$VNSTEP" --env VALUE_EXPECTILE="$VEXP"
    --env MAX_DELTA=20 --env MEAN_WEIGHT=0.1 --env AMAX="${AMAX_OVERRIDE:-2.5}"
    --env CKPT_SELECT=1 --env CKPT_VAL_SEEDS="50 51"
    --env TRAIN_OVERRIDES="$BASE_OVR $EXTRA"
    --env TOL_RELABEL="$TOL" --env NEAR_FRAC="$NEAR" --env NEAR_MAX="$NMAX"
    --env EXPECTILE_NEAR="$EXPN" --env NEAR_STEPS="${NEAR_STEPS:-3}" --env NEAR_WEIGHT="$NW"
    --env ACTOR_NEAR_FRAC="$ANEAR" --env ACTOR_NEAR_MAX="${ACTOR_NEAR_MAX:-2}" --env GOAL_TILE="$TILE" --env IMAG_MC="$IMAG"
    --env SMOOTH="$SMOOTH" --env SMOOTH_STD="$SMOOTH_STD" --env EXPAND_MODE="$EXPAND_MODE" --env TD_WEIGHT="$TD_WEIGHT"
    --env EVAL_SEEDS="42 43 44" --env EVAL_CONDS="rlp cem_value cem_tdvalue" --env SMOKE=0
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer
    --env PANTHEON_USER=armin@pantheon.inc)
  if [ "${DRY:-0}" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}" 2>&1 | grep -c "Submitted\|Check logs" || true; fi
  echo "-> $TAG (arm $ARM: wf=$WF nstep=$VNSTEP near=$NEAR/$NMAX smooth=$SMOOTH/$SMOOTH_STD expand=$EXPAND/${EXPAND_MODE:-opt} td_w=${TD_WEIGHT:-1} extra='$EXTRA')"
done
