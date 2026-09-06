#!/usr/bin/env bash
# Dyna v2 driver: per base, ITERS x (collect+anchored-finetune -> config-B
# train+eval with per-task arrays), (no restarts, no cross-seed selection by directive). Polls the managed-job queue every 10 min (never faster: sshd lockout
# history). Usage: run_dyna2.sh <lewm|pldm> [ITERS=2] [DATE]
set -uo pipefail
export PATH="$HOME/.sky/bin:$PATH"
BASE=${1:?lewm|pldm}; ITERS=${2:-2}; DATE=${3:-20260903}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
USERV=armin@pantheon.inc
LOG=${DYNA2_LOG:-$ROOT/scratchpad/dyna2_driver_${BASE}${TAG_SUFFIX:-}.log}; mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date -u +%m%d-%H:%M)] $*" | tee -a "$LOG"; }
PSUF=""; [ "$BASE" = pldm ] && PSUF="-p"
BASE_ACTOR_TAG=rlp-cu-v2l05-${DATE/20260903/20260902}; [ "$BASE" = pldm ] && BASE_ACTOR_TAG=rlp-cu-v2l05-p-p-20260902
ACTOR_GLOB="g_cube_${BASE}_unig_ctrl_a2.5_s*.pt"
ANCHOR=${ANCHOR_WEIGHT:-1.0}; FREEZE=${FREEZE_ENCODER:-0}
SUF=${TAG_SUFFIX:-}                 # ablation arms: distinct tags, same pipeline
CFT=${COLLECT_FROM_TAG:-}           # ablation arms: reuse a finished collection

job_status(){ # <name> -> status word or ""
  sky jobs queue --limit 400 2>/dev/null | grep -E "^ *[0-9]+ +- +$1 " | head -1 \
    | grep -oE "SUCCEEDED|FAILED_SETUP|FAILED_PRECHECKS|FAILED_NO_RESOURCE|FAILED_CONTROLLER|FAILED|CANCELLED|CANCELLING|RUNNING|STARTING|PENDING|RECOVERING|SUBMITTED" | head -1
}
wait_job(){ # <name> -> 0 on SUCCEEDED, 1 otherwise
  local NAME=$1 st
  while :; do
    sleep 600
    st=$(job_status "$NAME")
    case "$st" in
      SUCCEEDED) log "$NAME SUCCEEDED"; return 0;;
      FAILED*|CANCELLED|CANCELLING) log "$NAME $st"; return 1;;
      "") log "$NAME not in queue";;
      *) : ;;
    esac
  done
}
launch_collect(){ # <iter> <actor_tag> <wm_src> <out_tag>
  local IT=$1 AT=$2 WS=$3 OT=$4
  log "launch collect+ft $OT (actors $AT, wm ${WS:-base})"
  sky jobs launch scripts/sky/dyna2/dyna2_collect_ft.yaml -n "$OT" --priority p1 -y --async \
    --env BASE=$BASE --env ITER=$IT --env ACTOR_TAG="$AT" --env ACTOR_GLOB="$ACTOR_GLOB" \
    --env WM_SRC="$WS" --env OUT_TAG="$OT" --env ANCHOR_WEIGHT=$ANCHOR --env FREEZE_ENCODER=$FREEZE \
    --env EPOCHS=1 --env NCALL=12 --env OFFSETS="25 100" --env MAXPAR=4 --env SMOKE=0 --env COLLECT_FROM_TAG="$CFT" \
    --env PANTHEON_USER=$USERV 2>&1 | tail -1 | tee -a "$LOG"
}
launch_train(){ # <tag> <wm_dir> <cache_version> [solver_extra]
  local TAG=$1 WMD=$2 CV=$3 SX=${4:-} RO=0 IMP=""
  [ -n "$SX" ] && { RO=1; IMP=$5; }
  log "launch train+eval $TAG (wm $WMD, cache $CV, reuse=$RO ${SX})"
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$TAG" --priority p1 -y --async \
    --env ENVNAME=cube --env BASE=$BASE --env GRID=unig --env ONLYCFG=unig_ctrl_a2.5 \
    --env SPLIT=0 --env SMOKE=0 --env INCLUDE_WINNERS=0 --env STAGED=0 --env FULLCACHE=0 \
    --env REUSE_ONLY=$RO --env ACTOR_IMPORT_TAG="$IMP" --env EVAL_RAWDIR=volume --env EVAL_SOLVER_EXTRA="$SX" \
    --env CU_DYNA=0 --env CU_DYNA_WM="$WMD" --env CU_EXPAND=1.0 --env CU_GAMMA=0.98 --env TR_NSTEP=1 \
    --env CRIT_DEPTH=2 --env UNIG_LAYERS=2 --env UNIG_MD=20 --env UNIG_K=8 --env ACR_LAM=0.5 \
    --env STEPS=6000 --env TD_STEPS=12000 --env BATCH=256 --env MAXPAR=4 --env RH=5 \
    --env CKPT_SELECT=1 --env CKPT_EVERY=2000 --env CKPT_VAL_SEEDS="48 49 50 51" \
    --env CACHE_VERSION="$CV" --env EXPERIMENT_TAG="$TAG" \
    --env TRAIN_SEEDS="0 1 2" --env EVAL_SEEDS="42 43 44" --env UNIG_OFF="25,100" \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer --env PANTHEON_USER=$USERV 2>&1 | tail -1 | tee -a "$LOG"
}

log "=== DYNA2 driver base=$BASE iters=$ITERS anchor=$ANCHOR freeze=$FREEZE ==="
ACTOR_TAG=$BASE_ACTOR_TAG; WM_SRC=""
for IT in $(seq 1 $ITERS); do
  CT=rlp-cu-dyna2-${BASE}${SUF}-it${IT}-${DATE}
  TT=rlp-cu-dyna2t-${BASE}${SUF}-it${IT}-${DATE}
  WMDIR=/checkpoints/$USERV/$CT/wm
  ST=$(job_status "$CT")
  if [ "$ST" != SUCCEEDED ]; then
    case "$ST" in RUNNING|STARTING|PENDING|RECOVERING|SUBMITTED) log "$CT already $ST; waiting";; *) launch_collect $IT "$ACTOR_TAG" "$WM_SRC" "$CT";; esac
    if ! wait_job "$CT"; then   # one retry: a full-raid0 node fails fast in setup
      log "collect it$IT failed once; relaunching"; sleep 60
      launch_collect $IT "$ACTOR_TAG" "$WM_SRC" "$CT"; wait_job "$CT" || { log "ABORT at collect it$IT"; exit 1; }
    fi
  fi
  ST=$(job_status "$TT")
  if [ "$ST" != SUCCEEDED ]; then
    case "$ST" in RUNNING|STARTING|PENDING|RECOVERING|SUBMITTED) log "$TT already $ST; waiting";; *) launch_train "$TT" "$WMDIR" "cu-dyna2-${BASE}${SUF}-it${IT}-n1s2-v1";; esac
    if ! wait_job "$TT"; then   # one retry (preemption storms at p2 killed the first attempt 8x)
      log "train it$IT failed once; relaunching"; sleep 60
      launch_train "$TT" "$WMDIR" "cu-dyna2-${BASE}${SUF}-it${IT}-n1s2-v1"; wait_job "$TT" || { log "ABORT at train it$IT"; exit 1; }
    fi
  fi
  ACTOR_TAG=$TT; WM_SRC=$WMDIR
done
# no deploy-time restarts by directive (2026-09-04): the planner runs its K=8
# refinement once, always; the seed lottery is reported, not selected away.
log "DYNA2 DONE base=$BASE final actors $ACTOR_TAG wm $WM_SRC"
