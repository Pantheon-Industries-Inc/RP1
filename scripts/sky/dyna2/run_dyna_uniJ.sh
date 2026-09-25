#!/usr/bin/env bash
# Dyna under the unified recipe, iterations 1..N, both bases, both horizon stacks -- RLP is RETRAINED WITH THE BASE
# CELLS' WINNING CONFIGS (the locked shared stop: teacher budget T, actor step N), never re-selected.
#   scripts/sky/dyna2/run_dyna_uniJ.sh <N iters> [DATE]
# Iteration k, per base:
#   (1) collection actors = the h25-stack deployed actors of iteration k-1 (k=1: the base cell's shared pair, staged
#       from its seed-split tags into rlp-cu-dynaJ-<base>-actors-it1-<date>; k>1: the previous dyna h25 cell tag)
#   (2) dyna2_collect_ft.yaml: collect at offsets 25+100 with the six actors, anchored fine-tune from the previous
#       WM (k=1: the base WM) -> /checkpoints/<user>/rlp-cu-dynaJ-<base>-it<k>-<date>/wm
#   (3) per stack: one 4-GPU job, six seeds, the cell's fixed config (DYNA_ROW/DYNA_STEP), lock-free 8 slots,
#       evaluated on validation (48-51) and test (42-44) draws -> tag rlp-cu-uniJ-h<hz>[-p]-dyna<k>-<date>
#   (4) scripts/cube_dyna_report.sh -> per-seed val/test, n=6 row (appended to scratchpad/dynaJ_results.md)
# The iteration that "works best" is chosen on the VALIDATION mean; its test row is the reported one.
set -uo pipefail
N=${1:-3}; DATE=${2:-20260923}; BASEDATE=${BASEDATE:-20260921}
# HZLIST: which horizon stacks to RETRAIN and report. User decision 2026-09-24: h25 only -- Dyna's gain is
# h25-specific (the h25 and h100 hard cores are disjoint; two iterations moved 1 of 17 h100 core tasks), so the
# h100 retrains cost four GPU-hours per iteration for a number inside seed noise. COLLECTION still runs at BOTH
# offsets (OFFSETS="25 100" below): the both-offset collection is what made iteration 2 work at all, and cutting
# it would break comparability with the config-B Dyna rows.
HZLIST=${DYNA_HZ:-"25 100"}
# WORKSPACE: each researcher's SkyPilot workspace has one node (8 GPUs) guaranteed plus a share of the
# spare pool (user, 2026-09-24). Pin Dyna to it explicitly rather than relying on the client-wide
# preferred workspace, so the chain lands in the right quota whoever runs the driver.
WS=${WORKSPACE:-armin}; WSARG=""; [ -n "$WS" ] && WSARG="--workspace $WS"
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"; export PATH="$HOME/.sky/bin:$PATH"
USERV=armin@pantheon.inc; SP=${SP:-$ROOT/scratchpad}; mkdir -p "$SP"; LOG=$SP/dynaJ_driver.log; RES=$SP/dynaJ_results.md
log(){ echo "[$(date -u +%m%d-%H:%M)] $*" | tee -a "$LOG"; }
jid(){ grep -o "sky jobs logs [0-9]*" | grep -o "[0-9]*$" | head -1; }
wait_job(){ local id=$1 st=""; while :; do st=$(sky jobs queue --limit 400 2>/dev/null | awk -v j=$id '$1==j' | grep -oE 'SUCCEEDED|FAILED[A-Z_]*|CANCELLED' | head -1); [ -n "$st" ] && { echo "$st"; return; }; sleep 300; done; }
pick_of(){ grep -h "^LOCK $1:" $SP/lock_$1.out 2>/dev/null | tail -1 | sed -E 's/.*row=([^ ]+) step=([^ ]+).*/\1 \2/'; }
# START_IT: resume a killed driver at a later iteration (step 3 has no idempotency guard, so a
# restart at an iteration whose retrain jobs are already RUNNING would launch duplicates).
for IT in $(seq ${START_IT:-1} $N); do
  log "=== Dyna-uniJ iteration $IT (fixed configs from the base cells)"
  declare -A CID
  for BASE in lewm pldm; do
    P=""; [ "$BASE" = pldm ] && P="-p"
    H25CELL=rlp-cu-uniJ-h25${P}-${BASEDATE}; read -r ROW25 STEP25 <<< "$(pick_of "$H25CELL")"
    [ -n "${ROW25:-}" ] || { log "no base lock for $H25CELL"; exit 1; }
    CTCHK=rlp-cu-dynaJ-${BASE}-it${IT}-${DATE}
    if [ "$(sky jobs queue --limit 400 2>/dev/null | awk -v t="$CTCHK" '$0 ~ t {for(i=1;i<=NF;i++) if ($i=="SUCCEEDED") {print "y"; exit}}' | head -1)" = y ]; then
      log "collect+ft $CTCHK already SUCCEEDED; skipping (no re-stage)"; CID[$BASE]=done; continue
    fi
    if [ "$IT" = 1 ]; then
      AIMP=rlp-cu-dynaJ-${BASE}-actors-it1-${DATE}; STEM=${H25CELL%-*}
      ITEMS=""; for s in 0 1 2; do ITEMS="$ITEMS $H25CELL:cube:$BASE:$ROW25:$s:$STEP25"; done
      for s in 3 4; do ITEMS="$ITEMS ${STEM}-s345-${BASEDATE}:cube:$BASE:$ROW25:$s:$STEP25"; done; ITEMS="$ITEMS ${STEM}-s5-${BASEDATE}:cube:$BASE:$ROW25:5:$STEP25"
      SID=$(sky jobs launch -y -d $WSARG --name "dynaJ-stage-${BASE}" --env ITEMS="$ITEMS" --env DEST_TAG="$AIMP" scripts/sky/tools/fixed_stage.yaml 2>&1 | jid)
      st=$(wait_job $SID); log "stage $BASE -> $AIMP ($SID): $st"; [ "$st" = SUCCEEDED ] || exit 1
      WMSRC=""
    else
      AIMP=rlp-cu-uniJ-h25${P}-dyna$((IT-1))-${DATE}          # previous iteration's h25 cell: deployed finals + values
      WMSRC=/checkpoints/$USERV/rlp-cu-dynaJ-${BASE}-it$((IT-1))-${DATE}/wm
    fi
    CT=rlp-cu-dynaJ-${BASE}-it${IT}-${DATE}
    # idempotent: a collect+ft job of this tag that already SUCCEEDED exported its world model -> skip on a restart.
    # (The driver runs on the laptop, so /checkpoints is NOT visible here -- ask the job queue, not the filesystem.)
    PREVST=$(sky jobs queue --limit 400 2>/dev/null | awk -v t="$CT" '$0 ~ t {for(i=1;i<=NF;i++) if ($i=="SUCCEEDED") {print "SUCCEEDED"; exit}}' | head -1)
    if [ "$PREVST" = SUCCEEDED ]; then log "collect+ft $CT already SUCCEEDED; skipping"; CID[$BASE]=done; continue; fi
    CID[$BASE]=$(sky jobs launch scripts/sky/dyna2/dyna2_collect_ft.yaml -n "$CT" --priority p1 $WSARG -y -d \
      --env BASE=$BASE --env ITER=$IT --env ACTOR_TAG="$AIMP" --env ACTOR_GLOB="g_cube_${BASE}_${ROW25}_s*.pt" --env NACTORS=6 \
      --env WM_SRC="$WMSRC" --env OUT_TAG="$CT" --env ANCHOR_WEIGHT=1.0 --env FREEZE_ENCODER=0 --env EPOCHS=1 --env NCALL=6 \
      --env OFFSETS="25 100" --env MAXPAR=4 --env SMOKE=0 --env PANTHEON_USER=$USERV 2>&1 | jid)
    log "collect+ft $CT job ${CID[$BASE]} (actors $AIMP, row $ROW25, wm ${WMSRC:-base})"
  done
  for BASE in lewm pldm; do
    [ "${CID[$BASE]}" = done ] && continue
    st=$(wait_job ${CID[$BASE]}); log "collect+ft $BASE it$IT: $st"; [ "$st" = SUCCEEDED ] || { log "ABORT iteration $IT"; exit 1; }
  done
  # (3) retrain the fixed configs on the fine-tuned WMs: 4 cells x one 4-GPU job (six seeds, lock-free 8 slots)
  declare -A TID
  for BASE in lewm pldm; do P=""; [ "$BASE" = pldm ] && P="-p"; WM=/checkpoints/$USERV/rlp-cu-dynaJ-${BASE}-it${IT}-${DATE}/wm
    for HZ in $HZLIST; do
      CELL=rlp-cu-uniJ-h${HZ}${P}-${BASEDATE}; read -r ROW STEP <<< "$(pick_of "$CELL")"; [ -n "${ROW:-}" ] || { log "no base lock for $CELL"; exit 1; }
      OUT=$(ENVS=cube HZ=$HZ ONLY_BASES=$BASE GPUS=4 SLOTS=8 CUBE_LOCK=none DATE=$DATE SEEDS="0 1 2 3 4 5" WORKSPACE="$WS" \
            DYNA_WM=$WM DYNA_IT=$IT DYNA_ROW=$ROW DYNA_STEP=$STEP bash scripts/sky/unigamma/launch_uniB.sh 2>&1)
      TID[$BASE$HZ]=$(echo "$OUT" | jid); log "retrain $BASE h$HZ it$IT: row $ROW step $STEP -> job ${TID[$BASE$HZ]:-?}"
    done
  done
  for k in "${!TID[@]}"; do st=$(wait_job ${TID[$k]}); log "retrain job ${TID[$k]} ($k): $st"; done
  # (4) read the rows
  for BASE in lewm pldm; do P=""; [ "$BASE" = pldm ] && P="-p"
    for HZ in $HZLIST; do TAG=rlp-cu-uniJ-h${HZ}${P}-dyna${IT}-${DATE}
      WORKSPACE="$WS" SP=$SP scripts/cube_dyna_report.sh "$TAG" "$HZ" | tee -a "$RES" | tee -a "$LOG"
    done
  done
  unset CID TID
done
log "=== Dyna-uniJ done: pick the iteration per cell by VALIDATION mean in $RES"
