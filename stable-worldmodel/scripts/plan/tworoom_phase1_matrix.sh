#!/bin/bash
# TwoRoom matrix campaign, PHASE 1: per-base latent caches, TD warm-starts and
# LIPv4 actors -- everything the pre-Dyna matrix needs.
#
# Differs from tworoom_bases_phase1.sh in one way that matters: TD is trained
# with THREE initialization seeds (0,1,2), not one, because the matrix reports
# TD+solver as a 3-training-seed mean. LIPv4 likewise gets 3 seeds (that part
# matches run_bases_lip4.sh, minus the interleaved cards).
#
# Usage: tworoom_phase1_matrix.sh <base:lejepa|pldm|dinowm> <gpu> [iters]
#   base -> checkpoint "<base>_tworoom" (see convert_tworoom_bases.py)
#   iters = LIP refinement K, default 8; DINO may need a lower cap (77224-d
#           latent), pass e.g. 4 and the actors get a "k4" suffix.
# Idempotent: existing caches / TD files / actors are skipped.
set -u
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/swm_home}
export CODE=${CODE:-/workspace/code/stable-worldmodel}
export PYTHONPATH=$CODE
export HF_HOME=${HF_HOME:-/root/hf}
export TQDM_DISABLE=1
# OMP cap is load-bearing on this box: uncapped threads cost ~40x on TwoRoom.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

BASE=$1
GPU=$2
ITERS=${3:-8}
SUFF=""; [ "$ITERS" != "8" ] && SUFF="k${ITERS}"

case "$BASE" in
  lejepa|pldm|dinowm) ;;
  *) echo "unknown base $BASE (expected lejepa|pldm|dinowm)"; exit 1;;
esac

TRM=$CODE/scripts/trm
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
CACHES=/workspace/caches
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
CKPT="${BASE}_tworoom"
H5=$CACHES/tworoom_play.h5
CACHE1=$CACHES/tworoom_${BASE}_fs1.pt
CACHE5=$CACHES/tworoom_${BASE}_fs5.pt
TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-28800}

mkdir -p "$LOGS" "$CACHES" "$MET" "$ACT"
DRV=$LOGS/phase1_matrix_${BASE}${SUFF}.log
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][p1-${BASE}] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }

# ------------------------------------------------------------------ caches
if [ ! -f "$CACHE1" ]; then
  log "fs1 latent cache (stride 1 -- dense states, per the TD cache-stride lesson)"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$TRM/cache_latents.py" --wm "$CKPT" \
    --dataset tworoom_play.lance --out "$CACHE1" --state-key state --batch-size 256 \
    > "$LOGS/cache_${BASE}_fs1.log" 2>&1 || die "fs1 cache failed"
fi
log "fs1 ok ($(du -h "$CACHE1" | cut -f1))"

if [ ! -f "$CACHE5" ]; then
  log "fs5 subsample (frameskip 5 -- the LIP/action-block view)"
  $PY "$TRM/subsample_cache.py" --in "$CACHE1" --out "$CACHE5" --frameskip 5 \
    > "$LOGS/cache_${BASE}_fs5.log" 2>&1 || die "fs5 subsample failed"
fi
log "fs5 ok"

# ------------------------------------------------- TD warm-starts x 3 seeds
# tau 0.1: LOW expectile for cost-to-go / quasimetric distance (optimistic
# toward the min). n-step 50 in primitive steps.
for ts in 0 1 2; do
  TD=$MET/td_${BASE}_e0.1_n50_s${ts}.pt
  if [ ! -f "$TD" ]; then
    log "TD warm-start seed ${ts} (tau 0.1, n-step 50, 6k steps)"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$PLAN/train_metric.py" --cache "$CACHE1" \
      --learner td --head quasimetric --expectile 0.1 --n-step 50 \
      --steps 6000 --seed "$ts" --out "$TD" \
      > "$LOGS/td_${BASE}_s${ts}.log" 2>&1 || die "TD seed ${ts} failed"
  fi
  log "TD s${ts} ok"
done

# ------------------------------------------------------- LIPv4 actors x 3 seeds
# Canonical triple-perfect TwoRoom recipe: arch v4 (gate-free min0), amax 2.2,
# max-delta 12. Each actor warm-starts from the TD critic of the SAME seed so
# the two 3-seed families stay paired.
for s in 0 1 2; do
  OUT=$ACT/trm_${BASE}_v4${SUFF}_s${s}.pt
  TD=$MET/td_${BASE}_e0.1_n50_s${s}.pt
  if [ ! -f "$OUT" ]; then
    [ -f "$TD" ] || die "missing TD warm-start $TD"
    log "LIPv4 train seed ${s} (K=${ITERS})"
    CUDA_VISIBLE_DEVICES=$GPU timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" \
      --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm "$CKPT" \
      --init-value "$TD" \
      --arch v4 --amax 2.2 --max-delta 12 --iters "$ITERS" --horizon 5 --steps 8000 \
      --n-step 50 --expectile 0.1 --expectile-final 0.03 \
      --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --actor-lr 3e-4 --actor-lr-final 3e-5 \
      --seed "$s" \
      --out "$OUT" --out-value "$MET/trm_${BASE}_v4${SUFF}_s${s}_value.pt" \
      > "$LOGS/train_lip_${BASE}${SUFF}_s${s}.log" 2>&1 \
      || die "LIP seed ${s} failed (see $LOGS/train_lip_${BASE}${SUFF}_s${s}.log)"
  fi
  log "LIP s${s} ok"
done

log "PHASE1_${BASE}${SUFF}_DONE"
