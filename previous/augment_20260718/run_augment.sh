#!/bin/bash
# GRASP-MISS DATASET AUGMENTATION for LIP on v2WM (2026-07-18). User ask: train
# LIP on the usual (lewm-cube expert) dataset + explicitly augment ~10% grasp-miss
# in BOTH the critic AND the actor data. WM stays v2WM (isolate the data effect).
#
# Baseline (matched, in hand): LIPv4 --arch v4 on v2WM clean caches = 87.6
# (the gradient-ablation control, seeds 0/1/2 = 86.7/88.0/88.0).
# This run = SAME arch/seeds/recipe, only the caches (+ derived TD) differ.
# Prior (ftmisses): grasp-miss x10 in critic ALONE on v2WM = -5 (WM can't imagine
# misses). New variable: misses in the ACTOR context data too.
#
# Pipeline: collect long (90-frame) misses -> concat -> encode(v2WM) -> merge x10
# into fs1(critic)+fs5(actor)+combined-actions-h5 -> TD on aug fs1 -> LIPv4 x3 seeds.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
SCR=/workspace/scripts
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
CACHES=/workspace/caches
DAT=/workspace/datasets
PY=python3
WMV2=/workspace/ckpts/ogbench_cube_single_v2WM
H5=$DAT/lewm_cube_full/cube_single_expert.h5
EXP_FS1=$CACHES/cube_v2wm_fs1.pt
AUGDIR=$CACHES/augment
AUG_FS1=$AUGDIR/aug_fs1.pt
AUG_FS5=$AUGDIR/aug_fs5.pt
AUG_ACT=$AUGDIR/aug_actions.h5
AUG_TD=$MET/aug_dE_t003n50.pt
MISS_ALL=$DAT/miss_long/miss_all.h5
MISS_FS1=$CACHES/miss_fs1.pt
SUM=$RES/summary_augment.csv
DRV=$LOGS/driver_augment.log
NWORKERS=${1:-16}
TARGET=${2:-8}
REP=${3:-10}
TRAIN_TIMEOUT=28800
EVAL_TIMEOUT=14400

mkdir -p "$LOGS" "$RES" "$MET" "$ACT" "$CACHES" "$AUGDIR" "$DAT/miss_long" /workspace/miss_collect /workspace/swm_home
touch "$SUM"

log(){ echo "[$(date +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }

run_eval(){ # name gpu seed offset budget extra-hydra-args...
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5; shift 5
  local c; c=$(sc "$name"); [ -n "$c" ] && { log "eval ${name}: cached (${c})"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name cube \
    seed="$seed" eval.dataset_name="$H5" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    output.filename="${name}.txt" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED (see log)"; echo "${name},FAIL" >> "$SUM"; return 1; }
  echo "${name},${sr}" >> "$SUM"; log "eval ${name}: ${sr}"
}

log "=== augment campaign start (pid $$) nworkers=$NWORKERS target=$TARGET rep=$REP ==="
[ -f "$EXP_FS1" ] || die "expert fs1 cache missing"
[ -f "$H5" ] || die "expert h5 missing"

# ---------------------------------------------------------------- 1. collect misses
if [ ! -f "$MISS_ALL" ]; then
  log "phase 1: collect grasp-miss episodes ($NWORKERS workers x $TARGET)"
  rm -f "$DAT/miss_long"/shard_misslong_*.h5
  local_i=0
  for i in $(seq 0 $((NWORKERS-1))); do
    $PY "$SCR/collect_miss_long.py" "$i" "$NWORKERS" "$TARGET" \
      > "/workspace/miss_collect/w${i}.log" 2>&1 &
  done
  wait
  n_sh=$(ls "$DAT/miss_long"/shard_misslong_*.h5 2>/dev/null | wc -l)
  [ "$n_sh" -gt 0 ] || die "no miss shards produced"
  log "collected $n_sh shards; concatenating"
  $PY "$SCR/concat_shards.py" "$DAT/miss_long/shard_misslong_*.h5" "$MISS_ALL" \
    > "$LOGS/concat_miss.log" 2>&1 || die "concat failed"
  grep -h "wrote" "$LOGS/concat_miss.log" | tail -1 | tee -a "$DRV"
fi
log "miss_all ready"

# ---------------------------------------------------------------- 2. encode (v2WM)
if [ ! -f "$MISS_FS1" ]; then
  log "phase 2: encode miss_all with v2WM (stride 1)"
  CUDA_VISIBLE_DEVICES=0 $PY "$SCR/cache_cube_full.py" \
    --h5 "$MISS_ALL" --wm "$WMV2" --stride 1 --out "$MISS_FS1" \
    > "$LOGS/encode_miss.log" 2>&1 || die "encode failed"
fi
log "miss_fs1 ready"

# ---------------------------------------------------------------- 3. merge x REP
if [ ! -f "$AUG_FS1" ] || [ ! -f "$AUG_FS5" ] || [ ! -f "$AUG_ACT" ]; then
  log "phase 3: merge x$REP -> aug fs1/fs5 + combined actions h5"
  $PY "$SCR/merge_augment.py" "$AUGDIR" "$REP" "$EXP_FS1" "$H5" "$MISS_FS1" "$MISS_ALL" \
    > "$LOGS/merge_augment.log" 2>&1 || die "merge failed"
  grep -hE "share|miss fs5 len|aug_actions" "$LOGS/merge_augment.log" | tee -a "$DRV"
fi
log "augmented caches ready"

# ---------------------------------------------------------------- 4. TD teacher on aug fs1
if [ ! -f "$AUG_TD" ]; then
  log "phase 4: TD teacher on aug fs1 (quasimetric tau0.03 n50 6k)"
  CUDA_VISIBLE_DEVICES=0 timeout $TRAIN_TIMEOUT $PY "$PLAN/train_metric.py" \
    --cache "$AUG_FS1" --learner td --head quasimetric \
    --expectile 0.03 --n-step 50 --steps 6000 --seed 0 \
    --out "$AUG_TD" > "$LOGS/td_augment.log" 2>&1 || die "TD failed"
fi
log "aug TD ready"

# ---------------------------------------------------------------- 5. train LIPv4 x3 seeds
train_aug(){ # gpu seed
  local gpu=$1 seed=$2
  local out="$ACT/lipaug_v2wm_s${seed}.pt"
  if [ ! -f "$out" ]; then
    log "train lipaug s${seed}: start gpu${gpu} (arch v4, aug caches + aug TD)"
    CUDA_VISIBLE_DEVICES=$gpu timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" \
      --cache "$AUG_FS5" --cache-td "$AUG_FS1" --h5 "$AUG_ACT" --wm "$WMV2" \
      --init-value "$AUG_TD" \
      --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5 \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 \
      --seed "$seed" --out "$out" --out-value "$MET/lipaug_v2wm_s${seed}_value.pt" \
      > "$LOGS/train_lipaug_s${seed}.log" 2>&1 \
      || { log "train lipaug s${seed}: FAILED (see log)"; return 1; }
    local ef; ef=$(grep -E "^step" "$LOGS/train_lipaug_s${seed}.log" | tail -1 \
      | grep -oE "E_final [0-9.]+" | awk '{print $2}')
    log "train lipaug s${seed}: done, E_final ${ef:-NA}"
  fi
  local s
  for s in 42 43 44; do
    run_eval "lipaug_v2wm_s${seed}_h25_s${s}" "$gpu" "$s" 25 50 \
      policy="$WMV2" solver=lip "solver.actor_path=$out"
  done
}
log "phase 5: LIPv4 on augmented data, seeds 0/1/2 (gpu1/2/3)"
train_aug 1 0 &
train_aug 2 1 &
train_aug 3 2 &
wait

log "=== AUGMENT SUMMARY (v2WM + ~10% grasp-miss in critic+actor, h25 s42/43/44) ==="
mean3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{ if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"} else {printf "%.1f",(a+b+c)/3} }'; }
tot=0; n=0
for sd in 0 1 2; do
  a=$(sc lipaug_v2wm_s${sd}_h25_s42); b=$(sc lipaug_v2wm_s${sd}_h25_s43); c=$(sc lipaug_v2wm_s${sd}_h25_s44)
  m=$(mean3 "$a" "$b" "$c"); log "lipaug s${sd} : ${a}/${b}/${c} -> ${m}"
  [ "$m" != "NA" ] && { tot=$(awk "BEGIN{print $tot+$m}"); n=$((n+1)); }
done
[ "$n" -gt 0 ] && log ">>> LIP+aug 3-seed mean = $(awk "BEGIN{printf \"%.1f\",$tot/$n}")  (baseline LIPv4/v2WM 87.6; ftmisses critic-only -5 = 82.7)"
log "AUGMENT DONE"
