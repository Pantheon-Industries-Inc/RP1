#!/bin/bash
# TwoRoom min0 campaign (2026-07-14, pod a 4xH100).
# Goal: min0 tandem actor ([A, grad V, E] inputs, LIP-AC warm) with a PERFECT
# 12-cell card: {std,hard} x {h25,h50} x eval seeds {42,43,44}, n=50.
#
# Phases (idempotent; rows cached in /workspace/results/summary_tworoom.csv):
#   anchors: lv1_e01n50_seed1 card (expect 100 everywhere; hard gate >=98)
#            + latent+CEM std/hard h50 s42 (expect 68 / 56; data-identity check)
#   screen : 6 arms, seed 0 -> hard_h25_s42 + hard_h50_s42 + std_h50_s42
#   cards  : every arm with screen sum == 300 -> full 12-cell card
#   seeds  : best arm -> retrain seeds 1,2 -> full cards
# Select on success only; E_final logged as indicator, never selects.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
H5=/workspace/caches/tworoom_play.h5
CACHE1=/workspace/caches/tworoom_lewm_fs1.pt
CACHE5=/workspace/caches/tworoom_lewm_fs5.pt
TD=$MET/td_e0.1_n50.pt
SUM=$RES/summary_tworoom.csv
DRV=$LOGS/driver_tworoom_min0.log
TRAIN_TIMEOUT=10800
EVAL_TIMEOUT=3600

mkdir -p "$LOGS" "$RES" "$MET" "$ACT"
touch "$SUM"

log(){ echo "[$(date +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

run_eval(){ # name gpu seed offset budget surface(std|hard) extra...
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5 surface=$6; shift 6
  if grep -q "^${name}," "$SUM"; then
    log "eval ${name}: cached ($(sc "$name"))"; return 0
  fi
  local hard=()
  [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name tworoom_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" \
    "+video_dir=$RES/videos_${name}" "${hard[@]}" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "${sr:-}" ]; then
    log "eval ${name}: FAILED (see $LOGS/eval_${name}.log)"
    echo "${name},FAIL" >> "$SUM"; return 1
  fi
  echo "${name},${sr}" >> "$SUM"
  log "eval ${name}: ${sr}"
}

card(){ # tag gpu actor -> 12 cells sequentially on one gpu
  local tag=$1 gpu=$2 actor=$3
  for surface in std hard; do
    for seed in 42 43 44; do
      run_eval "card_${tag}_${surface}_h25_s${seed}" "$gpu" "$seed" 25 50 "$surface" solver=lip "solver.actor_path=$actor"
      run_eval "card_${tag}_${surface}_h50_s${seed}" "$gpu" "$seed" 50 100 "$surface" solver=lip "solver.actor_path=$actor"
    done
  done
  log "card ${tag} complete"
}

card_sum(){ # tag -> sum of 12 cells (FAIL/missing -> -1)
  local tag=$1 total=0 v
  for surface in std hard; do
    for seed in 42 43 44; do
      for h in h25 h50; do
        v=$(sc "card_${tag}_${surface}_${h}_s${seed}")
        { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && { echo "-1"; return; }
        total=$(awk "BEGIN{print $total + $v}")
      done
    done
  done
  echo "$total"
}

# ------------------------------------------------------------------ arms
# TwoRoom-canonical base: teacher/critic tau 0.1 CONST (tworoom lesson: tau0.03
# gradient field misleads LIP at h50), n-step 50, K8 lr3e-4 8k, warm from TD.
BASE_COMMON="--cache $CACHE5 --cache-td $CACHE1 --h5 $H5 --wm lewm_tworoom
  --init-value $TD --horizon 5 --iters 8 --steps 8000 --n-step 50
  --expectile 0.1 --critic-lr 1e-3 --actor-lr 3e-4"
DROPS="--drop-z0 --drop-zg"

flags(){ case $1 in
  base)   echo "$DROPS";;
  ng)     echo "$DROPS --no-gate";;
  sched)  echo "$DROPS --critic-lr-final 1e-4 --expectile-final 0.03 --actor-lr-final 3e-5";;
  md12)   echo "$DROPS --max-delta 12";;
  ax35)   echo "$DROPS --amax 3.5";;
  fullin) echo "";;
  *) return 1;;
esac; }

train_arm(){ # gpu arm seed
  local gpu=$1 arm=$2 seed=$3
  local out="$ACT/trm_min0_${arm}_s${seed}.pt" fl
  fl=$(flags "$arm") || die "unknown arm $arm"
  [ -f "$out" ] && { log "train ${arm}_s${seed}: cached"; return 0; }
  log "train ${arm}_s${seed}: start gpu${gpu} [$fl]"
  echo "CMD: train_lip_ac.py $BASE_COMMON $fl --seed $seed" > "$LOGS/train_trm_min0_${arm}_s${seed}.log"
  CUDA_VISIBLE_DEVICES=$gpu timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" \
    $BASE_COMMON $fl --seed "$seed" \
    --out "$out" --out-value "$MET/trm_min0_${arm}_s${seed}_value.pt" \
    >> "$LOGS/train_trm_min0_${arm}_s${seed}.log" 2>&1 \
    || { log "train ${arm}_s${seed}: FAILED (see log)"; return 1; }
  local ef; ef=$(grep -E "^step" "$LOGS/train_trm_min0_${arm}_s${seed}.log" | tail -1 \
    | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  log "train ${arm}_s${seed}: done, E_final ${ef:-NA} (indicator only)"
}

# ------------------------------------------------------------------ anchors
log "=== phase anchors"
(
  for seed in 42 43 44; do
    run_eval "card_lv1ctrl_std_h25_s${seed}" 0 "$seed" 25 50 std solver=lip "solver.actor_path=$ACT/lv1_e01n50_seed1.pt"
    run_eval "card_lv1ctrl_std_h50_s${seed}" 0 "$seed" 50 100 std solver=lip "solver.actor_path=$ACT/lv1_e01n50_seed1.pt"
  done
) &
(
  for seed in 42 43 44; do
    run_eval "card_lv1ctrl_hard_h25_s${seed}" 1 "$seed" 25 50 hard solver=lip "solver.actor_path=$ACT/lv1_e01n50_seed1.pt"
    run_eval "card_lv1ctrl_hard_h50_s${seed}" 1 "$seed" 50 100 hard solver=lip "solver.actor_path=$ACT/lv1_e01n50_seed1.pt"
  done
) &
run_eval "latent_std_h50_s42" 2 42 50 100 std &
run_eval "latent_hard_h50_s42" 3 42 50 100 hard &
wait

anchor_fail=0
for surface in std hard; do
  for seed in 42 43 44; do
    for h in h25 h50; do
      v=$(sc "card_lv1ctrl_${surface}_${h}_s${seed}")
      { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && { log "anchor cell ${surface}_${h}_s${seed}: MISSING/FAIL"; anchor_fail=1; continue; }
      awk "BEGIN{exit !($v < 98)}" && { log "anchor cell ${surface}_${h}_s${seed}: $v < 98"; anchor_fail=1; }
    done
  done
done
[ "$anchor_fail" -eq 0 ] || die "anchor card degraded — rebuilt data does not reproduce the old harness"
log "anchor card OK (all cells >= 98; expected 100s). latent anchors: std_h50 $(sc latent_std_h50_s42) (ref 68.0), hard_h50 $(sc latent_hard_h50_s42) (ref 56.0)"

# ------------------------------------------------------------------ screen
log "=== phase screen (6 arms, seed 0)"
train_arm 0 base 0 & train_arm 1 ng 0 & train_arm 2 sched 0 & train_arm 3 md12 0 &
wait
train_arm 0 ax35 0 & train_arm 1 fullin 0 &
wait
log "screen trainings done"

ARMS="base ng sched md12 ax35 fullin"
gpu=0
for arm in $ARMS; do
  a="$ACT/trm_min0_${arm}_s0.pt"
  [ -f "$a" ] || { log "screen ${arm}: no ckpt, skip"; continue; }
  run_eval "m0_${arm}_hard_h25_s42" "$gpu" 42 25 50 hard solver=lip "solver.actor_path=$a" &
  gpu=$(( (gpu + 1) % 4 )); [ "$gpu" -eq 0 ] && wait
done
wait
gpu=0
for arm in $ARMS; do
  a="$ACT/trm_min0_${arm}_s0.pt"
  [ -f "$a" ] || continue
  run_eval "m0_${arm}_hard_h50_s42" "$gpu" 42 50 100 hard solver=lip "solver.actor_path=$a" &
  gpu=$(( (gpu + 1) % 4 )); [ "$gpu" -eq 0 ] && wait
done
wait
gpu=0
for arm in $ARMS; do
  a="$ACT/trm_min0_${arm}_s0.pt"
  [ -f "$a" ] || continue
  run_eval "m0_${arm}_std_h50_s42" "$gpu" 42 50 100 std solver=lip "solver.actor_path=$a" &
  gpu=$(( (gpu + 1) % 4 )); [ "$gpu" -eq 0 ] && wait
done
wait

log "=== screen results"
declare -A SSUM
for arm in $ARMS; do
  a=$(sc "m0_${arm}_hard_h25_s42"); b=$(sc "m0_${arm}_hard_h50_s42"); c=$(sc "m0_${arm}_std_h50_s42")
  if [ -z "$a" ] || [ -z "$b" ] || [ -z "$c" ] || [ "$a" = "FAIL" ] || [ "$b" = "FAIL" ] || [ "$c" = "FAIL" ]; then
    SSUM[$arm]=-1; log "screen ${arm}: INVALID (${a:-na}/${b:-na}/${c:-na})"
  else
    SSUM[$arm]=$(awk "BEGIN{print $a + $b + $c}")
    log "screen ${arm}: ${a}/${b}/${c} sum ${SSUM[$arm]}"
  fi
done

# ------------------------------------------------------------------ cards
# every arm with a perfect screen gets a card; fallback: top-2 by screen sum
PROMO=""
for arm in $ARMS; do
  [ "${SSUM[$arm]}" = "300" ] && PROMO="$PROMO $arm"
done
if [ -z "$PROMO" ]; then
  log "no arm screened 300/300 — falling back to top-2 by screen sum"
  PROMO=$(for arm in $ARMS; do echo "${SSUM[$arm]} $arm"; done | sort -rn | head -2 | awk '{print $2}' | tr '\n' ' ')
fi
log "=== phase cards: promoting [$PROMO]"
gpu=0
for arm in $PROMO; do
  card "m0${arm}-s0" "$gpu" "$ACT/trm_min0_${arm}_s0.pt" &
  gpu=$(( (gpu + 1) % 4 )); [ "$gpu" -eq 0 ] && wait
done
wait

best=""; best_sum=-1
for arm in $PROMO; do
  s=$(card_sum "m0${arm}-s0")
  log "card m0${arm}-s0 sum: $s"
  awk "BEGIN{exit !($s > $best_sum)}" && { best_sum=$s; best=$arm; }
done
[ -n "$best" ] || die "no valid cards"
log "best arm: $best (card sum $best_sum / 1200)"

# ------------------------------------------------------------------ seeds
log "=== phase seeds: $best seeds 1,2"
train_arm 0 "$best" 1 & train_arm 1 "$best" 2 &
wait
card "m0${best}-s1" 0 "$ACT/trm_min0_${best}_s1.pt" &
card "m0${best}-s2" 1 "$ACT/trm_min0_${best}_s2.pt" &
wait

log "=== FINAL SUMMARY"
sort "$SUM" | tee -a "$DRV"
perfect=""
for arm in $PROMO; do
  for s in 0 1 2; do
    [ "$(card_sum "m0${arm}-s${s}")" = "1200" ] && perfect="$perfect m0${arm}-s${s}"
  done
done
log "PERFECT CARDS:${perfect:- none}"
touch "$RES/tworoom_min0.DONE"
log "ALL DONE"
