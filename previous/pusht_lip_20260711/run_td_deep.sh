#!/bin/bash
# PushT deeper TD sweep (second-stage autonomous driver, coexists with run_all_pusht.sh).
#   A: train tau x nstep grid around/below the stage-1 winner (n5 was a grid edge)
#   B: longer-training (20k steps) + higher-capacity variants in the promising region
#   Evals wait until the main driver finishes (no GPU contention):
#     all configs on h50 s42 -> top 8 on h25 s42 -> winner confirmed on all 6 cells
#     + blend cost mode (latent + TD standardized sum) for the winner
#   If the deep winner beats the stage-1 teacher (136 = 76+60), retrain LIP against it.
# Idempotent: metrics skipped if .pt exists, evals skipped if named in summary.csv.
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
CACHE1=/workspace/caches/pusht_official_fs1.pt
CACHE5=/workspace/caches/pusht_official_fs5.pt
H5=/workspace/swm_home/datasets/pusht_expert_train.h5
WM=lewm_pusht_official
PY=python3
EVAL_TIMEOUT=14400
OLD_TD_SCORE=136

log() { echo "[$(date +%H:%M:%S)] [deep] $*" | tee -a "$LOGS/driver.log"; }
die() { log "FATAL: $*"; exit 1; }

run_eval() { # name gpu seed offset budget, then extra hydra args
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5; shift 5
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached ($(grep "^${name}," "$RES/summary.csv" | tail -1 | cut -d, -f2))"
    return 0
  fi
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu timeout "$EVAL_TIMEOUT" $PY "$PLAN/eval_wm.py" \
    --config-name pusht_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" \
    "+video_dir=$RES/videos_${name}" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "${sr:-}" ]; then
    log "eval ${name}: FAILED (no success_rate; see $LOGS/eval_${name}.log)"
    echo "${name},FAIL" >> "$RES/summary.csv"
    return 1
  fi
  echo "${name},${sr}" >> "$RES/summary.csv"
  log "eval ${name}: ${sr}"
}

get_sr() { grep "^${1}," "$RES/summary.csv" | tail -1 | cut -d, -f2; }

eval_all_cells() { # prefix, extra args...
  local prefix=$1; shift
  run_eval "${prefix}_h25_s42" 0 42 25 50 "$@" &
  run_eval "${prefix}_h25_s43" 1 43 25 50 "$@" &
  run_eval "${prefix}_h25_s44" 2 44 25 50 "$@" &
  run_eval "${prefix}_h50_s42" 3 42 50 100 "$@" &
  wait
  run_eval "${prefix}_h50_s43" 0 43 50 100 "$@" &
  run_eval "${prefix}_h50_s44" 1 44 50 100 "$@" &
  wait
}

td_train() { # tag tau nstep steps hidden depth gpu
  local tag=$1 tau=$2 nstep=$3 steps=$4 hidden=$5 depth=$6 gpu=$7
  local out="$MET/td2_${tag}.pt"
  [ -f "$out" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_metric.py" --cache "$CACHE1" \
    --learner td --head quasimetric --expectile "$tau" --n-step "$nstep" \
    --steps "$steps" --hidden-dim "$hidden" --depth "$depth" \
    --out "$out" > "$LOGS/td2_${tag}.log" 2>&1 \
    || log "TD2 train ${tag} FAILED"
}

# ---------------------------------------------------------------- training
# GPUs 0,1,3 (main driver's K8 LIP training holds GPU 2 for now)
log "stage A+B: training 37 deep-sweep configs"
TAGS=()
for tau in 0.01 0.03 0.05 0.1 0.3; do
  for n in 1 2 3 10 15; do
    TAGS+=("e${tau}_n${n}:${tau}:${n}:6000:256:2")
  done
done
for tau in 0.03 0.1; do
  for n in 2 5 10; do
    TAGS+=("e${tau}_n${n}_st20k:${tau}:${n}:20000:256:2")
    TAGS+=("e${tau}_n${n}_big:${tau}:${n}:6000:512:3")
  done
done

GPUS=(0 1 3)
i=0
for spec in "${TAGS[@]}"; do
  IFS=: read -r tag tau n steps hidden depth <<< "$spec"
  td_train "$tag" "$tau" "$n" "$steps" "$hidden" "$depth" "${GPUS[$((i % 3))]}" &
  i=$((i + 1))
  if [ $((i % 3)) -eq 0 ]; then wait; fi
done
wait
log "deep sweep trained ($(ls $MET/td2_*.pt 2>/dev/null | wc -l) checkpoints)"

# ---------------------------------------------------------------- wait for main driver
log "waiting for main driver (run_all_pusht.sh) to finish before evals"
while pgrep -f "run_all_push[t]" >/dev/null; do sleep 60; done
log "main driver done; starting deep-sweep evals"

# ---------------------------------------------------------------- eval stage 1: h50 s42
gpu=0
for spec in "${TAGS[@]}"; do
  IFS=: read -r tag _ <<< "$spec"
  m="$MET/td2_${tag}.pt"
  [ -f "$m" ] || continue
  run_eval "td2cem_${tag}_h50_s42" "$gpu" 42 50 100 "+metric=$m" &
  gpu=$(( (gpu + 1) % 4 ))
  if [ "$gpu" -eq 0 ]; then wait; fi
done
wait

# ---------------------------------------------------------------- eval stage 2: top 8 on h25 s42
top=$(for spec in "${TAGS[@]}"; do
  IFS=: read -r tag _ <<< "$spec"
  sr=$(get_sr "td2cem_${tag}_h50_s42")
  [ -n "$sr" ] && [ "$sr" != "FAIL" ] && echo "$sr $tag"
done | sort -rn | head -8 | awk "{print \$2}")
[ -n "$top" ] || die "no valid h50 scores in deep sweep"
log "top-8 by h50 s42: $(echo $top | tr '\n' ' ')"

gpu=0
for tag in $top; do
  run_eval "td2cem_${tag}_h25_s42" "$gpu" 42 25 50 "+metric=$MET/td2_${tag}.pt" &
  gpu=$(( (gpu + 1) % 4 ))
  if [ "$gpu" -eq 0 ]; then wait; fi
done
wait

# ---------------------------------------------------------------- pick winner (h25+h50 s42)
best=""; best_score=-1
for tag in $top; do
  a=$(get_sr "td2cem_${tag}_h25_s42"); b=$(get_sr "td2cem_${tag}_h50_s42")
  { [ -z "$a" ] || [ "$a" = "FAIL" ] || [ -z "$b" ] || [ "$b" = "FAIL" ]; } && continue
  s=$(awk "BEGIN{print $a + $b}")
  if awk "BEGIN{exit !($s > $best_score)}"; then best_score=$s; best=$tag; fi
done
[ -n "$best" ] || die "no deep config produced two valid selection scores"
TD2_WIN="$MET/td2_${best}.pt"
log "deep TD winner: td2_${best} (h25+h50 s42 = ${best_score}; stage-1 winner = ${OLD_TD_SCORE})"

log "confirm deep TD winner on all cells (replacement + blend cost modes)"
eval_all_cells "td2win_${best}" "+metric=$TD2_WIN"
eval_all_cells "td2blend_${best}" "+metric=$TD2_WIN" "+cost_mode=blend"

# ---------------------------------------------------------------- optional LIP retrain
if awk "BEGIN{exit !($best_score > $OLD_TD_SCORE + 4)}"; then
  log "deep winner beats stage-1 teacher; retraining LIP against td2_${best}"
  lip_train2() { # iters lr steps gpu
    local iters=$1 lr=$2 steps=$3 gpu=$4
    local out="$ACT/lip2_k${iters}_lr${lr}_st${steps}.pt"
    [ -f "$out" ] && return 0
    CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip.py" --cache "$CACHE5" --h5 "$H5" \
      --wm $WM --value "$TD2_WIN" --horizon 5 --iters "$iters" \
      --steps "$steps" --lr "$lr" --out "$out" \
      > "$LOGS/lip2_k${iters}_lr${lr}_st${steps}.log" 2>&1 \
      || log "LIP2 train k${iters} FAILED"
  }
  lip_train2 4 3e-4 4000 0 &
  lip_train2 8 3e-4 8000 1 &
  wait
  log "LIP2 trained"
  for tag2 in k4_lr3e-4_st4000 k8_lr3e-4_st8000; do
    a="$ACT/lip2_${tag2}.pt"
    [ -f "$a" ] || continue
    run_eval "lip2_${tag2}_h25_s42" 0 42 25 50 solver=lip "solver.actor_path=$a" &
    run_eval "lip2_${tag2}_h50_s42" 1 42 50 100 solver=lip "solver.actor_path=$a" &
    wait
  done
  L_WIN=""; l_best=-1
  for tag2 in k4_lr3e-4_st4000 k8_lr3e-4_st8000; do
    a=$(get_sr "lip2_${tag2}_h25_s42"); b=$(get_sr "lip2_${tag2}_h50_s42")
    { [ -z "$a" ] || [ "$a" = "FAIL" ] || [ -z "$b" ] || [ "$b" = "FAIL" ]; } && continue
    s=$(awk "BEGIN{print $a + $b}")
    if awk "BEGIN{exit !($s > $l_best)}"; then l_best=$s; L_WIN=$tag2; fi
  done
  if [ -n "$L_WIN" ]; then
    log "LIP2 winner: lip2_${L_WIN} (${l_best}); confirming on all cells"
    eval_all_cells "lip2win_${L_WIN}" solver=lip "solver.actor_path=$ACT/lip2_${L_WIN}.pt"
  fi
else
  log "deep winner does not beat stage-1 teacher by >4; keeping stage-1 LIP"
fi

log "DEEP ALL DONE"
sort "$RES/summary.csv" | grep -E "^td2|^lip2"
