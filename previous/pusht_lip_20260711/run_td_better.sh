#!/bin/bash
# Better-TD round 2: ensemble pessimism, input-noise smoothing, gamma=0.99,
# full-data 20k-step training — all on the history architecture (n5), plus a
# single-frame ensemble control. Evals gate on all other drivers finishing.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1 MUJOCO_GL=osmesa
PLAN=/workspace/code/stable-worldmodel/scripts/plan
TRM=/workspace/code/stable-worldmodel/scripts/trm
RES=/workspace/results; LOGS=/workspace/logs; MET=/workspace/metrics; ACT=/workspace/actors
CACHE1=/workspace/caches/pusht_official_fs1.pt
CACHEFULL=/workspace/caches/pusht_official_fs1_full.pt
CACHE5=/workspace/caches/pusht_official_fs5.pt
H5=/workspace/swm_home/datasets/pusht_expert_train.h5
log() { echo "[$(date +%H:%M:%S)] [better] $*" | tee -a "$LOGS/driver.log"; }
run_eval() {
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5; shift 5
  grep -q "^${name}," "$RES/summary.csv" && { log "eval ${name}: cached"; return 0; }
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu timeout 14400 python3 "$PLAN/eval_wm.py" --config-name pusht_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" "+video_dir=$RES/videos_${name}" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$RES/summary.csv"; return 1; }
  echo "${name},${sr}" >> "$RES/summary.csv"; log "eval ${name}: ${sr}"
}
get_sr() { grep "^${1}," "$RES/summary.csv" | tail -1 | cut -d, -f2; }
eval_all_cells() {
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

log "training round-2 configs"
[ -f "$MET/hist_e0.03_n5_s1.pt" ] || CUDA_VISIBLE_DEVICES=1 python3 /workspace/code/train_metric_hist.py \
  --cache "$CACHE1" --expectile 0.03 --n-step 5 --steps 6000 --seed 1 \
  --out "$MET/hist_e0.03_n5_s1.pt" > "$LOGS/hist_s1.log" 2>&1 &
[ -f "$MET/hist_e0.03_n5_s2.pt" ] || CUDA_VISIBLE_DEVICES=2 python3 /workspace/code/train_metric_hist.py \
  --cache "$CACHE1" --expectile 0.03 --n-step 5 --steps 6000 --seed 2 \
  --out "$MET/hist_e0.03_n5_s2.pt" > "$LOGS/hist_s2.log" 2>&1 &
[ -f "$MET/td_e0.03_n5_s1.pt" ] || CUDA_VISIBLE_DEVICES=3 python3 "$PLAN/train_metric.py" \
  --cache "$CACHE1" --learner td --head quasimetric --expectile 0.03 --n-step 5 \
  --steps 6000 --seed 1 --out "$MET/td_e0.03_n5_s1.pt" > "$LOGS/td_s1.log" 2>&1 &
wait
[ -f "$MET/td_e0.03_n5_s2.pt" ] || CUDA_VISIBLE_DEVICES=1 python3 "$PLAN/train_metric.py" \
  --cache "$CACHE1" --learner td --head quasimetric --expectile 0.03 --n-step 5 \
  --steps 6000 --seed 2 --out "$MET/td_e0.03_n5_s2.pt" > "$LOGS/td_s2.log" 2>&1 &
[ -f "$MET/hist_e0.03_n5_noise0.1.pt" ] || CUDA_VISIBLE_DEVICES=2 python3 /workspace/code/train_metric_hist.py \
  --cache "$CACHE1" --expectile 0.03 --n-step 5 --steps 6000 --input-noise 0.1 \
  --out "$MET/hist_e0.03_n5_noise0.1.pt" > "$LOGS/hist_noise.log" 2>&1 &
[ -f "$MET/hist_e0.03_n5_g0.99.pt" ] || CUDA_VISIBLE_DEVICES=3 python3 /workspace/code/train_metric_hist.py \
  --cache "$CACHE1" --expectile 0.03 --n-step 5 --steps 6000 --gamma 0.99 \
  --out "$MET/hist_e0.03_n5_g0.99.pt" > "$LOGS/hist_g99.log" 2>&1 &
wait
log "seed/noise/gamma configs trained; building full cache"
if [ ! -f "$CACHEFULL" ]; then
  CUDA_VISIBLE_DEVICES=0 python3 "$TRM/cache_latents.py" \
    --wm lewm_pusht_official --dataset pusht_expert_train.h5 --out "$CACHEFULL" \
    --state-key state --batch-size 512 > "$LOGS/cache_full.log" 2>&1 || log "full cache FAILED"
fi
if [ -f "$CACHEFULL" ]; then
  [ -f "$MET/hist_e0.03_n5_full20k.pt" ] || CUDA_VISIBLE_DEVICES=0 python3 /workspace/code/train_metric_hist.py \
    --cache "$CACHEFULL" --expectile 0.03 --n-step 5 --steps 20000 \
    --out "$MET/hist_e0.03_n5_full20k.pt" > "$LOGS/hist_full20k.log" 2>&1 || log "full20k FAILED"
fi
log "round-2 training done; waiting for other drivers"
while pgrep -f "run_td_dee[p]|run_td_his[t]|run_lip_restart[s]" >/dev/null; do sleep 120; done
log "GPUs free; round-2 selection evals"

ENS_HIST="$MET/hist_e0.03_n5.pt,$MET/hist_e0.03_n5_s1.pt,$MET/hist_e0.03_n5_s2.pt"
ENS_TD="$MET/td_e0.03_n5.pt,$MET/td_e0.03_n5_s1.pt,$MET/td_e0.03_n5_s2.pt"
spec_of() {
  case "$1" in
    ens_hist) echo "$ENS_HIST" ;;
    ens_td) echo "$ENS_TD" ;;
    hist_noise0.1) echo "$MET/hist_e0.03_n5_noise0.1.pt" ;;
    hist_g0.99) echo "$MET/hist_e0.03_n5_g0.99.pt" ;;
    hist_full20k) echo "$MET/hist_e0.03_n5_full20k.pt" ;;
  esac
}
TAGS="ens_hist ens_td hist_noise0.1 hist_g0.99 hist_full20k"
for off in "25 50 h25" "50 100 h50"; do
  set -- $off; o=$1; b=$2; hn=$3
  gpu=0
  for tag in $TAGS; do
    mm=$(spec_of "$tag")
    case "$mm" in
      *,*) : ;;
      *) [ -f "$mm" ] || continue ;;
    esac
    run_eval "better_${tag}_${hn}_s42" "$gpu" 42 "$o" "$b" "+metric='${mm}'" &
    gpu=$(( (gpu + 1) % 4 ))
    if [ "$gpu" -eq 0 ]; then wait; fi
  done
  wait
done
best=""; best_score=-1
for tag in $TAGS; do
  a=$(get_sr "better_${tag}_h25_s42"); b=$(get_sr "better_${tag}_h50_s42")
  { [ -z "$a" ] || [ "$a" = "FAIL" ] || [ -z "$b" ] || [ "$b" = "FAIL" ]; } && continue
  s=$(awk "BEGIN{print $a + $b}")
  if awk "BEGIN{exit !($s > $best_score)}"; then best_score=$s; best=$tag; fi
done
[ -n "$best" ] || { log "FATAL: no round-2 scores"; exit 1; }
log "round-2 winner: ${best} (h25+h50 s42 = ${best_score})"
BEST_SPEC=$(spec_of "$best")
eval_all_cells "betterwin_${best}" "+metric='${BEST_SPEC}'"
eval_all_cells "betterblend_${best}" "+metric='${BEST_SPEC}'" "+cost_mode=blend"
if awk "BEGIN{exit !($best_score > 140)}"; then
  TEACH="${BEST_SPEC%%,*}"
  log "round-2 teacher beats stage-1 (136); LIP retrain against ${TEACH}"
  if [ ! -f "$ACT/lip_better_k8.pt" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 "$PLAN/train_lip.py" --cache "$CACHE5" --h5 "$H5" \
      --wm lewm_pusht_official --value "$TEACH" --horizon 5 --iters 8 \
      --steps 8000 --lr 3e-4 --out "$ACT/lip_better_k8.pt" \
      > "$LOGS/lip_better_k8.log" 2>&1 || log "lip_better train FAILED"
  fi
  [ -f "$ACT/lip_better_k8.pt" ] && eval_all_cells "lipbetterR8" solver=lip \
    "solver.actor_path=$ACT/lip_better_k8.pt" solver.restarts=8 solver.restart_noise=0.5
fi
log "BETTER ALL DONE"
