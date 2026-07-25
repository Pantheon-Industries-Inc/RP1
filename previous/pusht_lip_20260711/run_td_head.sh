#!/bin/bash
# Head/asymmetry ablation at the deep-winner config (tau=0.01, n=1):
#   quasimetric (incumbent, 154) vs plain asymmetric MLP vs symmetrized MLP,
#   plus QRL-style constraint training (induces true asymmetry offline).
# Trainings run now (GPUs 2,3); evals gate on all other drivers.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1 MUJOCO_GL=osmesa
PLAN=/workspace/code/stable-worldmodel/scripts/plan
RES=/workspace/results; LOGS=/workspace/logs; MET=/workspace/metrics
CACHE1=/workspace/caches/pusht_official_fs1.pt
log() { echo "[$(date +%H:%M:%S)] [head] $*" | tee -a "$LOGS/driver.log"; }
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

log "training head-ablation configs (mlp / mlp-sym / qrl)"
[ -f "$MET/head_mlp.pt" ] || CUDA_VISIBLE_DEVICES=2 python3 "$PLAN/train_metric.py" \
  --cache "$CACHE1" --learner td --head mlp --expectile 0.01 --n-step 1 \
  --steps 6000 --out "$MET/head_mlp.pt" > "$LOGS/head_mlp.log" 2>&1 &
[ -f "$MET/head_mlpsym.pt" ] || CUDA_VISIBLE_DEVICES=3 python3 "$PLAN/train_metric.py" \
  --cache "$CACHE1" --learner td --head mlp --symmetric --expectile 0.01 --n-step 1 \
  --steps 6000 --out "$MET/head_mlpsym.pt" > "$LOGS/head_mlpsym.log" 2>&1 &
wait
[ -f "$MET/head_qrl.pt" ] || CUDA_VISIBLE_DEVICES=2 python3 /workspace/code/train_metric_qrl.py \
  --cache "$CACHE1" --steps 6000 --out "$MET/head_qrl.pt" > "$LOGS/head_qrl.log" 2>&1 \
  || log "qrl train FAILED"
log "head-ablation configs trained; asymmetry re-check on qrl"
[ -f "$MET/head_qrl.pt" ] && python3 /workspace/code/asym_analysis.py "$MET/head_qrl.pt" \
  > "$LOGS/asym_qrl.log" 2>&1 && tail -6 "$LOGS/asym_qrl.log" | tee -a "$LOGS/driver.log"

log "waiting for other drivers before evals"
while pgrep -f "run_td_dee[p]|run_td_his[t]|run_td_bette[r]" >/dev/null; do sleep 120; done
log "GPUs free; head-ablation evals"
gpu=0
for tag in mlp mlpsym qrl; do
  m="$MET/head_${tag}.pt"; [ -f "$m" ] || continue
  run_eval "head_${tag}_h25_s42" "$gpu" 42 25 50 "+metric=$m" &
  gpu=$(( gpu + 1 ))
done
wait
gpu=0
for tag in mlp mlpsym qrl; do
  m="$MET/head_${tag}.pt"; [ -f "$m" ] || continue
  run_eval "head_${tag}_h50_s42" "$gpu" 42 50 100 "+metric=$m" &
  gpu=$(( gpu + 1 ))
done
wait
for tag in mlp mlpsym qrl; do
  a=$(get_sr "head_${tag}_h25_s42"); b=$(get_sr "head_${tag}_h50_s42")
  { [ -z "$a" ] || [ "$a" = "FAIL" ] || [ -z "$b" ] || [ "$b" = "FAIL" ]; } && continue
  s=$(awk "BEGIN{print $a + $b}")
  log "head ${tag}: h25 ${a} h50 ${b} combined ${s} (quasimetric incumbent = 154)"
  if awk "BEGIN{exit !($s > 158)}"; then
    log "head ${tag} beats incumbent; confirming all cells"
    eval_all_cells "headwin_${tag}" "+metric=$MET/head_${tag}.pt"
  fi
done
log "HEAD ALL DONE"
