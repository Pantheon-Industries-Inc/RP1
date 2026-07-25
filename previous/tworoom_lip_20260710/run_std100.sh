#!/bin/bash
# Chase a 100-on-all-12-cells pure-v1 LIP card:
#   1) full card for lv1_t-ed256 (other selection-perfect teacher, never carded)
#   2) retrain winner recipe (teacher td_e0.1_n50, K8 lr3e-4 8k) with seeds 1,2 -> full cards
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel TQDM_DISABLE=1 MUJOCO_GL=osmesa
PLAN=/workspace/code/stable-worldmodel/scripts/plan
LOGS=/workspace/logs; RES=/workspace/results; ACT=/workspace/actors
H5=/workspace/caches/tworoom_play.h5; CACHE5=/workspace/caches/tworoom_lewm_fs5.pt
TD=/workspace/metrics/td_e0.1_n50.pt
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/std100.log"; }

run_eval() { # name gpu seed offset budget surface actor
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5 surface=$6 actor=$7
  grep -q "^${name}," "$RES/summary.csv" && return 0
  local hard=(); [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu python3 "$PLAN/eval_wm.py" --config-name tworoom_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" "+video_dir=$RES/videos_${name}" \
    "${hard[@]}" solver=lip "solver.actor_path=$actor" > "$LOGS/eval_${name}.log" 2>&1
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$RES/summary.csv"; return 1; }
  echo "${name},${sr}" >> "$RES/summary.csv"; log "eval ${name}: ${sr}"
}

card() { # tag gpu actor  -> runs all 12 cells sequentially on one gpu
  local tag=$1 gpu=$2 actor=$3
  for surface in std hard; do
    for seed in 42 43 44; do
      run_eval "card_${tag}_${surface}_h25_s${seed}" "$gpu" "$seed" 25 50 "$surface" "$actor"
      run_eval "card_${tag}_${surface}_h50_s${seed}" "$gpu" "$seed" 50 100 "$surface" "$actor"
    done
  done
  log "card ${tag} complete"
}

# wait for GPU1 to be free (h10 training / hypers script may still hold it)
until ! pgrep -f "train_lip.py" > /dev/null; do sleep 60; done

log "phase 1: ed256 card (GPU0) + seed retrains (GPU1)"
card t-ed256 0 "$ACT/lv1_t-ed256.pt" &
(
  for s in 1 2; do
    out="$ACT/lv1_e01n50_seed${s}.pt"
    [ -f "$out" ] && continue
    CUDA_VISIBLE_DEVICES=1 python3 "$PLAN/train_lip.py" --cache "$CACHE5" --h5 "$H5" \
      --wm lewm_tworoom --value "$TD" --horizon 5 --iters 8 --steps 8000 --lr 3e-4 \
      --seed "$s" --out "$out" > "$LOGS/lv1_e01n50_seed${s}.log" 2>&1 \
      && log "trained seed${s}" || log "seed${s} train FAILED"
  done
) &
wait

log "phase 2: seed cards"
card e01n50-seed1 0 "$ACT/lv1_e01n50_seed1.pt" &
card e01n50-seed2 1 "$ACT/lv1_e01n50_seed2.pt" &
wait

log "STD100 DONE"
grep -E "^card_" "$RES/summary.csv" | sort
