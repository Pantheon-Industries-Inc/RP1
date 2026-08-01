#!/bin/bash
# BATCH-SIZE sweep at the winning 1000-step recipe (user, 2026-08-01).
#
# Centre = the grid1k winner: 1000 steps, lr 1e-3, uniform lambda, amax 1.8,
# mean-weight 0.1 -> held 48.9 pooled (bar: Latent+CEM-window 44.7).
# Batch was never swept; it is the last untouched canonical LIPv4 knob.
#
# Two-sided prior, which is why it is worth measuring rather than guessing:
#   * LARGER batch = more effective optimisation per step, and this recipe exists
#     precisely because over-optimisation destroys deployment
#     (corr(E_final, HELD) = +0.583) -- so bigger could be WORSE at fixed steps;
#   * LARGER batch = less gradient noise, and part of what the actor exploits is
#     noise in the value gradient -- so bigger could be BETTER.
#
# batch {32, 64, 128, 256, 512} x training seeds {0,1,2} = 15 trainings.
# Runs on GPUs 0-2 while the Dyna collection occupies 3-5.
# Screen on SELECTION seeds {50,51}; card the top 2 on REPORTING seeds {42..47}.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
W3=/workspace/metrics/window3_lejepa.pt
SUM=/workspace/results/summary_batch_lejepa.csv; touch "$SUM"
L=/workspace/logs/batch; mkdir -p "$L"
GPUS="0 1 2"
SEL="50 51"
REP="42 43 44 45 46 47"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][batch] $*"; }

BASE="--cache /workspace/caches/canon_lejepa_fs5.pt \
 --cache-td /workspace/caches/canon_lejepa_fs1.pt \
 --h5 /workspace/reacher_slim.h5 --wm $WM --pad-context --init-value $W3 \
 --arch v4 --iters 8 --horizon 5 --max-delta 12 --n-step 50 \
 --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
 --lambda-schedule uniform --lambda-base 0.5 --mean-weight 0.1 \
 --amax 1.8 --actor-lr 1e-3 --actor-lr-final 1e-4 --steps 1000"

BATCHES="32 64 128 256 512"

train_one(){ local b=$1 seed=$2 gpu=$3
  local A=/workspace/actors/lip4_b${b}_s${seed}.pt
  [ -f "$A" ] && { log "train b${b} s${seed}: exists"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 21600 python3 "$PLAN/train_lip_ac.py" \
    $BASE --batch "$b" --seed "$seed" \
    --out "$A" --out-value "/workspace/metrics/lip4_b${b}_s${seed}_value.pt" \
    > "$L/train_b${b}_s${seed}.log" 2>&1 \
    && log "train b${b} s${seed} DONE" || log "train b${b} s${seed} FAILED"
}
ev(){ local gpu=$1 nm=$2 seed=$3 actor=$4
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=lip solver.actor_path="$actor" solver.rollout_compat=false \
    solver.batch_size=10 output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
meanof(){ local pre=$1 key=$2; shift 2; local t=0 n=0 e
  for s in "$@"; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

log "15 trainings (5 batch sizes x 3 seeds) on GPUs ${GPUS}"
i=0
for b in $BATCHES; do for seed in 0 1 2; do
  gpu=$(echo $GPUS | cut -d" " -f$(( (i % 3) + 1 )))
  train_one "$b" "$seed" "$gpu" &
  i=$((i + 1))
  [ $((i % 9)) -eq 0 ] && wait
done; done
wait
log "trainings drained"

log "screening on selection seeds {${SEL}}"
i=0
for b in $BATCHES; do
  gpu=$(echo $GPUS | cut -d" " -f$(( (i % 3) + 1 )))
  ( for seed in 0 1 2; do
      A=/workspace/actors/lip4_b${b}_s${seed}.pt; [ -f "$A" ] || continue
      for s in $SEL; do ev $gpu "bt_b${b}_s${seed}_sel${s}" $s "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % 3)) -eq 0 ] && wait
done
wait

: > /workspace/results/batch_screen.txt
for b in $BATCHES; do
  tot=0; k=0
  for seed in 0 1 2; do
    v=$(meanof "bt_b${b}_s${seed}_sel" held $SEL); [ "$v" = "0.0" ] && continue
    tot=$(awk "BEGIN{print $tot+$v}"); k=$((k+1))
  done
  [ "$k" -eq 0 ] && continue
  m=$(awk "BEGIN{printf \"%.1f\", $tot/$k}")
  echo "$m $b" >> /workspace/results/batch_screen.txt
  log "  batch ${b}: selection held ${m} (n=${k} seeds)"
done
log "--- screen ranking (batch 128 is the 48.9 incumbent) ---"
sort -rn /workspace/results/batch_screen.txt | while read -r m b; do log "  ${m}  batch ${b}"; done

log "carding the top 2 on reporting seeds {${REP}}"
i=0
sort -rn /workspace/results/batch_screen.txt | head -2 | awk '{print $2}' | while read -r b; do
  ta=0; tb=0; k=0
  gpu=$(echo $GPUS | cut -d" " -f$(( (i % 3) + 1 )))
  for seed in 0 1 2; do
    A=/workspace/actors/lip4_b${b}_s${seed}.pt; [ -f "$A" ] || continue
    for s in $REP; do ev $gpu "bt_b${b}_s${seed}_e${s}" $s "$A"; done
    v=$(meanof "bt_b${b}_s${seed}_e" held $REP); w=$(meanof "bt_b${b}_s${seed}_e" held10 $REP)
    log "  CARD batch ${b} s${seed}: HELD ${v} | @0.1 ${w}"
    ta=$(awk "BEGIN{print $ta+$v}"); tb=$(awk "BEGIN{print $tb+$w}"); k=$((k+1))
  done
  [ "$k" -gt 0 ] && log "POOLED batch ${b}: HELD $(awk "BEGIN{printf \"%.1f\", $ta/$k}") | @0.1 $(awk "BEGIN{printf \"%.1f\", $tb/$k}") over ${k} seeds"
  i=$((i + 1))
done
log "BATCH_DONE -- incumbent batch 128 = 48.9 | bar Latent+CEM-window 44.7"
