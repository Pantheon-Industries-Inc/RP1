#!/bin/bash
# ONE RECIPE FOR ALL SEEDS: fixed 1000-step training, no per-actor selection.
#
# Why: per-checkpoint early stopping (+10.4 pooled) is not a single recipe --
# it picks a different stopping point per seed (1000/2000/5000/final/7000), so
# the "method" silently includes a selection procedure. A defensible recipe has
# to be one number that everybody uses.
#
# Reading the selection-seed curves, the mean held@0.05 over all 9 actors at each
# snapshot is: 1000->49.2, 2000->43.9, 3000->44.9, 4000->43.2, 5000->44.4,
# 6000->41.3, 7000->42.9, 8000->43.6. Step 1000 is best for every actor
# simultaneously -- the peak for the seeds that collapse, and near-flat for the
# ones that do not. So it is a genuine single hyperparameter, not a compromise.
#
# This evaluates the step-1000 snapshot of all 9 wave-E actors on the REPORTING
# seeds 42-47 (disjoint from the 50/51 draws the curves were built on), so the
# resulting number is honest for a recipe of "train 1000 steps".
#
# NB the curve is still rising as steps decrease, so the true optimum may be
# BELOW 1000; run_finegrain.sh probes 125-1000 to find out.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
SUM=/workspace/results/summary_fixed1k_lejepa.csv; touch "$SUM"
L=/workspace/logs/fixed1k; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][fix1k] $*"; }

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
mean(){ local pre=$1 key=$2; local t=0 n=0 e
  for s in 42 43 44 45 46 47; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

log "evaluating the step-1000 snapshot of all 9 actors on reporting seeds 42-47"
g=0
for tag in base lr1e4 gearly; do
  ta=0; tb=0; k=0
  for seed in 0 1 2; do
    A=/workspace/actors/lip4_es_${tag}_s${seed}.pt.step1000.pt
    [ -f "$A" ] || { log "  MISSING $A"; continue; }
    for s in 42 43 44 45 46 47; do
      ev $(( (g % 4) + 2 )) "f1k_${tag}_s${seed}_e${s}" $s "$A"; g=$((g+1))
    done
    v=$(mean "f1k_${tag}_s${seed}_e" held); w=$(mean "f1k_${tag}_s${seed}_e" held10)
    log "  CARD ${tag} s${seed} @1000 steps: HELD ${v} | @0.1 ${w}"
    ta=$(awk "BEGIN{print $ta+$v}"); tb=$(awk "BEGIN{print $tb+$w}"); k=$((k+1))
  done
  [ "$k" -gt 0 ] && log "POOLED ${tag} @1000: HELD $(awk "BEGIN{printf \"%.1f\", $ta/$k}") | @0.1 $(awk "BEGIN{printf \"%.1f\", $tb/$k}") over ${k} seeds"
done
log "FIXED1K_DONE -- bar: Latent+CEM-window 44.7 @0.05 / 84.3 @0.1"
