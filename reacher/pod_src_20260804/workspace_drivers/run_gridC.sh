#!/bin/bash
# GRID C -- LIP optimisation informed by the seed-0 post-mortem (2026-07-31).
#
# WHY THIS GRID AND NOT MORE amax/mean-weight. Wave A settled those: across
# 3 training seeds, amax {1.8,2.0,2.2,2.6} spans 33.4-35.3 held and mean-weight
# {0.1,0.3,0.5} spans 33.9-36.2 -- every level inside noise. What DID move
# things was the learning rate (matched-seed 55.0/53.7 at 1e-4 vs 47.3/47.7 at
# 3e-4), and what dominates everything is the training-seed lottery: seed 0
# lands at 4.7-16.3 held while its siblings reach 41.7-57.7.
#
# The post-mortem explains the lottery. Over 20 actors, corr(E_final, HELD) =
# +0.583: driving the imagined cost DOWN makes real performance WORSE. Best
# imagined (0.778) held 11.0; worst imagined (1.921) held 51.3. Seed 0 is not
# broken -- it is the seed that optimises hardest, and the world model cannot
# support it (25-step open-loop error ~0.11 rad vs a 0.05 rad tolerance).
#
# So this grid tests the two levers that attack over-optimisation directly,
# crossed with the LR that is known to matter:
#
#   lambda-schedule  uniform | geom-early | geom-late
#       geom-early weights the FIRST refinement iterations, i.e. stops paying
#       the actor to keep refining into WM error. geom-late is its falsification
#       twin: if the mechanism is real, geom-late should be WORSE than uniform.
#   steps            4000 | 8000
#       early stopping is the crudest version of the same idea, and it makes a
#       sharp prediction: if over-optimisation is the cause, 4000 steps should
#       RESCUE SEED 0 specifically.
#   actor-lr         1e-4 | 3e-4
#
# 3 x 2 x 2 = 12 configs x training seeds {0,1,2} = 36 trainings.
# Setting: h25, HELD@0.05 primary (0.1 rad also recorded), window3 critic,
# pad-context, 1-frame conditioning, plain deploy, amax 2.2, iters 8.
# Bar: Latent+CEM-window 44.7 @0.05 / 84.3 @0.1.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
W3=/workspace/metrics/window3_lejepa.pt
SUM=/workspace/results/summary_gridC_lejepa.csv; touch "$SUM"
L=/workspace/logs/gridC; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][gridC] $*"; }

BASE="--cache /workspace/caches/canon_lejepa_fs5.pt \
 --cache-td /workspace/caches/canon_lejepa_fs1.pt \
 --h5 /workspace/reacher_slim.h5 --wm $WM --pad-context --init-value $W3 \
 --arch v4 --amax 2.2 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
 --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
 --mean-weight 0.1 --lambda-base 0.5"

SCHEDS="uniform geom-early geom-late"
STEPSET="4000 8000"
LRSET="1e-4 3e-4"

tag(){ # sched steps lr  -- pure parameter expansion: no subshell, no `tr`
  local sc="$1" st="$2" lr="$3"
  case "$sc" in uniform) sc=u ;; geom-early) sc=ge ;; geom-late) sc=gl ;; esac
  local lrs="${lr//e-/e}"          # 1e-4 -> 1e4, keeps the arms distinct
  echo "${sc}st$(( st / 1000 ))k${lrs}"
}
lrf(){ case "$1" in 1e-4) echo 1e-5;; 3e-4) echo 3e-5;; *) echo 1e-5;; esac; }

train_cell(){ # sched steps lr seed gpu
  local sc=$1 st=$2 lr=$3 seed=$4 gpu=$5
  local t; t=$(tag "$sc" "$st" "$lr")
  local A=/workspace/actors/lip4_gc_${t}_s${seed}.pt
  [ -f "$A" ] && { log "train ${t} s${seed}: exists"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$PLAN/train_lip_ac.py" \
    $BASE --lambda-schedule "$sc" --steps "$st" \
    --actor-lr "$lr" --actor-lr-final "$(lrf $lr)" --seed "$seed" \
    --out "$A" --out-value "/workspace/metrics/lip4_gc_${t}_s${seed}_value.pt" \
    > "$L/train_${t}_s${seed}.log" 2>&1 \
    && log "train ${t} s${seed} DONE" || log "train ${t} s${seed} FAILED"
}

log "grid C: 12 configs x seeds {0,1,2} = 36 trainings, 3 per GPU"
i=0
for sc in $SCHEDS; do for st in $STEPSET; do for lr in $LRSET; do
  for seed in 0 1 2; do
    train_cell "$sc" "$st" "$lr" "$seed" $((i % 6)) &
    i=$((i + 1))
    [ $((i % 18)) -eq 0 ] && wait
  done
done; done; done
wait
log "grid C trainings drained"

# ---- evals: SERIAL. Six concurrent EGL streams reproduce the SIGABRT
# ("timeout: the monitored command dumped core"); one at a time alongside the
# trainings was verified stable.
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
  t10=$(grep -oE "0.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
mean(){ local pre=$1 key=$2; local t=0 n=0 e
  for s in 42 43 44 45 46 47; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

log "grid C evals (serial)"
g=0
for sc in $SCHEDS; do for st in $STEPSET; do for lr in $LRSET; do
  t=$(tag "$sc" "$st" "$lr")
  for seed in 0 1 2; do
    A=/workspace/actors/lip4_gc_${t}_s${seed}.pt; [ -f "$A" ] || continue
    for s in 42 43 44 45 46 47; do
      ev $(( (g % 3) + 3 )) "gc_${t}_s${seed}_e${s}" $s "$A"; g=$((g+1))
    done
    log "  CARD ${t} s${seed}: HELD $(mean "gc_${t}_s${seed}_e" held) | @0.1 $(mean "gc_${t}_s${seed}_e" held10)"
  done
  a=0; b=0; k=0
  for seed in 0 1 2; do
    v=$(mean "gc_${t}_s${seed}_e" held); [ "$v" = "0.0" ] && continue
    a=$(awk "BEGIN{print $a+$v}"); b=$(awk "BEGIN{print $b+$(mean "gc_${t}_s${seed}_e" held10)}"); k=$((k+1))
  done
  [ "$k" -gt 0 ] && log "POOLED ${t} (${sc}, ${st} steps, lr ${lr}): HELD $(awk "BEGIN{printf \"%.1f\", $a/$k}") | @0.1 $(awk "BEGIN{printf \"%.1f\", $b/$k}") over ${k} seeds"
done; done; done
log "GRIDC_DONE -- bar: Latent+CEM-window 44.7 @0.05 / 84.3 @0.1"
