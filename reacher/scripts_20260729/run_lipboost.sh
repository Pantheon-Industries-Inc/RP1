#!/bin/bash
# LIP BOOST sweep, h25 only, on the 4xH100 pod (queued behind the h100 cards).
# Goal: put LIP SIGNIFICANTLY above Latent+CEM-window (89.7). Baseline for every
# arm below: amax 1.8 / iters 8 / window3 critic / pad-context = 88.3 (n=6).
#
# Each arm targets a SPECIFIC measured defect rather than random hyper jitter:
#
#  A distilL2   --init-value l2window3.pt
#      At h25 the UNLEARNED window-L2 is the best cost measured (89.7) while the
#      learned window quasimetric is 86.7 -- yet LIP has only ever distilled the
#      latter. LIP tracks its critic, so distil the better cost. (Needs the
#      param-free-critic patch; TD/EMA disabled, actor still backprops through V.)
#
#  B delta6     --max-delta 6 --td-max-delta 6
#      h25's goal is 25 primitive steps = 5 action blocks, but training samples
#      goals up to 12 blocks away. Match the deployed goal distance.
#
#  C replay03   --replay-prob 0.3
#      Trains partly on the actor's OWN previously-imagined windows, i.e. the
#      distribution a deployed replan actually queries, beyond what pad-context
#      fixes (which only collapses the context, not its content).
#
#  D expand05   --expand-weight 0.5
#      Value expansion on planner rollouts: makes the critic accurate on the
#      states the planner visits. Probe D found the value ranks correctly on data
#      but the planner's own terminal states are where it matters.
#
#  E bc01       --bc-weight 0.1
#      LIP is documented to UNDER-ACTUATE: plan rms 0.32-0.36 vs the data's 0.999,
#      and success ranks WITH plan magnitude. A trust region toward data actions
#      attacks that directly (deploy-time plan_scale was already refuted).
#
#  F steps16k   --steps 16000
#      The ~9-point training-seed spread may be under-training; double the budget.
#
# 6 arms x 2 training seeds = 12 trainings (3 per GPU, one wave), then 6-seed
# h25 evals. Plain deploy only (no restarts, per user).
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
W3=/workspace/metrics/window3_lejepa.pt
L2W=/workspace/metrics/l2window3.pt
SUM=/workspace/results/summary_lipboost_lejepa.csv; touch "$SUM"
L=/workspace/logs/lipboost; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][boost] $*"; }

log "waiting for the h100 cards to free the GPUs"
while pgrep -f "run_h100w3" >/dev/null; do sleep 60; done
log "GPUs free -- LIP boost sweep starts (h25 only, plain deploy)"

declare -A ARMS=(
  [distilL2]="--init-value $L2W"
  [delta6]="--init-value $W3 --max-delta 6 --td-max-delta 6"
  [replay03]="--init-value $W3 --replay-prob 0.3"
  [expand05]="--init-value $W3 --expand-weight 0.5"
  [bc01]="--init-value $W3 --bc-weight 0.1"
  [steps16k]="--init-value $W3 --steps 16000"
)
BASE="--cache /workspace/caches/canon_lejepa_fs5.pt \
 --cache-td /workspace/caches/canon_lejepa_fs1.pt \
 --h5 $CANON --wm $WM --pad-context \
 --arch v4 --amax 1.8 --iters 8 --horizon 5 --max-delta 12 \
 --steps 8000 --batch 128 --n-step 50 \
 --expectile 0.1 --expectile-final 0.03 \
 --critic-lr 1e-3 --critic-lr-final 1e-4 \
 --actor-lr 3e-4 --actor-lr-final 3e-5"

tr(){ local tag=$1 seed=$2 gpu=$3
  local A=/workspace/actors/lip4_bst_${tag}_s${seed}.pt
  [ -f "$A" ] && { log "train ${tag} s${seed}: exists"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$PLAN/train_lip_ac.py" \
    $BASE ${ARMS[$tag]} --seed "$seed" \
    --out "$A" --out-value "/workspace/metrics/lip4_bst_${tag}_s${seed}_value.pt" \
    > "$L/train_${tag}_s${seed}.log" 2>&1 \
    && log "train ${tag} s${seed} DONE" || log "train ${tag} s${seed} FAILED"
}
ev(){ local gpu=$1 nm=$2 seed=$3 actor=$4
  grep -q "^${nm}," "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=lip solver.actor_path="$actor" solver.rollout_compat=false \
    solver.batch_size=10 output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local ever
  ever=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  echo "${nm},latched=${ever:-FAIL}" >> "$SUM"
  log "  ${nm}: latched ${ever:-FAIL}"
}
mean6(){ local pre=$1; local t=0 n=0 e
  for s in 42 43 44 45 46 47; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "latched=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

# ---- trainings: 12 jobs, 3 per GPU
i=0
for tag in distilL2 delta6 replay03 expand05 bc01 steps16k; do
  for seed in 0 1; do
    tr "$tag" "$seed" $((i % 4)) &
    i=$((i + 1))
  done
done
wait
log "boost trainings drained"

# ---- evals: one arm per GPU stream, both seeds
runarm(){ local tag=$1 gpu=$2
  for seed in 0 1; do
    A=/workspace/actors/lip4_bst_${tag}_s${seed}.pt
    [ -f "$A" ] || { log "eval ${tag} s${seed}: no actor"; continue; }
    for s in 42 43 44 45 46 47; do ev $gpu "bst_${tag}_s${seed}_e${s}" $s "$A"; done
    log "CARD6 ${tag} s${seed}: latched h25 $(mean6 "bst_${tag}_s${seed}_e")"
  done
  local m0 m1
  m0=$(mean6 "bst_${tag}_s0_e"); m1=$(mean6 "bst_${tag}_s1_e")
  log "ARM ${tag}: seed means ${m0} / ${m1} [ref amax1.8 88.3 | TARGET > Latent+CEM 89.7]"
}
runarm distilL2 0 & runarm delta6 1 & runarm replay03 2 & runarm expand05 3 &
wait
runarm bc01 0 & runarm steps16k 1 &
wait
log "LIPBOOST_DONE"
