#!/bin/bash
# LIP BOOST wave 2 -- arms aimed at HELD-at-end specifically, from the
# value-vs-planner diagnostic (2026-07-29):
#   * the planner is NOT the bottleneck: at a fixed window3 value, LIP holds 44.2
#     while greedy CEM holds 11.7 -- LIP already extracts +32 over search.
#   * the VALUE barely sees settling: stopped-vs-moving AUC 0.610 (window3) /
#     0.631 (l2window3) / 0.505 (1-frame quasimetric = chance), against 0.886
#     achievable at a 1-step lag that action_block=5 cannot deploy.
#   * and the DATA almost never shows holding: only 0.4% of at-goal states are
#     still at goal 25 steps later in the random-play dataset.
#
# So: express settling where it IS expressible (the action sequence), and try a
# cost that is exact about position (the oracle readout) to bound how much a
# better value can buy.
#
#   actpen001/005/02  --act-penalty {0.01, 0.05, 0.2}   end-at-rest term on A[:,-1]
#   distilOracle      --init-value oracle_lejepa.pt     exact qpos cost (1-frame)
#   actpen005_d6      act-penalty 0.05 + max-delta 6    best-guess combination
#
# 5 arms x 2 seeds = 10 trainings, 3/GPU on 4 GPUs; then 6-seed h25 evals ranked
# by HELD. Queued behind wave 1.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
W3=/workspace/metrics/window3_lejepa.pt
ORC=/workspace/metrics/oracle_lejepa.pt
SUM=/workspace/results/summary_lipboost2_lejepa.csv; touch "$SUM"
L=/workspace/logs/lipboost2; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][boost2] $*"; }

log "waiting for boost wave 1"
while pgrep -f run_lipboost.sh >/dev/null; do sleep 60; done
log "wave 1 clear -- wave 2 starts (HELD-targeted)"

declare -A ARMS=(
  [actpen001]="--init-value $W3 --act-penalty 0.01"
  [actpen005]="--init-value $W3 --act-penalty 0.05"
  [actpen02]="--init-value $W3 --act-penalty 0.2"
  [actpen005_d6]="--init-value $W3 --act-penalty 0.05 --max-delta 6 --td-max-delta 6"
  [distilOracle]="--init-value $ORC"
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
  local A=/workspace/actors/lip4_b2_${tag}_s${seed}.pt
  [ -f "$A" ] && { log "train ${tag} s${seed}: exists"; return 0; }
  [ "$tag" = "distilOracle" ] && [ ! -f "$ORC" ] && { log "SKIP distilOracle: no oracle metric"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$PLAN/train_lip_ac.py" \
    $BASE ${ARMS[$tag]} --seed "$seed" \
    --out "$A" --out-value "/workspace/metrics/lip4_b2_${tag}_s${seed}_value.pt" \
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
  local h l
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  l=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  echo "${nm},held=${h:-FAIL},latched=${l:-FAIL}" >> "$SUM"
  log "  ${nm}: HELD ${h:-FAIL} | latched ${l:-FAIL}"
}
mean6(){ local pre=$1; local t=0 n=0 e
  for s in 42 43 44 45 46 47; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "held=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

i=0
for tag in actpen001 actpen005 actpen02 actpen005_d6 distilOracle; do
  for seed in 0 1; do tr "$tag" "$seed" $((i % 4)) & i=$((i + 1)); done
done
wait
log "wave-2 trainings drained"

runarm(){ local tag=$1 gpu=$2
  for seed in 0 1; do
    A=/workspace/actors/lip4_b2_${tag}_s${seed}.pt
    [ -f "$A" ] || continue
    for s in 42 43 44 45 46 47; do ev $gpu "b2_${tag}_s${seed}_e${s}" $s "$A"; done
    log "CARD6 ${tag} s${seed}: HELD h25 $(mean6 "b2_${tag}_s${seed}_e")"
  done
  log "ARM ${tag}: HELD $(mean6 "b2_${tag}_s0_e") / $(mean6 "b2_${tag}_s1_e") [refs: LIP-w best 54.0, Latent+CEM-w 42.7]"
}
runarm actpen001 0 & runarm actpen005 1 & runarm actpen02 2 & runarm actpen005_d6 3 &
wait
runarm distilOracle 0 &
wait
log "LIPBOOST2_DONE"
