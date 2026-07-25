#!/bin/bash
# Round 4: replan curriculum (DAgger-in-imagination) on the k12 recipe.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home MUJOCO_GL=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PLAN=/workspace/code/stable-worldmodel/scripts/plan
LOGS=/workspace/logs; RES=/workspace/results; MET=/workspace/metrics; ACT=/workspace/actors
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5
WM=/workspace/ckpts/PLDM_OgBench_lewm
log() { echo "[$(date +%H:%M:%S)] PLDM-R4: $*" | tee -a "$LOGS/pipeline_pldm.log"; }
while pgrep -f "run_pldm_r3.sh" > /dev/null; do sleep 60; done
log "round 3 done; round 4 starting"
train() { # name replayprob seed gpu
  [ -f "$ACT/pldm_$1.pt" ] && return 0
  CUDA_VISIBLE_DEVICES=$4 python3 $PLAN/train_lip_ac.py \
    --cache /workspace/caches/cube_pldm_fs5.pt --cache-td /workspace/caches/cube_pldm_fs1.pt \
    --h5 $H5 --wm $WM --init-value $MET/cf_pldm_t003n50.pt \
    --horizon 5 --iters 12 --steps 8000 --n-step 50 --batch 128 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --amax 3.5 \
    --drop-z0 --drop-zg --no-gate --replay-prob $2 --seed $3 \
    --out $ACT/pldm_$1.pt --out-value $MET/pldm_$1_value.pt > $LOGS/tr_$1.log 2>&1
  log "trained $1 rc=$?"
}
train replay05_s0 0.5 0 0 &
train replay05_s1 0.5 1 1 &
train replay03_s0 0.3 0 2 &
train replay03_s1 0.3 1 3 &
wait
run() { # name actor seed gpu
  grep -q "^$1," $RES/summary.csv && return 0
  CUDA_VISIBLE_DEVICES=$4 timeout 14400 python3 $PLAN/eval_wm.py --config-name cube \
    seed=$3 eval.dataset_name=$H5 ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 output.filename=$1.txt \
    policy=$WM solver=lip solver.actor_path=$ACT/$2.pt > $LOGS/ev_$1.log 2>&1
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" $LOGS/ev_$1.log | tail -1 | grep -oE "[0-9.]+$")
  echo "$1,${sr:-FAIL}" >> $RES/summary.csv
  log "eval $1: ${sr:-FAIL}"
}
for arm in replay05_s0 replay05_s1 replay03_s0 replay03_s1; do
  run lip_${arm}_h25_s42 pldm_$arm 42 0 &
  run lip_${arm}_h25_s44 pldm_$arm 44 1 &
  wait
done
log "ROUND 4 COMPLETE"
sort $RES/summary.csv | grep replay | tee -a $LOGS/pipeline_pldm.log
