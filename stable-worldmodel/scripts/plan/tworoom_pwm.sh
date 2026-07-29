#!/bin/bash
# PWM-arm driver: train the reactive policy (train_pwm_ac.py) and evaluate it
# (solver=pwm) on the canonical TwoRoom protocol, matching the campaign's
# conventions exactly (same dataset pin, same seeds, same driver-log format as
# it10_rerun.sh / tworoom_matrix.sh).
#
#   tworoom_pwm.sh train <base> <gpu> [train_seeds...]     # default 0 1 2
#   tworoom_pwm.sh eval  <base> <gpu> [train_seeds...]     # protocol regime
#   tworoom_pwm.sh eval1 <base> <gpu> [train_seeds...]     # receding_horizon=1
#
# eval  = campaign protocol: 5 blocks (25 primitive steps) executed per replan,
#         plan tail imagined through the WM (fill=imagine).
# eval1 = pure reactive regime: replan every block, zero WM rollouts at deploy.
#
# Assets assumed (preflight checks them):
#   /workspace/caches/tworoom_canon_<base><RTAG>_fs{1,5}.pt
#   /workspace/metrics/td_canon_<base><RTAG>_e0.1_n50_s<seed>.pt
#   world model registered as <base>_tworoom under $STABLEWM_HOME
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel
export HF_HOME=/root/hf TQDM_DISABLE=1 OMP_NUM_THREADS=14 MKL_NUM_THREADS=14
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HDF5_PLUGIN_PATH=$(python3 -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')
cd /workspace/code/stable-worldmodel

STAGE=$1; BASE=$2; GPU=$3; shift 3
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
CANON=/workspace/datasets_canon/tworoom/tworoom.h5
DRIVER=/workspace/logs/pwm_driver.log
CK=${BASE}_tworoom
EXTRA=(); [ "$BASE" = dinowm ] && EXTRA=(dataset.keys_to_cache='[action]')
case $BASE in
  lejepa|pldm) RTAG=""; BATCH=128 ;;
  dinowm)      RTAG="_r200000"; BATCH=16 ;;   # 77,224-d latents: memory-bound
  *) echo "unknown base $BASE" >&2; exit 1 ;;
esac
FS5=/workspace/caches/tworoom_canon_${BASE}${RTAG}_fs5.pt
FS1=/workspace/caches/tworoom_canon_${BASE}${RTAG}_fs1.pt

preflight(){
  local ok=1
  for f in "$FS5" "$FS1"; do [ -f "$f" ] || { echo "MISSING cache $f" >&2; ok=0; }; done
  for s in "${SEEDS[@]}"; do
    [ -f /workspace/metrics/td_canon_${BASE}${RTAG}_e0.1_n50_s${s}.pt ] || {
      echo "MISSING td warm-start seed $s" >&2; ok=0; }
  done
  [ $ok -eq 1 ] || exit 3
}

if [ "$STAGE" = train ]; then
  preflight
  for s in "${SEEDS[@]}"; do
    OUT=/workspace/actors/pwm_${BASE}${RTAG}_s${s}.pt
    OUTV=/workspace/metrics/pwm_${BASE}${RTAG}_s${s}_value.pt
    [ -f "$OUT" ] && { echo "PWM-TRAIN $BASE s$s: exists, skip" >> $DRIVER; continue; }
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/plan/train_pwm_ac.py \
      --cache "$FS5" --cache-td "$FS1" --wm $CK \
      --init-value /workspace/metrics/td_canon_${BASE}${RTAG}_e0.1_n50_s${s}.pt \
      --out "$OUT" --out-value "$OUTV" \
      --batch $BATCH --seed $s \
      > /workspace/logs/pwm_train_${BASE}_s${s}.log 2>&1 \
      && echo "PWM-TRAIN $BASE s$s: OK" >> $DRIVER \
      || echo "PWM-TRAIN $BASE s$s: FAIL" >> $DRIVER
  done
  echo "PWM-TRAIN_DONE $BASE gpu$GPU" >> $DRIVER
  exit 0
fi

# ---- eval stages ------------------------------------------------------------
TAG=proto; RH=()
[ "$STAGE" = eval1 ] && { TAG=rh1; RH=(plan_config.receding_horizon=1); }
[ "$STAGE" = eval ] || [ "$STAGE" = eval1 ] || { echo "unknown stage $STAGE" >&2; exit 2; }

for s in "${SEEDS[@]}"; do
  ACT=/workspace/actors/pwm_${BASE}${RTAG}_s${s}.pt
  [ -f "$ACT" ] || { echo "PWM-EVAL $BASE t$s: actor missing" >> $DRIVER; continue; }
  for sd in 42 43 44; do for off in 25 50; do
    lg=/workspace/logs/pwm_${TAG}_${BASE}_t${s}_h${off}_sd${sd}.log
    grep -q success_rate $lg 2>/dev/null && continue
    CUDA_VISIBLE_DEVICES=$GPU timeout 14400 python3 scripts/plan/eval_wm.py --config-name tworoom \
      policy=$CK eval.dataset_name=$CANON dataset.stats=$CANON "${EXTRA[@]}" \
      seed=$sd eval.goal_offset_steps=$off eval.eval_budget=$((off*2)) \
      solver=pwm solver.actor_path="$ACT" solver.batch_size=10 "${RH[@]}" \
      output.filename=pwm_${TAG}_${BASE}_t${s}_h${off}_sd${sd}.txt > $lg 2>&1
    v=$(grep -oE 'success_rate[^0-9]*[0-9.]+' $lg | tail -1 | grep -oE '[0-9.]+$')
    echo "PWM $TAG $BASE t$s h${off} sd${sd}: ${v:-FAIL}" >> $DRIVER
  done; done
done
echo "PWM-EVAL_DONE $TAG $BASE gpu$GPU" >> $DRIVER
