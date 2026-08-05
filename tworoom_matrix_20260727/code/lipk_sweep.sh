#!/bin/bash
# LIPv4 deploy-budget truncation: K in {4,2} vs the campaign K=8.
# Same protocol as the matrix LIP cells: solver=lip, lip.yaml defaults
# (batch 1, n_steps 0, restarts 1), canonical h5, std surface.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel
export HF_HOME=/root/hf TQDM_DISABLE=1 OMP_NUM_THREADS=14 MKL_NUM_THREADS=14
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
cd /workspace/code/stable-worldmodel
BASE=$1; GPU=$2
CANON=/workspace/datasets_canon/tworoom/tworoom.h5
for K in 4 2; do for s in 0 1 2; do
  A=/workspace/actors/lipk/trm_canon_${BASE}_v4_s${s}_K${K}.pt
  for sd in 42 43 44; do for off in 25 50; do
    lg=/workspace/logs/lipk_${BASE}_K${K}_s${s}_h${off}_sd${sd}.log
    grep -q success_rate $lg 2>/dev/null && continue
    CUDA_VISIBLE_DEVICES=$GPU timeout 7200 python3 scripts/plan/eval_wm.py --config-name tworoom \
      policy=${BASE}_tworoom eval.dataset_name=$CANON dataset.stats=$CANON \
      seed=$sd eval.goal_offset_steps=$off eval.eval_budget=$((off*2)) \
      solver=lip "solver.actor_path=$A" \
      output.filename=lipk_${BASE}_K${K}_s${s}_h${off}_sd${sd}.txt > $lg 2>&1
    v=$(grep -oE "success_rate[^0-9]*[0-9.]+" $lg | tail -1 | grep -oE "[0-9.]+$")
    echo "LIPK $BASE K$K s$s h$off sd$sd: ${v:-FAIL}" >> /workspace/logs/lipk_driver.log
  done; done
done; done
echo "LIPK_DONE $BASE gpu$GPU" >> /workspace/logs/lipk_driver.log
