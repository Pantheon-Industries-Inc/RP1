#!/bin/bash
# Paper-protocol rerun: CEM at n_steps=10 (LeWM paper specifies 10 iterations for
# all non-PushT environments; our campaign used 30). std surface only.
# Usage: it10_rerun.sh <base> <gpu> <arm:latent|td> [td_seeds...]
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel
export HF_HOME=/root/hf TQDM_DISABLE=1 OMP_NUM_THREADS=14 MKL_NUM_THREADS=14
export HDF5_PLUGIN_PATH=$(python3 -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')
cd /workspace/code/stable-worldmodel
BASE=$1; GPU=$2; ARM=$3; shift 3
CANON=/workspace/datasets_canon/tworoom/tworoom.h5
CK=${BASE}_tworoom
EXTRA=(); [ "$BASE" = dinowm ] && EXTRA=(dataset.keys_to_cache='[action]')
case $BASE in lejepa|pldm) TDTAG="" ;; dinowm) TDTAG="_r200000" ;; esac

run(){ # tag metricflag sd off
  local tag=$1 mf=$2 sd=$3 off=$4 bud=$(($4*2))
  local lg=/workspace/logs/it10_${BASE}_${tag}_h${off}_sd${sd}.log
  grep -q success_rate $lg 2>/dev/null && return 0
  local M=(); [ -n "$mf" ] && M=(+metric=$mf)
  CUDA_VISIBLE_DEVICES=$GPU timeout 14400 python3 scripts/plan/eval_wm.py --config-name tworoom \
    policy=$CK eval.dataset_name=$CANON dataset.stats=$CANON "${EXTRA[@]}" \
    seed=$sd eval.goal_offset_steps=$off eval.eval_budget=$bud \
    solver=cem solver.n_steps=10 solver.batch_size=10 "${M[@]}" \
    output.filename=it10_${BASE}_${tag}_h${off}_sd${sd}.txt > $lg 2>&1
  local v=$(grep -oE 'success_rate[^0-9]*[0-9.]+' $lg | tail -1 | grep -oE '[0-9.]+$')
  echo "IT10 $BASE $tag h${off} sd${sd}: ${v:-FAIL}" >> /workspace/logs/it10_driver.log
}

for sd in 42 43 44; do for off in 25 50; do
  if [ "$ARM" = latent ]; then
    run latent_cem "" $sd $off
  else
    for ts in "$@"; do
      run td_cem_t${ts} /workspace/metrics/td_canon_${BASE}${TDTAG}_e0.1_n50_s${ts}.pt $sd $off
    done
  fi
done; done
echo "IT10_DONE $BASE $ARM gpu$GPU" >> /workspace/logs/it10_driver.log
