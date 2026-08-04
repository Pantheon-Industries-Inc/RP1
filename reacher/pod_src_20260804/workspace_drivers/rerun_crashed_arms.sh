#!/bin/bash
# lejepa TD+CEM and pldm TD+Adam scored 0/6: every eval core-dumped under
# contention. Serial on GPU 0, which is how the two previous crashed arms
# (Dyna TD+CEM, valsteps lejepa e0.1) both recovered.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_baselines.csv
L=/workspace/logs/baselines
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][rerun] $*"; }
run(){ local B=$1 arm=$2; shift 2
  for s in 42 43 44 45 46 47; do
    nm=bl_${B}_${arm}_s${s}
    grep -q "^${nm},held=[0-9]" "$SUM" && continue
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
      policy=/workspace/swm_home/checkpoints/${B}_reacher \
      eval.dataset_name="$CANON" dataset.stats="$CANON" \
      +eval.ep_range=8000:10000 seed=$s \
      eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
      "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
    h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
    t=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
    [ -z "$h" ] && log "  !! ${nm} STILL no score: $(tail -1 $L/${nm}.log | cut -c1-60)"
    echo "${nm},held=${h:-FAIL},held10=${t:-FAIL}" >> "$SUM"
  done
  a=$(grep "^bl_${B}_${arm}_s" "$SUM" | grep -oE "held=[0-9.]+" | cut -d= -f2 | awk "{t+=\$1;n++}END{printf \"%.1f\", (n?t/n:0)}")
  b=$(grep "^bl_${B}_${arm}_s" "$SUM" | grep -oE "held10=[0-9.]+" | cut -d= -f2 | awk "{t+=\$1;n++}END{printf \"%.1f\", (n?t/n:0)}")
  n=$(grep -c "^bl_${B}_${arm}_s.*held=[0-9]" "$SUM")
  log "${B} ${arm}: @0.05 ${a} | @0.1 ${b}  (${n}/6 scored)"
}
run lejepa tdcem  solver=cem  solver.n_steps=10 "+metric=/workspace/metrics/window3_lejepa_e005.pt"
run pldm   tdadam solver=adam solver.n_steps=10 solver.num_samples=300 "+metric=/workspace/metrics/window3_pldm_e005.pt"
log RERUN_DONE
