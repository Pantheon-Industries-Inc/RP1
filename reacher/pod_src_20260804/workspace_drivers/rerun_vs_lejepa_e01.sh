#!/bin/bash
# The lejepa expectile-0.1 arm of the value-steps ladder: all 8 evals core-dumped
# under contention (4 CEM groups + 30 LIP trainings sharing 5 GPUs) and were
# recorded as 0.0. The Dyna TD+CEM arm crashed the same way and then succeeded
# when re-run serially, so these go one at a time on the now-idle GPU 0.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_valsteps.csv
L=/workspace/logs/valsteps
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][vsfix] $*"; }
for st in 6000 20000 60000 150000; do
  W=/workspace/metrics/window3_lejepa_e01.pt
  [ "$st" != "6000" ] && W=/workspace/metrics/window3_lejepa_e01_st${st}.pt
  [ -f "$W" ] || { log "missing $W"; continue; }
  for s in 50 51; do
    nm=vs_lejepa_e01_st${st}_sel${s}
    grep -q "^${nm},held=[0-9]" "$SUM" && continue
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
      policy=/workspace/swm_home/checkpoints/lejepa_reacher \
      eval.dataset_name="$CANON" dataset.stats="$CANON" \
      +eval.ep_range=8000:10000 seed=$s \
      eval.goal_offset_steps=25 eval.eval_budget=50 \
      solver=cem solver.n_steps=10 solver.batch_size=10 \
      "+metric=$W" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
    h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
    t=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
    [ -z "$h" ] && log "  !! ${nm} still no score: $(tail -1 $L/${nm}.log | cut -c1-60)"
    echo "${nm},held=${h:-FAIL},held10=${t:-FAIL}" >> "$SUM"
  done
  a=$(grep "^vs_lejepa_e01_st${st}_sel" "$SUM" | grep -oE "held=[0-9.]+" | cut -d= -f2 | awk "{t+=\$1;n++}END{printf \"%.1f\", (n?t/n:0)}")
  log "lejepa e0.1 ${st} steps: TD+CEM ${a}"
done
log "VSFIX_DONE"
