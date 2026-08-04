#!/bin/bash
# Serial filler for the missing wave-A eval cells.
#
# Why serial: with 32 grid trainings occupying all 6 GPUs, running 6 concurrent
# EGL eval streams reproduces the known SIGABRT ("timeout: the monitored command
# dumped core"). A single eval alongside the trainings verified fine, so this
# walks the missing cells one at a time on GPU 3 (also spreading across 3/4/5 to
# use the least-loaded cards). Slower, but it finishes instead of failing.
#
# Idempotent: skips any cell already present with a numeric held value, and
# purges stale FAIL rows for the cells it is about to run.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
SUM=/workspace/results/summary_waveA_lejepa.csv
L=/workspace/logs/waveA
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][fill] $*"; }

grep -v "held=FAIL" "$SUM" > /tmp/f.csv && mv /tmp/f.csv "$SUM"

TAGS="centre amax18 amax20 amax26 mw03 mw05 alr1e4 alr1e3 steps16k"
GPUS="3 4 5"
gi=0
for tag in $TAGS; do
  for seed in 0 1 2; do
    A=/workspace/actors/lip4_wa_${tag}_s${seed}.pt
    [ -f "$A" ] || continue
    for s in 42 43 44 45 46 47; do
      nm=wa_${tag}_s${seed}_e${s}
      grep -q "^${nm},held=[0-9]" "$SUM" && continue
      gpu=$(echo $GPUS | cut -d" " -f$(( (gi % 3) + 1 ))); gi=$((gi+1))
      MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
      timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
        policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
        +eval.ep_range=8000:10000 seed=$s \
        eval.goal_offset_steps=25 eval.eval_budget=50 \
        solver=lip solver.actor_path="$A" solver.rollout_compat=false \
        solver.batch_size=10 output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
      h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
      l=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
      echo "${nm},held=${h:-FAIL},latched=${l:-FAIL}" >> "$SUM"
      log "  ${nm}: HELD ${h:-FAIL}"
    done
    t=0; n=0
    for s in 42 43 44 45 46 47; do
      e=$(grep "^wa_${tag}_s${seed}_e${s},"  "$SUM" | tail -1 | grep -oE "held=[0-9.]+" | grep -oE "[0-9.]+")
      [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
    done
    [ "$n" -gt 0 ] && log "CARD ${tag} s${seed}: HELD $(awk "BEGIN{printf \"%.1f\", $t/$n}") (n=${n})"
  done
done
log "FILL_SERIAL_DONE"
