#!/bin/bash
# Diagnostic: TD+CEM using the gamma-0.98 value, both bases, reporting seeds.
#
# Gamma 0.98 lifted LIP on lejepa by 6.8 points @0.1. That could be either
#   (a) the VALUE got better -- discounting fixed the cost-to-go, or
#   (b) only the actor optimisation improved, the value being unchanged.
# TD+CEM reads the value directly, with no learned actor in the loop, so it
# separates the two. Reference points, both measured on the same protocol:
#   TD+CEM with the gamma-1.0 value : lejepa (re-running) / pldm 11.3 / 33.7
#   L2+CEM, the parameter-free bar  : lejepa 44.7 / 84.3, pldm 39.3 / 78.3
# A jump toward the L2 numbers means (a), and says push on the value. Flat
# means (b), and says the value is still the bottleneck -- which would also
# explain why 25x more value-training steps changed nothing downstream.
#
# Serial on GPU 0: four CEM arms have core-dumped under contention in this
# campaign and every one recovered when re-run one at a time.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_tdcem_g098.csv; touch "$SUM"
L=/workspace/logs/tdcem98; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][tdg98] $*"; }

# do not contend with the crashed-arm re-run, which owns GPU 0
while pgrep -f rerun_crashed_arms >/dev/null 2>&1; do sleep 60; done

for B in lejepa pldm; do
  W=/workspace/metrics/window3_${B}_e005_g098.pt
  [ -f "$W" ] || { log "missing ${W} -- skipping ${B}"; continue; }
  for s in 42 43 44 45 46 47; do
    nm=tdg98_${B}_s${s}
    grep -q "^${nm},held=[0-9]" "$SUM" && continue
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
      policy=/workspace/swm_home/checkpoints/${B}_reacher \
      eval.dataset_name="$CANON" dataset.stats="$CANON" \
      +eval.ep_range=8000:10000 seed=$s \
      eval.goal_offset_steps=25 eval.eval_budget=50 \
      solver=cem solver.n_steps=10 solver.batch_size=10 \
      "+metric=$W" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
    h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
    t=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
    [ -z "$h" ] && log "  !! ${nm} no score: $(tail -1 "$L/${nm}.log" | cut -c1-60)"
    echo "${nm},held=${h:-FAIL},held10=${t:-FAIL}" >> "$SUM"
  done
  a=$(grep "^tdg98_${B}_s" "$SUM" | grep -oE "held=[0-9.]+" | cut -d= -f2 \
      | awk '{t+=$1;n++}END{printf "%.1f", (n?t/n:0)}')
  b=$(grep "^tdg98_${B}_s" "$SUM" | grep -oE "held10=[0-9.]+" | cut -d= -f2 \
      | awk '{t+=$1;n++}END{printf "%.1f", (n?t/n:0)}')
  n=$(grep -c "^tdg98_${B}_s.*held=[0-9]" "$SUM")
  bar="39.3 / 78.3"; [ "$B" = "lejepa" ] && bar="44.7 / 84.3"
  log "TD+CEM gamma0.98 ${B}: @0.05 ${a} | @0.1 ${b}  (${n}/6 scored)   [L2 bar ${bar}]"
done
log "TDCEM_G098_DONE"
