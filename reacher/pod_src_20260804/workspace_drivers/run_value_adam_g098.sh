#!/bin/bash
# Value+Adam on the gamma-0.98 value, so every "value + X" row shares one value.
#
# The baseline table has an internal inconsistency: TD+CEM was re-measured on
# the gamma-0.98 value (lejepa 51.3, pldm 51.3 @0.1) but TD+Adam was only ever
# run against the gamma-1.0 value (23.0 / 25.0). Since the reported recipe is
# gamma 0.98, quoting those two side by side would compare planners across
# different cost functions.
#
# Adam gets num_samples 300 x n_steps 10 -- exactly CEM's 3000-rollout budget --
# so the CEM/Adam row difference is planner, not compute.
#
# Serial-ish (2-wide): four sampling-solver arms have core-dumped under
# contention in this campaign and every one recovered when re-run with fewer in
# flight.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_vadam098.csv; touch "$SUM"
L=/workspace/logs/vadam098; mkdir -p "$L"
REP="42 43 44 45 46 47"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][vadam] $*"; }

ev(){ local gpu=$1 B=$2 nm=$3 seed=$4
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="/workspace/swm_home/checkpoints/${B}_reacher" \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=adam solver.n_steps=10 solver.num_samples=300 solver.batch_size=10 \
    "+metric=/workspace/metrics/window3_${B}_e005_g098.pt" \
    output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && log "  !! ${nm} no score: $(tail -1 "$L/${nm}.log" | cut -c1-60)"
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
meanof(){ local B=$1 key=$2; local t=0 n=0 e
  for s in $REP; do
    e=$(grep "^va_${B}_s${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"; }

log "Value+Adam (gamma-0.98 value), 2 bases x 6 reporting seeds, 2-wide"
( for s in $REP; do ev 0 lejepa "va_lejepa_s${s}" $s; done ) &
( for s in $REP; do ev 1 pldm   "va_pldm_s${s}"   $s; done ) &
wait
for B in lejepa pldm; do
  log "Value+Adam ${B}: @0.05 $(meanof $B held) | @0.1 $(meanof $B held10)  ($(grep -c "^va_${B}_s.*held=[0-9]" "$SUM")/6)   [gamma-1.0 value gave $([ $B = lejepa ] && echo 23.0 || echo 25.0) @0.1]"
done
log "VADAM_DONE"
