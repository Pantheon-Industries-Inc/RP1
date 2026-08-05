#!/bin/bash
# The PLDM baseline bars, which were never measured under this protocol.
#
# Everything carded so far -- Latent+CEM-window 44.7, TD+CEM-window 17.7, LIP
# 48.9 -- is lejepa. The joint sweep produces PLDM LIP numbers (18.7-34.0 on
# selection seeds) with nothing to compare them to, so "LIP beats the paper's
# method on PLDM" is currently unfalsifiable. This measures the two CEM bars on
# PLDM at the identical protocol.
#
# The only PLDM number on record is the paper-exact LATCHED anchor (75.3 vs
# lejepa 82.7). That is a different metric (latched, not held-at-end) and a
# different config, so it is not a substitute.
#
# PROTOCOL, identical to the lejepa cards:
#   held-at-end @ 0.05 rad, h25, eval budget 50, CEM 300x10, batch 10
#   eval pool episodes 8000:10000, reporting seeds 42..47
#
# l2window3.pt is reused as-is: its state_dict is empty (parameter-free L2 in
# window space) and both bases are 192-d, so 3x192 = 576 matches. The TD bar
# uses window3_pldm_e01.pt -- expectile 0.1, matching the lejepa TD+CEM card.
#
# Runs 2-wide on GPUs 4,5 so it does not starve the joint sweep.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/pldm_reacher
SUM=/workspace/results/summary_bars_pldm.csv; touch "$SUM"
L=/workspace/logs/bars; mkdir -p "$L"
REP="42 43 44 45 46 47"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][bars] $*"; }

[ -d "$WM" ] || { log "FATAL: no PLDM checkpoint at $WM"; ls /workspace/swm_home/checkpoints; exit 1; }

ev(){ local gpu=$1 nm=$2 seed=$3; shift 3
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=cem solver.n_steps=10 solver.batch_size=10 \
    "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
meanof(){ local pre=$1 key=$2; local t=0 n=0 e
  for s in $REP; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

log "PLDM bars on reporting seeds ${REP} (2-wide, GPUs 4,5)"
( for s in $REP; do ev 4 "bar_pldm_l2_s${s}" $s "+metric=/workspace/metrics/l2window3.pt"; done ) &
( for s in $REP; do ev 5 "bar_pldm_td_s${s}" $s "+metric=/workspace/metrics/window3_pldm_e01.pt"; done ) &
wait

log "======================= PLDM BARS ======================="
log "Latent+CEM-window (PLDM): HELD $(meanof bar_pldm_l2_s held) | @0.1 $(meanof bar_pldm_l2_s held10)   [lejepa 44.7 / 84.3]"
log "TD+CEM-window     (PLDM): HELD $(meanof bar_pldm_td_s held) | @0.1 $(meanof bar_pldm_td_s held10)   [lejepa 17.7 / 45.3]"
log "BARS_DONE"
