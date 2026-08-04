#!/bin/bash
# Fill the Value+CEM retry cells serially on one GPU. The 6-wide launch
# core-dumped several arms -- the fifth time in this campaign that concurrent
# sampling-solver evals have died, and every previous one recovered when re-run
# one at a time.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_vcem.csv
L=/workspace/logs/vcem
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][vcemfix] $*"; }
one(){ local B=$1 tag=$2 metric=$3 nsteps=$4 s=$5
  nm=vc_${tag}_${B}_s${s}
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy=/workspace/swm_home/checkpoints/${B}_reacher \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$s eval.goal_offset_steps=25 \
    eval.eval_budget=50 solver=cem solver.n_steps=$nsteps solver.batch_size=10 \
    "+metric=${metric}" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && log "  !! ${nm} STILL no score"
  echo "${nm},held=${h:-FAIL},held10=${t:-FAIL}" >> "$SUM"
}
for B in lejepa pldm; do for s in 42 43 44 45 46 47; do
  one $B e01 /workspace/metrics/window3_${B}_e01_g098.pt 10 $s
  one $B co  /workspace/metrics/lip4_leak6_${B}_s0_value.pt 10 $s
  one $B n30 /workspace/metrics/window3_${B}_e005_g098.pt 30 $s
done; done
mean(){ grep "^vc_${2}_${1}_s" "$SUM" | grep -oE "${3}=[0-9.]+" | cut -d= -f2 | awk "{t+=\$1;n++}END{printf \"%.1f\", (n?t/n:0)}"; }
for B in lejepa pldm; do
  log "--- ${B} ---"
  log "  A e0.05 n10 (current)  @0.1 51.3"
  for tag in e01 co n30; do
    log "  ${tag}  @0.1 $(mean $B $tag held10)  @0.05 $(mean $B $tag held)  ($(grep -c "^vc_${tag}_${B}_s.*held=[0-9]" "$SUM")/6)"
  done
done
log VCEMFIX_DONE
