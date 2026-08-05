#!/bin/bash
# Latent+CEM at n_steps 30, to give the 30-iteration Value+CEM number a partner.
#
# Value+CEM jumped 51.3 -> 63.3 (LeWM) and 51.3 -> 57.3 (PLDM) purely from CEM
# refinement iterations -- the expectile change did nothing (49.0 / 50.0) and
# LIP's co-trained critic barely moved it (49.3 / 54.3). But n_steps 30 is
# 300 x 30 = 9000 rollouts against the table's 3000, so the bar has to be given
# the same budget or the row is not comparable.
#
# Serial on one GPU: concurrent sampling-solver evals have core-dumped five
# times in this campaign and recovered every time when run one at a time.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_l2n30.csv; touch "$SUM"
L=/workspace/logs/l2n30; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][l2n30] $*"; }

for B in lejepa pldm; do
  for s in 42 43 44 45 46 47; do
    nm=l2n30_${B}_s${s}
    grep -q "^${nm},held=[0-9]" "$SUM" && continue
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
      policy=/workspace/swm_home/checkpoints/${B}_reacher \
      eval.dataset_name="$CANON" dataset.stats="$CANON" \
      +eval.ep_range=8000:10000 seed=$s eval.goal_offset_steps=25 \
      eval.eval_budget=50 solver=cem solver.n_steps=30 solver.batch_size=10 \
      "+metric=/workspace/metrics/l2window3.pt" \
      output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
    h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
    t=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
    [ -z "$h" ] && log "  !! ${nm} no score"
    echo "${nm},held=${h:-FAIL},held10=${t:-FAIL}" >> "$SUM"
  done
  a=$(grep "^l2n30_${B}_s" "$SUM" | grep -oE "held=[0-9.]+" | cut -d= -f2 \
      | awk '{t+=$1;n++}END{printf "%.1f",(n?t/n:0)}')
  b=$(grep "^l2n30_${B}_s" "$SUM" | grep -oE "held10=[0-9.]+" | cut -d= -f2 \
      | awk '{t+=$1;n++}END{printf "%.1f",(n?t/n:0)}')
  ref="39.3 / 78.3"; [ "$B" = "lejepa" ] && ref="44.7 / 84.3"
  log "Latent+CEM n30 ${B}: @0.05 ${a} | @0.1 ${b}  ($(grep -c "^l2n30_${B}_s.*held=[0-9]" "$SUM")/6)   [n10 was ${ref}]"
done
log "L2N30_DONE"
