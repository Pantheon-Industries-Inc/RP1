#!/bin/bash
# latent+CEM at rh=1, alignment-target sweep -- matched to the RLP sweep so the
# two curves are directly comparable.
#
# HOW CEM IS ALIGNED. LeWM.criterion always reads pred_emb[..., -1:, :], and it is
# shared by every sampling planner, so it was NOT touched. Instead CEMSolver
# truncates the SCORED candidate to the first k chunks (k = chunks the episode has
# left), which makes that terminal read land on the episode's last step. The
# emitted plan stays full-horizon. Guarded by align_deadline (default false) and
# skipped when k == horizon, so rh=5 is inert -- verified: rh5 + align on returns
# 82.0 on draw 42, exactly the banked f30_lcem3k_pre_lewm_e42.
#
# READ AGAINST: latent+CEM rh5 78.67, rh1 unaligned 75.33, and the RLP curve
# (rh5 90.00, rh1 unaligned 73.33, rh1 best-aligned 84.89 at target 40).
# latent+CEM has NO planner seed, so 3 draws per target; spread is draw-only.
# 15 cells, ~6 min.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=/workspace/code/stable-worldmodel/scripts/plan; L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_lcemalg.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
WM=/workspace/models/v2WM
EPHI=8000
NSLOT=8; NGPU=4; SLOTDIR=/tmp/lcemalgslots
mkdir -p "$SLOTDIR" "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
cd /workspace/code/stable-worldmodel

python3 - <<'PY' || exit 1
import sys, inspect
sys.path.insert(0, "/workspace/code/stable-worldmodel")
from stable_worldmodel.solver import CEMSolver
ok = ("align_deadline" in inspect.signature(CEMSolver.__init__).parameters
      and hasattr(CEMSolver, "set_align_remaining"))
print("[preflight] CEM alignment patch live:", ok)
sys.exit(0 if ok else 1)
PY
log "=== latent+CEM rh=1 alignment sweep, targets 25/30/35/40/50"

for tgt in 25 30 35 40 50; do
  for d in 42 43 44; do
    nm="f30_lcem3k_rh1a${tgt}_pre_lewm_e${d}"
    c=$(sc "$nm")
    [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm banked ($c)"; continue; }
    slot=$(acquire); g=$(( slot % NGPU ))
    (
      CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 14400 \
        python3 "$P/eval_wm.py" --config-name cube seed=$d \
        eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
        eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=${EPHI}:10000" \
        plan_config.receding_horizon=1 "+plan_config.eval_budget=${tgt}" \
        policy="$WM" solver=cem solver.n_steps=10 "+solver.align_deadline=true" \
        output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
      sr=""
      grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
        sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
      flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
      log "  $nm = ${sr:-FAIL}"
      release "$slot"
    ) &
  done
done
wait
log "cells done"
