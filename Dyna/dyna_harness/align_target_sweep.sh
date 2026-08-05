#!/bin/bash
# ALIGNMENT-TARGET SWEEP for LeWM RLP at rh=1. Eval-only, no retraining.
#
# WHAT IS BEING SWEPT. The patched policy computes
#     chunks_remaining = ceil((plan_config.eval_budget - steps_done) / action_block)
# and LIPSolver reads E and grad_A V at index min(horizon, chunks_remaining)-1.
# So `plan_config.eval_budget` is the ALIGNMENT TARGET -- the step the final plan
# is made to end on. It is NOT the episode length: the real cap stays
# eval.eval_budget=50 and is set by the world. (The field name is inherited from
# the first version of the patch and is a wart; it changes nothing about how long
# an episode runs.)
#
# WHY THIS SWEEP. The first aligned run targeted 50, which only bites once
# t+25 > 50, i.e. the LAST 4 of 10 solves -- six solves were untouched, which is
# arithmetically why only 48% of the cadence drop came back:
#     target 50: term_idx per solve = 4 4 4 4 4 4 3 2 1 0
#     target 25: term_idx per solve = 4 3 2 1 0 0 0 0 0 0
# Target 25 is the goal's own time (goal_offset_steps=25), i.e. "plan to arrive
# when the expert arrives". It is aggressive: past t=20 it collapses to a single
# chunk, making RLP a greedy 1-chunk policy and discarding the plan. Target 50 is
# conservative and keeps the full horizon. The right answer is in between and
# depends on whether the agent is on schedule, which is not knowable a priori --
# hence a sweep rather than a guess.
#
# READ AGAINST: rh1 plain 73.33 (no alignment), rh1 target-50 81.33 (banked),
# rh5 90.00 (aligned or not -- provably identical, max per-cell |diff| 0.00).
# 36 cells, ~8 min.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=/workspace/code/stable-worldmodel/scripts/plan; L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_algsweep.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
WM=/workspace/models/v2WM
EPHI=8000
NSLOT=8; NGPU=4; SLOTDIR=/tmp/algswslots
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
from stable_worldmodel.solver import LIPSolver
import stable_worldmodel as swm
ok = ("align_deadline" in inspect.signature(LIPSolver.__init__).parameters
      and hasattr(LIPSolver, "set_align_remaining")
      and "eval_budget" in swm.PlanConfig.__dataclass_fields__)
print("[preflight] alignment patch live:", ok)
sys.exit(0 if ok else 1)
PY
log "=== alignment-target sweep, rh=1, targets 25/30/35/40 (50 banked)"

for tgt in 25 30 35 40; do
  for s in 0 1 2; do
    for d in 42 43 44; do
      nm="f30_lip_rh1a${tgt}_pre_lewm_s${s}_e${d}"
      c=$(sc "$nm")
      [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm banked ($c)"; continue; }
      slot=$(acquire); g=$(( slot % NGPU ))
      (
        CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 14400 \
          python3 "$P/eval_wm.py" --config-name cube seed=$d \
          eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
          eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=${EPHI}:10000" \
          plan_config.receding_horizon=1 "+plan_config.eval_budget=${tgt}" \
          policy="$WM" solver=lip "+solver.align_deadline=true" \
          "solver.actor_path=/workspace/actors/lip4_re_lewm_exp30_s${s}.pt" \
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
done
wait
log "sweep cells done"
