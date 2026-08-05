#!/bin/bash
# CEM at rh=1 on LeWM -- the missing cell of the planner x cadence 2x2.
#
# All four cells are UNALIGNED ("plain"), which is the apples-to-apples set and
# also what the source papers do (neither DINO-WM nor PLDM deadline-aligns):
#     RLP        rh5 90.00   rh1 73.33   (banked)
#     TD+CEM 3k  rh5 85.11   rh1 <-- this run
#     latent+CEM rh5 78.67   rh1 <-- this run
#
# CEM is exposed to the SAME deadline misalignment as RLP: LeWM.criterion scores
# pred_emb[..., -1:, :] against the goal, i.e. the plan's LAST predicted step, so
# at rh=1 the solves from t=30 on optimise a latent at t+25 > 50 that the episode
# never reaches. An aligned CEM would need the fix applied to the cost path
# (LeWM.criterion / get_cost), not to lip.py, and is a separate job. Do not read
# these numbers as "CEM with feedback done right" -- they are CEM under the same
# handicap RLP had at rh=1, which is what makes the comparison fair.
#
# 12 cells: TD+CEM 3 seeds x 3 draws (its critic is the per-seed co-trained EMA
# teacher, same as the main table) + latent+CEM 3 draws (no planner seed).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=/workspace/code/stable-worldmodel/scripts/plan; L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_cemrh1.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
WM=/workspace/models/v2WM
EPHI=8000
NSLOT=8; NGPU=4; SLOTDIR=/tmp/cemrh1slots
mkdir -p "$SLOTDIR" "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
cd /workspace/code/stable-worldmodel
crit(){ echo /workspace/actors/lip4_re_lewm_exp30_s${1}_value.pt; }

ev(){ # name extra-overrides...
  local nm=$1 d=$2; shift 2
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm banked ($c)"; return 0; }
  local slot; slot=$(acquire); local g=$(( slot % NGPU ))
  ( # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 14400 \
      python3 "$P/eval_wm.py" --config-name cube seed=$d \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=${EPHI}:10000" \
      plan_config.receding_horizon=1 \
      policy="$WM" "$@" output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
    sr=""
    grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
      sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
    flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
    log "  $nm = ${sr:-FAIL}"; release "$slot" ) &
}

log "=== CEM at rh=1 (pid $$): 12 cells"
for d in 42 43 44; do
  ev "f30_lcem3k_rh1_pre_lewm_e${d}" "$d" solver=cem solver.n_steps=10
done
for s in 0 1 2; do
  for d in 42 43 44; do
    ev "f30_tcem3k_rh1_pre_lewm_s${s}_e${d}" "$d" solver=cem solver.n_steps=10 "+metric=$(crit $s)"
  done
done
wait
log "cells done"
