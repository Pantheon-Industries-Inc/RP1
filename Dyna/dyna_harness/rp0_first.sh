#!/bin/bash
# replay-prob 0.0 at 3 seeds, min objective, 6000 steps, LeWM.
#
# THE QUESTION: does the replay curriculum help AT ALL at rh=1? It banks
# post-FULL-plan states (tr[:, -3:], i.e. stride 5) -- the query an rh=5 replan
# makes, not an rh=1 one. So at rh=1 it may be feeding the actor the wrong
# mid-task distribution, and switching it off could beat every nonzero setting.
# The 1-seed screen trended monotone downward in replay-prob under the min
# objective (0.25 -> 88.00, 0.5 -> 86.00, 0.75 -> 83.33), which extrapolates here.
#
# If prob 0 >= prob 0.25 at 3 seeds, the replay axis collapses to "turn it off"
# and the rest of the grid (0.125, and the stride column) is largely moot -- the
# screen's trend would then be about removing a harmful curriculum rather than
# tuning a helpful one.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=/workspace/code/stable-worldmodel/scripts/plan; L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_rp0.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
WM=/workspace/models/v2WM; TD=/workspace/metrics/up_lewmpre_g98s12k_s0.pt
C5=/workspace/caches/v2_tr8000_fs5.pt; C1=/workspace/caches/v2_tr8000_fs1.pt
AH5=/workspace/datasets/expert_actions.h5
NGPU=4; NSLOT=4; SLOTDIR=/tmp/rp0slots
mkdir -p "$SLOTDIR" "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 10; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
cd /workspace/code/stable-worldmodel

BASE="--arch v4 --horizon 5 --iters 8 --n-step 50 --gamma 0.98 --batch 256 \
--amax 1.6 --expand-weight 3.0 --freeze-critic-frac 0.5 \
--actor-lr 3e-4 --actor-lr-final 3e-5 --critic-lr 1e-3 --critic-lr-final 1e-4 \
--expectile 0.1 --expectile-final 0.03 --mean-weight 0.1 --term-index min"

log "=== replay-prob 0.0, min objective, 3 seeds, 6000 steps"
for s in 0 1 2; do
  out=/workspace/actors/lip4_sc_min_rp0_s${s}.pt
  [ -f "$out" ] && { log "  s$s present"; continue; }
  slot=$(acquire)
  ( # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
      --cache "$C5" --cache-td "$C1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
      $BASE --replay-prob 0.0 --steps 6000 --seed "$s" \
      --out "$out" --out-value "${out%.pt}_value.pt" > "$L/sc_min_rp0_s${s}.log" 2>&1
    release "$slot" ) &
  log "  -> s$s"
done
wait
log "trains done: $(ls /workspace/actors/lip4_sc_min_rp0_s[0-9].pt 2>/dev/null | wc -l)/3"

NSLOT=8; rmdir "$SLOTDIR"/* 2>/dev/null || true
for s in 0 1 2; do
  A=/workspace/actors/lip4_sc_min_rp0_s${s}.pt
  [ -f "$A" ] || { log "  WARN no actor s$s"; continue; }
  for mode in rh1 rh5; do
    for d in 42 43 44; do
      nm="f30_lip_scmin_rp0_${mode}_pre_lewm_s${s}_e${d}"
      c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && continue
      if [ "$mode" = rh1 ]; then
        EX=(plan_config.receding_horizon=1 "+plan_config.eval_budget=50"
            "+solver.align_deadline=true" "+solver.align_mode=argmin")
      else
        EX=(plan_config.receding_horizon=5)
      fi
      slot=$(acquire)
      (
        CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) MUJOCO_EGL_DEVICE_ID=$(( slot % NGPU )) \
        timeout 14400 python3 "$P/eval_wm.py" --config-name cube seed=$d \
          eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
          eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=8000:10000" \
          "${EX[@]}" policy="$WM" solver=lip "solver.actor_path=$A" \
          output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
        sr=""
        grep -q "ep_range 8000:10000" "$L/eval_${nm}.log" && \
          sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
        flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
        log "  $nm = ${sr:-FAIL}"; release "$slot"
      ) &
    done
  done
done
wait
log "RP0_DONE"
