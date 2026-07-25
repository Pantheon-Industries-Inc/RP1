#!/bin/bash
# Does training the actor on FAR goals fix long horizon?
# Controlled: identical to baseline lip4_r2w_s1 EXCEPT --max-delta 40 (fs5 units
# = 200 primitive steps, i.e. goals spanning the full deploy range) vs default 10
# (= 50 steps). Same WM (wm2_e2), same value init, same expectile 0.03, same seed.
# Baseline lip4_r2w_s1 s42: h25=92 h100=82 h150=42 h200=0.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel; PLAN=$CODE/scripts/plan
LOGS=/workspace/logs; SUM=/workspace/results/summary_hcurric.csv
DRV=$LOGS/driver_hcurric.log
WM=/workspace/models/wm2_e2
ACTOR=/workspace/actors/lip4_md40_s1.pt
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
mkdir -p /workspace/results; touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

if [ ! -f "$ACTOR" ]; then
  log "training LIP actor with --max-delta 40 (goals up to 200 primitive steps)"
  CUDA_VISIBLE_DEVICES=0 timeout 14400 python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/r2w_expert_fs5.pt --cache-td /workspace/caches/r2w_expert_fs1.pt \
    --h5 /workspace/datasets/expert_actions.h5 --wm "$WM" \
    --init-value /workspace/metrics/r2w_expert_TD.pt \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5 --max-delta 40 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed 1 \
    --out "$ACTOR" --out-value /workspace/metrics/lip4_md40_s1_value.pt \
    > "$LOGS/train_lip4_md40_s1.log" 2>&1 || { log "TRAIN FAILED"; tail -3 "$LOGS/train_lip4_md40_s1.log"; exit 1; }
  log "actor trained ($(grep -oE 'E_final [0-9.]+' $LOGS/train_lip4_md40_s1.log | tail -1))"
fi

for H in 25 100 150 200; do
  B=$((2*H)); [ "$H" = 25 ] && B=50
  name="md40_h${H}_s42"
  c=$(sc "$name"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "$name cached ($c)"; continue; }
  CUDA_VISIBLE_DEVICES=0 timeout 10800 python3 "$PLAN/eval_wm.py" --config-name cube \
    seed=42 eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=$H eval.eval_budget=$B policy="$WM" solver=lip \
    "solver.actor_path=$ACTOR" output.filename="${name}.txt" > "$LOGS/eval_${name}.log" 2>&1
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "$name FAILED"; echo "$name,FAIL">>"$SUM"; continue; }
  echo "$name,$sr">>"$SUM"; log "$name = $sr"
done
log "=== max-delta 40 (far-goal training) vs baseline max-delta 10, s42 ==="
log "h25:  $(sc md40_h25_s42)  (baseline 92)"
log "h100: $(sc md40_h100_s42) (baseline 82)"
log "h150: $(sc md40_h150_s42) (baseline 42)"
log "h200: $(sc md40_h200_s42) (baseline 0)"
log "HCURRIC_DONE"
