#!/bin/bash
# DECISIVE TEST: does the rank-only value convert into PLANNING performance?
# LIP actor on wm2_e2, critic initialised from the rank-only value and kept
# pure-ranking through the tandem (--critic-td-weight 0 --critic-rank-weight 2).
# Everything else identical to the baseline actor lip4_r2w_s1 (seed 1, arch v4,
# amax 3.5, 6k steps). Baseline s42: h25 92 | h100 82 | h150 42 | h200 0.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/workspace/code/stable-worldmodel/scripts/plan
L=/workspace/logs; SUM=/workspace/results/summary_rankonly.csv
DRV=$L/driver_rankonly.log
WM=/workspace/models/wm2_e2
ACTOR=/workspace/actors/lip4_rankonly_s1.pt
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
mkdir -p /workspace/results; touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd /workspace/code/stable-worldmodel

if [ ! -f "$ACTOR" ]; then
  log "training LIP actor with PURE-RANKING critic (init vfix_rankonly)"
  CUDA_VISIBLE_DEVICES=0 timeout 14400 python3 "$P/train_lip_ac.py" \
    --cache /workspace/caches/r2w_expert_fs5.pt --cache-td /workspace/caches/r2w_expert_fs1.pt \
    --h5 /workspace/datasets/expert_actions.h5 --wm "$WM" \
    --init-value /workspace/metrics/vfix_rankonly.pt \
    --critic-td-weight 0 --critic-rank-weight 2.0 --critic-rank-margin 0.5 \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed 1 \
    --out "$ACTOR" --out-value /workspace/metrics/lip4_rankonly_s1_value.pt \
    > "$L/train_lip4_rankonly.log" 2>&1 || { log "TRAIN FAILED"; tail -5 "$L/train_lip4_rankonly.log" | tee -a "$DRV"; exit 1; }
  log "actor trained ($(grep -oE 'E_final [0-9.]+' "$L/train_lip4_rankonly.log" | tail -1))"
fi

for H in 25 100 150; do
  B=$((2*H)); [ "$H" = 25 ] && B=50
  n="ro_h${H}_s42"
  c=$(sc "$n"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "$n cached ($c)"; continue; }
  CUDA_VISIBLE_DEVICES=0 timeout 10800 python3 "$P/eval_wm.py" --config-name cube \
    seed=42 eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=$H eval.eval_budget=$B policy="$WM" solver=lip \
    "solver.actor_path=$ACTOR" output.filename="${n}.txt" > "$L/eval_${n}.log" 2>&1
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${n}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${n},${sr:-FAIL}" >> "$SUM"; log "$n = ${sr:-FAIL}"
done
log "=== RANK-ONLY vs baseline (s42) ==="
log "h25:  $(sc ro_h25_s42)  (baseline 92)"
log "h100: $(sc ro_h100_s42) (baseline 82)"
log "h150: $(sc ro_h150_s42) (baseline 42)"
log "RANKONLY_PLANNING_DONE"
