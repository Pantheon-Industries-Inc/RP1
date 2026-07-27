#!/bin/bash
# Fifth seed-0 arm: amax 2.2 + lowlr COMBINED.
# Individually on seed 0 (baseline 68.0, good seeds 81.3/88.7, target >=85):
#   amax 2.2  -> 84.0  (best single arm; boundary clipping limits how far the
#                       actor can push into the WM's fabrication regime)
#   lowlr     -> 80.0  (actor-lr 3e-4->1e-4, final 3e-5->1e-5)
#   iters 16  -> 80.0  (best E_final 1.68 but WORSE eval than amax22's 3.77 —
#                       harder optimization against a flawed WM = better
#                       fantasies, not better behaviour)
#   batch 256 -> 66.7  (no help)
# Both winners reduce optimization aggression by different mechanisms
# (action-space constraint vs step size), so they may compound.
# NO BC (--bc-weight stays 0, standing constraint).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/workspace/code/stable-worldmodel/scripts/plan
L=/workspace/logs; SUM=/workspace/results/summary_fixs0.csv; DRV=$L/driver_fixs0combo.log
WM=/workspace/models/v2WM; TD=/workspace/metrics/v2_expert_TD.pt
FS1=/workspace/caches/v2_expert_fs1.pt; FS5=/workspace/caches/v2_expert_fs5.pt
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd /workspace/code/stable-worldmodel

TAG=combo
OUT=/workspace/actors/lip4_fix_${TAG}.pt
if [ ! -f "$OUT" ]; then
  log "training $TAG = amax 2.2 + actor-lr 1e-4->1e-5 (seed 0, gpu0)"
  CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
    --horizon 5 --iters 8 --steps 6000 --amax 2.2 --n-step 50 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 1e-4 --actor-lr-final 1e-5 --arch v4 --seed 0 \
    --out "$OUT" --out-value /workspace/metrics/lip4_fix_${TAG}_value.pt \
    > "$L/train_fix_${TAG}.log" 2>&1 || { log "$TAG TRAIN FAILED"; tail -4 "$L/train_fix_${TAG}.log" | tee -a "$DRV"; exit 1; }
  log "$TAG trained (E_final $(grep -oE 'E_final [0-9.]+' "$L/train_fix_${TAG}.log" | tail -1 | awk '{print $2}'))"
fi

for d in 42 43 44; do
  n="fix_${TAG}_e${d}"
  c=$(sc "$n"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "$n cached ($c)"; continue; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 \
    eval.eval_budget=50 policy="$WM" solver=lip "solver.actor_path=$OUT" \
    output.filename="${n}.txt" > "$L/eval_${n}.log" 2>&1
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${n}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${n},${sr:-FAIL}" >> "$SUM"; log "$n = ${sr:-FAIL}"
done
m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
log "=== combo (amax2.2 + lowlr): $(sc fix_${TAG}_e42)/$(sc fix_${TAG}_e43)/$(sc fix_${TAG}_e44) -> $(m3 "$(sc fix_${TAG}_e42)" "$(sc fix_${TAG}_e43)" "$(sc fix_${TAG}_e44)") ==="
log "    refs: baseline 68.0 | lowlr 80.0 | iters16 80.0 | amax22 84.0 | target >=85 | good seeds 81.3/88.7"
log "FIXS0COMBO_DONE"
