#!/bin/bash
# Fourth seed-0 arm: BOUNDARY CLIPPING (--amax), user-endorsed (unlike BC).
# Our cube runs use amax 3.5 (loose; the code default is 2.5 and the help text
# notes expert action tails reach ~3.5). But the campaign's wins elsewhere used
# a TIGHTER clip: TwoRoom min0's wall-trap fix was amax 2.2 (+max-delta 12) with
# the explicit finding "clip substitutes for gate", and reacher had "amax 2.2 won
# both" bases. A loose clip lets the actor push into action regions where the WM
# is least trustworthy — the fabrication regime that creates spurious attractors,
# which is a plausible cause of seed 0's bad basin.
#
# Trains on GPU 3 concurrently with the other arms (training concurrency is fine),
# then waits for FIXS0V2_DONE before evaluating (evals must be solo).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/workspace/code/stable-worldmodel/scripts/plan
L=/workspace/logs; SUM=/workspace/results/summary_fixs0.csv; DRV=$L/driver_fixs0amax.log
WM=/workspace/models/v2WM; TD=/workspace/metrics/v2_expert_TD.pt
FS1=/workspace/caches/v2_expert_fs1.pt; FS5=/workspace/caches/v2_expert_fs5.pt
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd /workspace/code/stable-worldmodel

TAG=amax22
OUT=/workspace/actors/lip4_fix_${TAG}.pt
if [ ! -f "$OUT" ]; then
  log "training $TAG (amax 2.2 vs baseline 3.5), gpu3, seed 0"
  CUDA_VISIBLE_DEVICES=3 timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 2.2 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed 0 \
    --out "$OUT" --out-value /workspace/metrics/lip4_fix_${TAG}_value.pt \
    > "$L/train_fix_${TAG}.log" 2>&1 || { log "$TAG TRAIN FAILED"; exit 1; }
  log "$TAG trained (E_final $(grep -oE 'E_final [0-9.]+' "$L/train_fix_${TAG}.log" | tail -1 | awk '{print $2}'))"
fi

log "waiting for FIXS0V2_DONE before evaluating (solo-eval rule)"
for i in $(seq 1 240); do grep -q FIXS0V2_DONE "$L/driver_fixs0v2.log" 2>/dev/null && break; sleep 60; done
while pgrep -f "eval_w[m].py" > /dev/null; do sleep 45; done
sleep 15

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
log "=== amax22: $(sc fix_${TAG}_e42)/$(sc fix_${TAG}_e43)/$(sc fix_${TAG}_e44) -> $(m3 "$(sc fix_${TAG}_e42)" "$(sc fix_${TAG}_e43)" "$(sc fix_${TAG}_e44)")  (baseline s0 = 68.0, target >=85) ==="
log "FIXS0AMAX_DONE"
