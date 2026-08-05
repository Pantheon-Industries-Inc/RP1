#!/bin/bash
# Matched-seed POST-Dyna baseline. Runs after the v2WM replication so both
# columns of the headline comparison are measured under identical conditions
# (same recipe, same seed set, same harness, same draws).
#
#   pre-Dyna  (v2WM)          seeds 0,1,2 = 68.0 / 81.3 / 88.7   + new 3,4,5
#   post-Dyna (WM1=r1_5050b)  seeds 0,1,2 = 94.0 / 93.3 / 98.7   + new 3,4,5
#
# Historical pre-Dyna reference: 86.7 / 88.0 / 88.0 = 87.6 (3 seeds).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel; PLAN=$CODE/scripts/plan
L=/workspace/logs; SUM=/workspace/results/summary_dynarep.csv; DRV=$L/driver_dynarep.log
WM=/workspace/models/dyna_r1_5050b
TD=/workspace/metrics/r1w_expert_TD.pt
FS1=/workspace/caches/r1w_expert_fs1.pt
FS5=/workspace/caches/r1w_expert_fs5.pt
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
mkdir -p /workspace/results; touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }

log "waiting for V2REP_DONE"
for i in $(seq 1 400); do grep -q "V2REP_DONE" "$L/driver_v2rep.log" 2>/dev/null && break; sleep 60; done
grep -q "V2REP_DONE" "$L/driver_v2rep.log" || { log "v2rep never finished; aborting"; exit 1; }
while pgrep -f "train_lip_a[c]" > /dev/null || pgrep -f "eval_w[m]" > /dev/null; do sleep 60; done
sleep 20
cd "$CODE"

train_seed(){
  local gpu=$1 seed=$2
  local out="/workspace/actors/lip4_dynarep_s${seed}.pt"
  [ -f "$out" ] && { log "seed $seed cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 14400 python3 "$PLAN/train_lip_ac.py" \
    --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$seed" \
    --out "$out" --out-value "/workspace/metrics/lip4_dynarep_s${seed}_value.pt" \
    > "$L/train_lip4_dynarep_s${seed}.log" 2>&1 || { log "seed $seed TRAIN FAILED"; return 1; }
  log "seed $seed trained ($(grep -oE 'E_final [0-9.]+' "$L/train_lip4_dynarep_s${seed}.log" | tail -1))"
}
log "training Dyna seeds 3,4,5 in parallel (gpu 0/1/2)"
train_seed 0 3 & train_seed 1 4 & train_seed 2 5 &
wait

for s in 3 4 5; do
  A=/workspace/actors/lip4_dynarep_s${s}.pt
  [ -f "$A" ] || { log "seed $s actor missing"; continue; }
  for d in 42 43 44; do
    n="dynarep_s${s}_e${d}"
    c=$(sc "$n"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "$n cached ($c)"; continue; }
    CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$PLAN/eval_wm.py" --config-name cube \
      seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 policy="$WM" solver=lip \
      "solver.actor_path=$A" output.filename="${n}.txt" > "$L/eval_${n}.log" 2>&1
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${n}.log" | tail -1 | grep -oE "[0-9.]+$")
    echo "${n},${sr:-FAIL}" >> "$SUM"; log "$n = ${sr:-FAIL}"
  done
done

m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
log "=== POST-DYNA seed replication (existing s0/s1/s2 = 94.0 / 93.3 / 98.7) ==="
for s in 3 4 5; do
  log "new seed $s: $(sc dynarep_s${s}_e42)/$(sc dynarep_s${s}_e43)/$(sc dynarep_s${s}_e44) -> $(m3 "$(sc dynarep_s${s}_e42)" "$(sc dynarep_s${s}_e43)" "$(sc dynarep_s${s}_e44)")"
done
log "DYNAREP_DONE"
