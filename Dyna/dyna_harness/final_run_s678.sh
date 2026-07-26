#!/bin/bash
# FINAL REPORTABLE RUN — completely fresh seeds 6/7/8 on both sides.
#   Phase A: LIPv4 on v2WM        (pre-Dyna baseline)
#   Phase B: LIPv4 on WM1         (post-Dyna, dyna_r1_5050b)
# Identical recipe both sides; all evals SEQUENTIAL + osmesa (solo-eval rule).
# Reference distributions already measured (6 seeds each):
#   pre-Dyna  86.7  (88.7/87.3/84.0 clean seeds; historical ref 87.6)
#   post-Dyna 94.5  (94.0/93.3/98.7/92.0/94.0/94.7)
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel; PLAN=$CODE/scripts/plan
L=/workspace/logs; SUM=/workspace/results/summary_final678.csv; DRV=$L/driver_final678.log
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
mkdir -p /workspace/results; touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

# tag  wm  td  fs1  fs5  gpu  seed
train_one(){
  local tag=$1 wm=$2 td=$3 fs1=$4 fs5=$5 gpu=$6 seed=$7
  local out="/workspace/actors/lip4_${tag}_s${seed}.pt"
  [ -f "$out" ] && { log "$tag s$seed cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 14400 python3 "$PLAN/train_lip_ac.py" \
    --cache "$fs5" --cache-td "$fs1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$seed" \
    --out "$out" --out-value "/workspace/metrics/lip4_${tag}_s${seed}_value.pt" \
    > "$L/train_lip4_${tag}_s${seed}.log" 2>&1 || { log "$tag s$seed TRAIN FAILED"; return 1; }
  log "$tag s$seed trained ($(grep -oE 'E_final [0-9.]+' "$L/train_lip4_${tag}_s${seed}.log" | tail -1))"
}

eval_one(){ # tag wm seed draw
  local tag=$1 wm=$2 seed=$3 d=$4
  local n="${tag}_s${seed}_e${d}"
  local c; c=$(sc "$n"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "$n cached ($c)"; return 0; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$PLAN/eval_wm.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 policy="$wm" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_${tag}_s${seed}.pt" \
    output.filename="${n}.txt" > "$L/eval_${n}.log" 2>&1
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${n}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${n},${sr:-FAIL}" >> "$SUM"; log "$n = ${sr:-FAIL}"
}

m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }

# ---------------- Phase A: pre-Dyna (v2WM)
log "### PHASE A — LIPv4 on v2WM, seeds 6/7/8 ###"
train_one lipv2 /workspace/models/v2WM /workspace/metrics/v2_expert_TD.pt \
  /workspace/caches/v2_expert_fs1.pt /workspace/caches/v2_expert_fs5.pt 0 6 &
train_one lipv2 /workspace/models/v2WM /workspace/metrics/v2_expert_TD.pt \
  /workspace/caches/v2_expert_fs1.pt /workspace/caches/v2_expert_fs5.pt 1 7 &
train_one lipv2 /workspace/models/v2WM /workspace/metrics/v2_expert_TD.pt \
  /workspace/caches/v2_expert_fs1.pt /workspace/caches/v2_expert_fs5.pt 2 8 &
wait
for s in 6 7 8; do for d in 42 43 44; do eval_one lipv2 /workspace/models/v2WM "$s" "$d"; done; done

# ---------------- Phase B: post-Dyna (WM1)
log "### PHASE B — LIPv4 on WM1 (dyna_r1_5050b), seeds 6/7/8 ###"
train_one dyna /workspace/models/dyna_r1_5050b /workspace/metrics/r1w_expert_TD.pt \
  /workspace/caches/r1w_expert_fs1.pt /workspace/caches/r1w_expert_fs5.pt 0 6 &
train_one dyna /workspace/models/dyna_r1_5050b /workspace/metrics/r1w_expert_TD.pt \
  /workspace/caches/r1w_expert_fs1.pt /workspace/caches/r1w_expert_fs5.pt 1 7 &
train_one dyna /workspace/models/dyna_r1_5050b /workspace/metrics/r1w_expert_TD.pt \
  /workspace/caches/r1w_expert_fs1.pt /workspace/caches/r1w_expert_fs5.pt 2 8 &
wait
for s in 6 7 8; do for d in 42 43 44; do eval_one dyna /workspace/models/dyna_r1_5050b "$s" "$d"; done; done

log "=== FINAL RUN (seeds 6/7/8, h25, draws 42/43/44) ==="
for s in 6 7 8; do
  log "LIP  v2WM s$s: $(sc lipv2_s${s}_e42)/$(sc lipv2_s${s}_e43)/$(sc lipv2_s${s}_e44) -> $(m3 "$(sc lipv2_s${s}_e42)" "$(sc lipv2_s${s}_e43)" "$(sc lipv2_s${s}_e44)")"
done
for s in 6 7 8; do
  log "DYNA WM1  s$s: $(sc dyna_s${s}_e42)/$(sc dyna_s${s}_e43)/$(sc dyna_s${s}_e44) -> $(m3 "$(sc dyna_s${s}_e42)" "$(sc dyna_s${s}_e43)" "$(sc dyna_s${s}_e44)")"
done
log "FINAL678_DONE"
