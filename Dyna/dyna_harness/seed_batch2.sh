#!/bin/bash
# Pre-committed second batch: seeds 6,7,8 on BOTH sides, identical recipe and
# protocol (schedamax-6k v4, h25, draws 42/43/44, sequential osmesa evals).
# ALL seeds are reported — no selection on outcome.
#
# Running totals after this batch:
#   pre-Dyna  (v2WM) clean seeds 3,4,5 (88.7/87.3/84.0) + 6,7,8  -> 6 seeds
#   post-Dyna (WM1)  seeds 0-5 (94.0/93.3/98.7/92.0/94.0/94.7)   -> 9 seeds
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel; PLAN=$CODE/scripts/plan
L=/workspace/logs; DRV=$L/driver_batch2.log
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
cd "$CODE"

# arm: tag  wm  td  fs1  fs5  summary
train_one(){ # gpu seed tag wm td fs1 fs5
  local gpu=$1 seed=$2 tag=$3 wm=$4 td=$5 fs1=$6 fs5=$7
  local out="/workspace/actors/lip4_${tag}_s${seed}.pt"
  [ -f "$out" ] && { log "$tag s$seed cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 14400 python3 "$PLAN/train_lip_ac.py" \
    --cache "$fs5" --cache-td "$fs1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$seed" \
    --out "$out" --out-value "/workspace/metrics/lip4_${tag}_s${seed}_value.pt" \
    > "$L/train_lip4_${tag}_s${seed}.log" 2>&1 || { log "$tag s$seed TRAIN FAILED"; return 1; }
  log "$tag s$seed trained"
}

eval_one(){ # seed tag wm summary
  local seed=$1 tag=$2 wm=$3 sum=$4
  local A=/workspace/actors/lip4_${tag}_s${seed}.pt
  [ -f "$A" ] || { log "$tag s$seed missing"; return 1; }
  for d in 42 43 44; do
    local n="${tag}_s${seed}_e${d}"
    grep -q "^${n}," "$sum" 2>/dev/null && { log "$n cached"; continue; }
    CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$PLAN/eval_wm.py" --config-name cube \
      seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 policy="$wm" solver=lip \
      "solver.actor_path=$A" output.filename="${n}.txt" > "$L/eval_${n}.log" 2>&1
    local sr
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${n}.log" | tail -1 | grep -oE "[0-9.]+$")
    echo "${n},${sr:-FAIL}" >> "$sum"; log "$n = ${sr:-FAIL}"
  done
}

V2WM=/workspace/models/v2WM
V2TD=/workspace/metrics/v2_expert_TD.pt
V2F1=/workspace/caches/v2_expert_fs1.pt; V2F5=/workspace/caches/v2_expert_fs5.pt
D1WM=/workspace/models/dyna_r1_5050b
D1TD=/workspace/metrics/r1w_expert_TD.pt
D1F1=/workspace/caches/r1w_expert_fs1.pt; D1F5=/workspace/caches/r1w_expert_fs5.pt
SV=/workspace/results/summary_v2rep.csv
SD=/workspace/results/summary_dynarep.csv

log "=== batch2: training pre-Dyna seeds 6,7,8 (gpu 0/1/2) ==="
train_one 0 6 v2rep "$V2WM" "$V2TD" "$V2F1" "$V2F5" &
train_one 1 7 v2rep "$V2WM" "$V2TD" "$V2F1" "$V2F5" &
train_one 2 8 v2rep "$V2WM" "$V2TD" "$V2F1" "$V2F5" &
wait
log "=== batch2: training post-Dyna seeds 6,7,8 (gpu 0/1/2) ==="
train_one 0 6 dynarep "$D1WM" "$D1TD" "$D1F1" "$D1F5" &
train_one 1 7 dynarep "$D1WM" "$D1TD" "$D1F1" "$D1F5" &
train_one 2 8 dynarep "$D1WM" "$D1TD" "$D1F1" "$D1F5" &
wait
log "=== batch2: evaluating (sequential) ==="
for s in 6 7 8; do eval_one $s v2rep   "$V2WM" "$SV"; done
for s in 6 7 8; do eval_one $s dynarep "$D1WM" "$SD"; done

m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
g(){ grep -h "^${1}," "$2" 2>/dev/null | tail -1 | cut -d, -f2; }
log "=== BATCH2 RESULTS (all seeds reported) ==="
for s in 6 7 8; do
  log "pre-Dyna  s$s: $(g v2rep_s${s}_e42 $SV)/$(g v2rep_s${s}_e43 $SV)/$(g v2rep_s${s}_e44 $SV) -> $(m3 "$(g v2rep_s${s}_e42 $SV)" "$(g v2rep_s${s}_e43 $SV)" "$(g v2rep_s${s}_e44 $SV)")"
done
for s in 6 7 8; do
  log "post-Dyna s$s: $(g dynarep_s${s}_e42 $SD)/$(g dynarep_s${s}_e43 $SD)/$(g dynarep_s${s}_e44 $SD) -> $(m3 "$(g dynarep_s${s}_e42 $SD)" "$(g dynarep_s${s}_e43 $SD)" "$(g dynarep_s${s}_e44 $SD)")"
done
log "BATCH2_DONE"
