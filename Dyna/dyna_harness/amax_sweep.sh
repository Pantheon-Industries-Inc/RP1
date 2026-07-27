#!/bin/bash
# amax (boundary-clip) sweep on v2WM, h25, draws 42/43/44.
#
# Known points (3 seeds each):  amax 3.5 -> s0 68.0 s1 81.3 s2 88.7 = 79.3
#                               amax 2.2 -> s0 84.0 s1 84.7 s2 88.7 = 85.8
# Expect an inverted-U: too loose lets the actor push into the WM's fabrication
# regime (s0 collapses); too tight should starve it of authority, since expert
# action tails reach ~3.5 in z-scored units.
#
# Sweep NEW values {1.4, 1.8, 2.6, 3.0} on TWO seeds:
#   seed 0 = the fragile one (collapses at 3.5) -> does the clip fix the floor?
#   seed 2 = the robust one  (88.7 at both 3.5 and 2.2) -> does it cost the ceiling?
# 8 trainings in 2 rounds of 4 (OMP=8, <=4 concurrent — pid-quota rule), then
# 24 evals strictly sequential (solo-eval rule).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/workspace/code/stable-worldmodel/scripts/plan
L=/workspace/logs; SUM=/workspace/results/summary_amaxsweep.csv; DRV=$L/driver_amaxsweep.log
WM=/workspace/models/v2WM; TD=/workspace/metrics/v2_expert_TD.pt
FS1=/workspace/caches/v2_expert_fs1.pt; FS5=/workspace/caches/v2_expert_fs5.pt
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
mkdir -p /workspace/results; touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd /workspace/code/stable-worldmodel

train_one(){ # gpu amax seed
  local gpu=$1 am=$2 seed=$3
  local tag="a${am/./}_s${seed}"
  local out=/workspace/actors/lip4_amsw_${tag}.pt
  [ -f "$out" ] && { log "$tag cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
    --horizon 5 --iters 8 --steps 6000 --amax "$am" --n-step 50 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$seed" \
    --out "$out" --out-value /workspace/metrics/lip4_amsw_${tag}_value.pt \
    > "$L/train_amsw_${tag}.log" 2>&1 || { log "$tag TRAIN FAILED"; return 1; }
  log "$tag trained (E_final $(grep -oE 'E_final [0-9.]+' "$L/train_amsw_${tag}.log" | tail -1 | awk '{print $2}'))"
}

log "### amax sweep: round 1/2 — {1.4,1.8} x seeds {0,2} ###"
train_one 0 1.4 0 & train_one 1 1.4 2 & train_one 2 1.8 0 & train_one 3 1.8 2 &
wait
log "### amax sweep: round 2/2 — {2.6,3.0} x seeds {0,2} ###"
train_one 0 2.6 0 & train_one 1 2.6 2 & train_one 2 3.0 0 & train_one 3 3.0 2 &
wait

log "### evals (24, sequential) ###"
while pgrep -f "eval_w[m].py" > /dev/null; do sleep 45; done
for am in 1.4 1.8 2.6 3.0; do
  for s in 0 2; do
    tag="a${am/./}_s${s}"
    [ -f /workspace/actors/lip4_amsw_${tag}.pt ] || { log "$tag missing, skip"; continue; }
    for d in 42 43 44; do
      n="amsw_${tag}_e${d}"
      c=$(sc "$n"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "$n cached ($c)"; continue; }
      CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube seed=$d \
        eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 \
        eval.eval_budget=50 policy="$WM" solver=lip \
        "solver.actor_path=/workspace/actors/lip4_amsw_${tag}.pt" \
        output.filename="${n}.txt" > "$L/eval_${n}.log" 2>&1
      sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${n}.log" | tail -1 | grep -oE "[0-9.]+$")
      echo "${n},${sr:-FAIL}" >> "$SUM"; log "$n = ${sr:-FAIL}"
    done
  done
done

m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
log "=== AMAX SWEEP CARD (v2WM, h25, draws 42/43/44) ==="
log "  amax   seed0    seed2     [known: 2.2 -> 84.0 / 88.7 ; 3.5 -> 68.0 / 88.7]"
for am in 1.4 1.8 2.6 3.0; do
  t0="a${am/./}_s0"; t2="a${am/./}_s2"
  log "  $am    $(m3 "$(sc amsw_${t0}_e42)" "$(sc amsw_${t0}_e43)" "$(sc amsw_${t0}_e44)")     $(m3 "$(sc amsw_${t2}_e42)" "$(sc amsw_${t2}_e43)" "$(sc amsw_${t2}_e44)")"
done
log "AMAX_SWEEP_DONE"
