#!/bin/bash
# amax fine sweep 1.0-1.8 (0.2 spacing) x 3 seeds x 3 draws, v2WM, h25.
#
# Prior coarse sweep (seeds 0/2 only):
#   amax  1.4   1.8   2.2   2.6   3.0   3.5
#   s0   86.7  85.3  84.0  78.7  67.3  68.0   <- monotone: tighter clip, better
#   s2   87.3  90.0  88.7  86.7   ~70  88.7   <- turns over below 1.8
# => optimum near 1.8; this resolves the 1.0-1.8 region on ALL THREE seeds.
#
# Idempotent by design: reuses the SAME actor naming and the SAME summary CSV as
# the coarse sweep, so the four already-trained actors (1.4/1.8 x s0/s2) and their
# 12 eval cells are skipped. 11 new trainings, 33 new evals.
# OMP=8, <=4 concurrent trainings (pid-quota rule); evals strictly sequential.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/workspace/code/stable-worldmodel/scripts/plan
L=/workspace/logs; SUM=/workspace/results/summary_amaxsweep.csv; DRV=$L/driver_amaxsweep2.log
WM=/workspace/models/v2WM; TD=/workspace/metrics/v2_expert_TD.pt
FS1=/workspace/caches/v2_expert_fs1.pt; FS5=/workspace/caches/v2_expert_fs5.pt
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
mkdir -p /workspace/results; touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd /workspace/code/stable-worldmodel

AMAXES="1.0 1.2 1.4 1.6 1.8"
SEEDS="0 1 2"

train_one(){ # gpu amax seed
  local gpu=$1 am=$2 seed=$3
  local tag="a${am/./}_s${seed}"
  local out=/workspace/actors/lip4_amsw_${tag}.pt
  [ -f "$out" ] && { log "$tag already trained, reusing"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
    --horizon 5 --iters 8 --steps 6000 --amax "$am" --n-step 50 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$seed" \
    --out "$out" --out-value /workspace/metrics/lip4_amsw_${tag}_value.pt \
    > "$L/train_amsw_${tag}.log" 2>&1 || { log "$tag TRAIN FAILED"; return 1; }
  log "$tag trained (E_final $(grep -oE 'E_final [0-9.]+' "$L/train_amsw_${tag}.log" | tail -1 | awk '{print $2}'))"
}

# build the job list, skipping anything already trained
JOBS=""
for am in $AMAXES; do for s in $SEEDS; do
  [ -f "/workspace/actors/lip4_amsw_a${am/./}_s${s}.pt" ] || JOBS="$JOBS ${am}:${s}"
done; done
set -- $JOBS
log "### fine sweep: $# trainings needed (4 at a time) ###"
n=0
while [ $# -gt 0 ]; do
  for gpu in 0 1 2 3; do
    [ $# -eq 0 ] && break
    j=$1; shift
    train_one "$gpu" "${j%%:*}" "${j##*:}" &
    n=$((n+1))
  done
  wait
  log "  batch done ($n trained so far)"
done

log "### evals (sequential) ###"
while pgrep -f "eval_w[m].py" > /dev/null; do sleep 45; done
for am in $AMAXES; do for s in $SEEDS; do
  tag="a${am/./}_s${s}"
  [ -f /workspace/actors/lip4_amsw_${tag}.pt ] || { log "$tag actor missing, skip"; continue; }
  for d in 42 43 44; do
    nm="amsw_${tag}_e${d}"
    c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "$nm cached ($c)"; continue; }
    CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube seed=$d \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 \
      eval.eval_budget=50 policy="$WM" solver=lip \
      "solver.actor_path=/workspace/actors/lip4_amsw_${tag}.pt" \
      output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
    echo "${nm},${sr:-FAIL}" >> "$SUM"; log "$nm = ${sr:-FAIL}"
  done
done; done

m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
log "=== AMAX FINE-SWEEP CARD (v2WM, h25, draws 42/43/44) ==="
log "  amax    s0     s1     s2     3-seed mean"
for am in $AMAXES; do
  A=$(m3 "$(sc amsw_a${am/./}_s0_e42)" "$(sc amsw_a${am/./}_s0_e43)" "$(sc amsw_a${am/./}_s0_e44)")
  B=$(m3 "$(sc amsw_a${am/./}_s1_e42)" "$(sc amsw_a${am/./}_s1_e43)" "$(sc amsw_a${am/./}_s1_e44)")
  C=$(m3 "$(sc amsw_a${am/./}_s2_e42)" "$(sc amsw_a${am/./}_s2_e43)" "$(sc amsw_a${am/./}_s2_e44)")
  log "  $am   $A   $B   $C    $(awk -v a="$A" -v b="$B" -v c="$C" 'BEGIN{if(a=="NA"||b=="NA"||c=="NA"){print "NA"}else{printf "%.1f",(a+b+c)/3}}')"
done
log "  refs @2.2: 84.0 / 84.7 / 88.7 -> 85.8   |   @3.5: 68.0 / 81.3 / 88.7 -> 79.3"
log "AMAX_SWEEP2_DONE"
