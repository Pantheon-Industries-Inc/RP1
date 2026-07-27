#!/bin/bash
# Pre-Dyna LIP baseline under the WINNING recipe: amax 2.2 (vs canonical 3.5).
# 3 seeds x 3 draws on v2WM. Seed 0 already done (lip4_fix_amax22.pt -> 84.0),
# so this trains seeds 1 and 2 and evaluates them, then prints the 3-seed card.
#
# Reference (amax 3.5, same seeds/draws): s0 68.0 | s1 81.3 | s2 88.7 -> 79.3
# Seed-0 arms:  batch256 66.7 | lowlr 80.0 | iters16 80.0 | amax2.2 84.0 (best)
# Open question this answers: does amax 2.2 lift the MEAN, or only rescue the
# collapsed seed while costing the already-good ones (81.3 / 88.7)?
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/workspace/code/stable-worldmodel/scripts/plan
L=/workspace/logs; SUM=/workspace/results/summary_amax22.csv; DRV=$L/driver_amax22_3seed.log
WM=/workspace/models/v2WM; TD=/workspace/metrics/v2_expert_TD.pt
FS1=/workspace/caches/v2_expert_fs1.pt; FS5=/workspace/caches/v2_expert_fs5.pt
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
mkdir -p /workspace/results; touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd /workspace/code/stable-worldmodel

train_seed(){ # gpu seed
  local gpu=$1 seed=$2
  local out=/workspace/actors/lip4_amax22_s${seed}.pt
  [ -f "$out" ] && { log "s$seed cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
    --horizon 5 --iters 8 --steps 6000 --amax 2.2 --n-step 50 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$seed" \
    --out "$out" --out-value /workspace/metrics/lip4_amax22_s${seed}_value.pt \
    > "$L/train_amax22_s${seed}.log" 2>&1 || { log "s$seed TRAIN FAILED"; return 1; }
  log "s$seed trained (E_final $(grep -oE 'E_final [0-9.]+' "$L/train_amax22_s${seed}.log" | tail -1 | awk '{print $2}'))"
}

log "### amax 2.2 baseline: training seeds 1,2 on gpu1,gpu2 (s0 already done = 84.0) ###"
train_seed 1 1 &
train_seed 2 2 &
wait

# evals are solo: wait out the combo arm and anything else still evaluating
log "waiting for the combo arm + a clear eval lane"
for i in $(seq 1 180); do grep -q FIXS0COMBO_DONE "$L/driver_fixs0combo.log" 2>/dev/null && break; sleep 60; done
while pgrep -f "eval_w[m].py" > /dev/null; do sleep 45; done
sleep 15

for s in 1 2; do
  [ -f /workspace/actors/lip4_amax22_s${s}.pt ] || { log "s$s actor missing, skip"; continue; }
  for d in 42 43 44; do
    n="amax22_s${s}_e${d}"
    c=$(sc "$n"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "$n cached ($c)"; continue; }
    CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube seed=$d \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 \
      eval.eval_budget=50 policy="$WM" solver=lip \
      "solver.actor_path=/workspace/actors/lip4_amax22_s${s}.pt" \
      output.filename="${n}.txt" > "$L/eval_${n}.log" 2>&1
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${n}.log" | tail -1 | grep -oE "[0-9.]+$")
    echo "${n},${sr:-FAIL}" >> "$SUM"; log "$n = ${sr:-FAIL}"
  done
done

m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
S0=84.0   # 84/94/74, already measured
S1=$(m3 "$(sc amax22_s1_e42)" "$(sc amax22_s1_e43)" "$(sc amax22_s1_e44)")
S2=$(m3 "$(sc amax22_s2_e42)" "$(sc amax22_s2_e43)" "$(sc amax22_s2_e44)")
log "=== PRE-DYNA @ amax 2.2, 3 seeds x 3 draws (vs amax 3.5: 68.0 / 81.3 / 88.7 -> 79.3) ==="
log "  s0: 84.0/94.0/74.0 -> 84.0   (was 68.0)"
log "  s1: $(sc amax22_s1_e42)/$(sc amax22_s1_e43)/$(sc amax22_s1_e44) -> $S1   (was 81.3)"
log "  s2: $(sc amax22_s2_e42)/$(sc amax22_s2_e43)/$(sc amax22_s2_e44) -> $S2   (was 88.7)"
log "  3-seed mean: $(awk -v a=$S0 -v b="$S1" -v c="$S2" 'BEGIN{if(b=="NA"||c=="NA"){print "NA"}else{printf "%.1f",(a+b+c)/3}}')   (was 79.3)"
log "AMAX22_3SEED_DONE"
