#!/bin/bash
# PHASE 1 (v2) — fix the seed-0 actor collapse on v2WM. Target: s0 >= 85 (from 68.0).
# Symptom: E_final plateaus 7-10 and oscillates for all 6k steps (final 7.82) while
# seeds 1/2 descend to ~3.1; the resulting policy is uniformly weaker (its failures
# are a near-superset of a good seed's). NO BC (standing constraint).
#
# Arms, in the requested priority order — all on seed 0, else canonical recipe:
#   A batch256  — batch 128->256 (oscillation looks like gradient noise; averaging
#                 over more samples is the standard remedy).  [priority 1]
#   B lowlr     — actor-lr 3e-4->1e-4, final 3e-5->1e-5 (if it overshoots). [2]
#   C iters16   — 8->16 inner action-optimization steps.                    [3]
# Whichever wins is applied UNIFORMLY to seeds 0-4 in phase 2.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/workspace/code/stable-worldmodel/scripts/plan
L=/workspace/logs; SUM=/workspace/results/summary_fixs0.csv; DRV=$L/driver_fixs0v2.log
WM=/workspace/models/v2WM; TD=/workspace/metrics/v2_expert_TD.pt
FS1=/workspace/caches/v2_expert_fs1.pt; FS5=/workspace/caches/v2_expert_fs5.pt
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
mkdir -p /workspace/results; touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd /workspace/code/stable-worldmodel

run_variant(){ # gpu tag extra-args...
  local gpu=$1 tag=$2; shift 2
  local out=/workspace/actors/lip4_fix_${tag}.pt
  [ -f "$out" ] && { log "$tag cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
    --horizon 5 --n-step 50 --amax 3.5 --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 --arch v4 --seed 0 "$@" \
    --out "$out" --out-value /workspace/metrics/lip4_fix_${tag}_value.pt \
    > "$L/train_fix_${tag}.log" 2>&1 || { log "$tag TRAIN FAILED"; return 1; }
  log "$tag trained (E_final $(grep -oE 'E_final [0-9.]+' "$L/train_fix_${tag}.log" | tail -1 | awk '{print $2}'))"
}

log "### PHASE 1 v2: seed-0 fixes — baseline s0 = 68.0 (E_final 7.82), target >= 85 ###"
run_variant 0 batch256 --iters 8  --steps 6000 --actor-lr 3e-4 --actor-lr-final 3e-5 --batch 256 &
run_variant 1 lowlr    --iters 8  --steps 6000 --actor-lr 1e-4 --actor-lr-final 1e-5 &
run_variant 2 iters16  --iters 16 --steps 6000 --actor-lr 3e-4 --actor-lr-final 3e-5 &
wait

for tag in batch256 lowlr iters16; do
  [ -f /workspace/actors/lip4_fix_${tag}.pt ] || { log "$tag actor missing, skip"; continue; }
  for d in 42 43 44; do
    n="fix_${tag}_e${d}"
    c=$(sc "$n"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "$n cached ($c)"; continue; }
    CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube seed=$d \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 \
      eval.eval_budget=50 policy="$WM" solver=lip \
      "solver.actor_path=/workspace/actors/lip4_fix_${tag}.pt" output.filename="${n}.txt" \
      > "$L/eval_${n}.log" 2>&1
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${n}.log" | tail -1 | grep -oE "[0-9.]+$")
    echo "${n},${sr:-FAIL}" >> "$SUM"; log "$n = ${sr:-FAIL}"
  done
done

m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
log "=== SEED-0 FIX RESULTS (baseline 68.0 | good seeds 81-89 | target >=85) ==="
for tag in batch256 lowlr iters16; do
  log "$tag: $(sc fix_${tag}_e42)/$(sc fix_${tag}_e43)/$(sc fix_${tag}_e44) -> $(m3 "$(sc fix_${tag}_e42)" "$(sc fix_${tag}_e43)" "$(sc fix_${tag}_e44)")"
done
log "FIXS0V2_DONE"
