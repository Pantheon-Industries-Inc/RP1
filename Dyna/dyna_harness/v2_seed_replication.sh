#!/bin/bash
# Replicate the historical 87.6 pre-Dyna baseline on v2WM.
#
# Historical reference (ftmisses campaign, arm `v2neg0`): v2 WM, no negatives,
# v4 gate-free actor, schedamax-6k, h25, draws s42/43/44
#   -> seeds 86.7 / 88.0 / 88.0 = 87.6   (spread 1.3)
# Our stage-1 re-measurement, same recipe:
#   -> a0 68.0 / a1 81.3 / a2 88.7 = 79.3 (spread 20.7)  <-- suspicious
#
# 3 samples can't separate "unlucky seeds" from "noisier pipeline", so train 3
# FRESH seeds (3,4,5) and pool with the existing 3 for a 6-seed distribution.
# Rebuilds the v2 caches first (they were cleaned up; TD teacher + model survive).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel; PLAN=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; SUM=/workspace/results/summary_v2rep.csv; DRV=$L/driver_v2rep.log
WM=/workspace/models/v2WM
TD=/workspace/metrics/v2_expert_TD.pt
FS1=/workspace/caches/v2_expert_fs1.pt
FS5=/workspace/caches/v2_expert_fs5.pt
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
mkdir -p /workspace/results; touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

# ---- 1. rebuild v2 caches
if [ ! -f "$FS5" ]; then
  if [ ! -f "$FS1" ]; then
    log "rebuilding v2 fs1 cache (expert lance, v2WM encoder) ~30min"
    CUDA_VISIBLE_DEVICES=0 timeout 14400 python3 "$TRM/cache_latents.py" --wm "$WM" \
      --dataset "$EXPERT" --out "$FS1" --state-key privileged_block_0_pos \
      > "$L/cache_v2rep_fs1.log" 2>&1 || { log "fs1 FAILED"; tail -3 "$L/cache_v2rep_fs1.log"; exit 1; }
  fi
  python3 "$TRM/subsample_cache.py" --in "$FS1" --out "$FS5" --frameskip 5 \
    > "$L/cache_v2rep_fs5.log" 2>&1 || { log "fs5 FAILED"; exit 1; }
fi
log "v2 caches ready"

# ---- 2. three fresh actor seeds, identical recipe to stage-1 / historical
train_seed(){
  local gpu=$1 seed=$2
  local out="/workspace/actors/lip4_v2rep_s${seed}.pt"
  [ -f "$out" ] && { log "seed $seed cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 14400 python3 "$PLAN/train_lip_ac.py" \
    --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$seed" \
    --out "$out" --out-value "/workspace/metrics/lip4_v2rep_s${seed}_value.pt" \
    > "$L/train_lip4_v2rep_s${seed}.log" 2>&1 \
    || { log "seed $seed TRAIN FAILED"; return 1; }
  log "seed $seed trained ($(grep -oE 'E_final [0-9.]+' "$L/train_lip4_v2rep_s${seed}.log" | tail -1))"
}
log "training seeds 3,4,5 in parallel (gpu 0/1/2)"
train_seed 0 3 & train_seed 1 4 & train_seed 2 5 &
wait

# ---- 3. evaluate: 3 seeds x 3 draws, h25 (sequential — solo-eval rule)
for s in 3 4 5; do
  A=/workspace/actors/lip4_v2rep_s${s}.pt
  [ -f "$A" ] || { log "seed $s actor missing, skipping"; continue; }
  for d in 42 43 44; do
    n="v2rep_s${s}_e${d}"
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
log "=== v2WM SEED REPLICATION (historical: 86.7 / 88.0 / 88.0 = 87.6) ==="
log "existing  a0 68.0 | a1 81.3 | a2 88.7  (stage-1)"
for s in 3 4 5; do
  log "new seed $s: $(sc v2rep_s${s}_e42)/$(sc v2rep_s${s}_e43)/$(sc v2rep_s${s}_e44) -> $(m3 "$(sc v2rep_s${s}_e42)" "$(sc v2rep_s${s}_e43)" "$(sc v2rep_s${s}_e44)")"
done
log "V2REP_DONE"
