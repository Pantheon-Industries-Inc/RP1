#!/bin/bash
# Round-1 ladder on the winning Dyna arm (WM1 = dyna_r1_5050b):
#   caches -> TD -> LIPv4 x3 (tandem) -> 15 evals (sequential) + A/B probe.
# Single-pod (loop pod, 4x H100). Idempotent via artifact files / summary rows.
# Compare against WM0 card: CEM 76.0 / TD+CEM 78.0 / LIP a2 88.7 / div 36.1
# and the no-Dyna LIP 3-seed spread (a0 ~62, a1 81.3, a2 88.7).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
TRM=$CODE/scripts/trm
LOGS=/workspace/logs
RES=/workspace/results
PY=python3
WM=/workspace/models/dyna_r1_5050b
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
FS1=/workspace/caches/r1w_expert_fs1.pt
FS5=/workspace/caches/r1w_expert_fs5.pt
TD=/workspace/metrics/r1w_expert_TD.pt
SUM=$RES/summary_r1w.csv
DRV=$LOGS/driver_r1_ladder.log
PROBE=/workspace/probe_r1w_new
TRAIN_TIMEOUT=28800
EVAL_TIMEOUT=7200
mkdir -p "$LOGS" "$RES" /workspace/caches /workspace/metrics /workspace/actors "$PROBE"
touch "$SUM"

log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }

run_eval(){ # name gpu seed extra...
  local name=$1 gpu=$2 seed=$3
  shift 3
  local c
  c=$(sc "$name")
  if [ -n "$c" ] && [ "$c" != "FAIL" ]; then log "eval ${name}: cached (${c})"; return 0; fi
  CUDA_VISIBLE_DEVICES=$gpu timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name cube seed="$seed" eval.dataset_name="$EXPERT" ++bf16=true \
    eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
    output.filename="${name}.txt" "$@" > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$SUM"; return 1; }
  echo "${name},${sr}" >> "$SUM"; log "eval ${name}: ${sr}"
}

log "=== r1 winner ladder start (pid $$, WM=$WM) ==="

# 1. caches (gpu0)
if [ ! -f "$FS5" ]; then
  if [ ! -f "$FS1" ]; then
    log "caching fs1 (expert lance, WM1 encoder, gpu0)"
    CUDA_VISIBLE_DEVICES=0 timeout $TRAIN_TIMEOUT $PY "$TRM/cache_latents.py" \
      --wm "$WM" --dataset "$EXPERT" --out "$FS1" --state-key privileged_block_0_pos \
      > "$LOGS/cache_r1w_fs1.log" 2>&1 || die "fs1 cache failed"
  fi
  $PY "$TRM/subsample_cache.py" --in "$FS1" --out "$FS5" --frameskip 5 \
    > "$LOGS/cache_r1w_fs5.log" 2>&1 || die "fs5 failed"
fi
log "caches ready"

# 2. TD (gpu0)
if [ ! -f "$TD" ]; then
  CUDA_VISIBLE_DEVICES=0 timeout $TRAIN_TIMEOUT $PY "$PLAN/train_metric.py" \
    --cache "$FS1" --learner td --head quasimetric \
    --expectile 0.03 --n-step 50 --steps 6000 --seed 0 \
    --out "$TD" > "$LOGS/td_r1w.log" 2>&1 || die "TD failed"
fi
log "TD ready"

# 3. LIP x3 (gpu 0/1/2)
lip_train(){
  local gpu=$1 seed=$2
  local out="/workspace/actors/lip4_r1w_s${seed}.pt"
  [ -f "$out" ] && { log "lip4_r1w_s${seed}: cached"; return 0; }
  log "lip4_r1w_s${seed}: train (gpu${gpu})"
  CUDA_VISIBLE_DEVICES=$gpu timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" \
    --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 \
    --seed "$seed" --out "$out" --out-value "/workspace/metrics/lip4_r1w_s${seed}_value.pt" \
    > "$LOGS/train_lip4_r1w_s${seed}.log" 2>&1 \
    || { log "lip4_r1w_s${seed}: TRAIN FAILED"; return 1; }
  log "lip4_r1w_s${seed}: done"
}
lip_train 0 0 &
lip_train 1 1 &
lip_train 2 2 &
wait
for s in 0 1 2; do
  [ -f "/workspace/actors/lip4_r1w_s${s}.pt" ] || die "actor s${s} missing"
done
log "3 actors trained"

# 4. evals — sequential (quiet pod), probe on the first LIP eval
run_eval cem_r1w_e42 0 42 policy="$WM" solver=cem
run_eval cem_r1w_e43 0 43 policy="$WM" solver=cem
run_eval cem_r1w_e44 0 44 policy="$WM" solver=cem
run_eval cemtd_r1w_e42 0 42 policy="$WM" solver=cem "+metric=$TD"
run_eval cemtd_r1w_e43 0 43 policy="$WM" solver=cem "+metric=$TD"
run_eval cemtd_r1w_e44 0 44 policy="$WM" solver=cem "+metric=$TD"
first=1
for a in 0 1 2; do
  for s in 42 43 44; do
    if [ "$first" = 1 ]; then
      LIP_PROBE_DIR=$PROBE run_eval "lip_r1w_a${a}_e${s}" 0 "$s" policy="$WM" solver=lip "solver.actor_path=/workspace/actors/lip4_r1w_s${a}.pt"
      first=0
    else
      run_eval "lip_r1w_a${a}_e${s}" 0 "$s" policy="$WM" solver=lip "solver.actor_path=/workspace/actors/lip4_r1w_s${a}.pt"
    fi
  done
done

# 5. new-actor divergence on WM1
$PY /workspace/ab_divergence.py "$PROBE" /workspace/metrics/lip4_r1w_s0_value.pt \
  > "$LOGS/div_r1w_new.log" 2>&1
log "new-actor divergence: $(grep DIVERGENCE $LOGS/div_r1w_new.log | head -1)"

# 6. summary
mean3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{ if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"} else {printf "%.1f", (a+b+c)/3} }'; }
log "=== R1 WINNER CARD (WM1 = 50/50 arm; WM0 refs CEM 76.0 / TD+CEM 78.0 / LIP-best 88.7 / div 36.1) ==="
log "latent+CEM : $(sc cem_r1w_e42)/$(sc cem_r1w_e43)/$(sc cem_r1w_e44) -> $(mean3 "$(sc cem_r1w_e42)" "$(sc cem_r1w_e43)" "$(sc cem_r1w_e44)")"
log "TD+CEM     : $(sc cemtd_r1w_e42)/$(sc cemtd_r1w_e43)/$(sc cemtd_r1w_e44) -> $(mean3 "$(sc cemtd_r1w_e42)" "$(sc cemtd_r1w_e43)" "$(sc cemtd_r1w_e44)")"
for a in 0 1 2; do
  log "LIP a${a}     : $(sc lip_r1w_a${a}_e42)/$(sc lip_r1w_a${a}_e43)/$(sc lip_r1w_a${a}_e44) -> $(mean3 "$(sc lip_r1w_a${a}_e42)" "$(sc lip_r1w_a${a}_e43)" "$(sc lip_r1w_a${a}_e44)")"
done
log "R1_LADDER_DONE"
