#!/bin/bash
# Dyna stage-1 on v2WM (pod A): expert caches -> TD -> LIPv4 x3, then push
# artifacts to pod B and trigger its eval ladder (run_evals_b.sh, 2-way).
# NO evals on pod A (concurrency lesson; anchor eval runs separately).
# v2 references: latent+CEM 80/84/62=75.3 | TD+CEM 79.3 | LIPv4-6k 87.6 | div ~20.9
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
TRM=$CODE/scripts/trm
LOGS=/workspace/logs
PY=python3
WM=/workspace/models/v2WM
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
FS1=/workspace/caches/v2_expert_fs1.pt
FS5=/workspace/caches/v2_expert_fs5.pt
TD=/workspace/metrics/v2_expert_TD.pt
DRV=$LOGS/driver_stage1v2.log
BSSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i /root/.ssh/dyna_ab -p 11346 root@157.66.254.39"
TRAIN_TIMEOUT=28800
mkdir -p $LOGS /workspace/caches /workspace/metrics /workspace/actors /workspace/swm_home

log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }

log "=== stage-1 v2 driver start (pid $$) ==="

# 0. action h5 (expert lance -> action+ep_offset+ep_len)
if [ ! -f "$AH5" ]; then
  $PY /workspace/build_action_h5.py /workspace/datasets/ogb_cube_single "$AH5" >> "$DRV" 2>&1 || die "action h5 build failed"
fi
log "action h5 ready"

# 1. caches on gpu1 (gpu0 runs the anchor eval until ~07:40)
if [ ! -f "$FS5" ]; then
  if [ ! -f "$FS1" ]; then
    log "caching fs1 (expert lance, v2WM encoder, gpu1)"
    CUDA_VISIBLE_DEVICES=1 timeout $TRAIN_TIMEOUT $PY "$TRM/cache_latents.py" \
      --wm "$WM" --dataset "$EXPERT" --out "$FS1" --state-key privileged_block_0_pos \
      > "$LOGS/cache_v2_fs1.log" 2>&1 || die "fs1 cache failed"
  fi
  $PY "$TRM/subsample_cache.py" --in "$FS1" --out "$FS5" --frameskip 5 \
    > "$LOGS/cache_v2_fs5.log" 2>&1 || die "fs5 subsample failed"
fi
log "caches ready"

# 2. TD teacher (gpu1)
if [ ! -f "$TD" ]; then
  log "TD teacher (td/quasimetric tau.03 n50 6k, gpu1)"
  CUDA_VISIBLE_DEVICES=1 timeout $TRAIN_TIMEOUT $PY "$PLAN/train_metric.py" \
    --cache "$FS1" --learner td --head quasimetric \
    --expectile 0.03 --n-step 50 --steps 6000 --seed 0 \
    --out "$TD" > "$LOGS/td_v2.log" 2>&1 || die "TD training failed"
fi
log "TD ready"

# 3. LIPv4 x3 (gpu0/2/3 — gpu0 is free once the anchor eval finishes)
lip_train(){ # gpu seed
  local gpu=$1 seed=$2
  local out="/workspace/actors/lip4_v2_s${seed}.pt"
  [ -f "$out" ] && { log "lip4_v2_s${seed}: cached"; return 0; }
  log "lip4_v2_s${seed}: train start (gpu${gpu})"
  CUDA_VISIBLE_DEVICES=$gpu timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" \
    --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 \
    --seed "$seed" --out "$out" --out-value "/workspace/metrics/lip4_v2_s${seed}_value.pt" \
    > "$LOGS/train_lip4_v2_s${seed}.log" 2>&1 \
    || { log "lip4_v2_s${seed}: TRAIN FAILED"; return 1; }
  log "lip4_v2_s${seed}: done"
}
lip_train 0 0 &
lip_train 2 1 &
lip_train 3 2 &
wait
A0=/workspace/actors/lip4_v2_s0.pt; A1=/workspace/actors/lip4_v2_s1.pt; A2=/workspace/actors/lip4_v2_s2.pt
[ -f $A0 ] || die "actor s0 missing"
[ -f $A1 ] || die "actor s1 missing"
[ -f $A2 ] || die "actor s2 missing"
log "3 actors trained"

# 4. push artifacts to pod B + trigger its ladder (evals live on B)
tar czf - -C /workspace metrics/v2_expert_TD.pt actors/lip4_v2_s0.pt actors/lip4_v2_s1.pt actors/lip4_v2_s2.pt \
  | $BSSH "tar xzf - -C /workspace" || die "artifact push to B failed"
$BSSH "mkdir -p /workspace/probe_v2_base && setsid bash /workspace/run_evals_b.sh v2base /workspace/models/v2WM -a /workspace/actors/lip4_v2_s0.pt,/workspace/actors/lip4_v2_s1.pt,/workspace/actors/lip4_v2_s2.pt -m /workspace/metrics/v2_expert_TD.pt -p /workspace/probe_v2_base < /dev/null > /dev/null 2>&1 &" \
  || die "B ladder trigger failed"
log "pod B ladder triggered (tag v2base, 15 evals, ~4.5h)"
log "STAGE1V2_DONE"
