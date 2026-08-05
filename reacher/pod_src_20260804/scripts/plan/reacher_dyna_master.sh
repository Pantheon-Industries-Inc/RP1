#!/bin/bash
# Reacher DYNA pod master (87.120.211.204:19342, 2xH100): stage 0+1.
# Gates on the 10k collection, validates the 4.49-passthrough conversion on
# the proto dataset first, then per GPU:
#   GPU0 lejepa / GPU1 pldm: phase1 -> LIPv4 amax 2.2 x 3 seeds + cards
#     (sweep skipped: 2.2 won on both WMs in the first reacher campaign;
#      those baselines/cards carry over — same seed-3072 dataset draw).
# Divergence probe + Dyna round-1 are driven interactively after this.
set -u
LOGS=/workspace/logs
PLAN=/workspace/stable-worldmodel/scripts/plan
MASTER=$LOGS/reacher_dyna_master.log
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][master] $*" | tee -a "$MASTER"; }

export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export MUJOCO_GL=osmesa

log "waiting for proto collection"
while tmux has-session -t rproto 2>/dev/null; do sleep 60; done
grep -q "Completed random data collection" "$LOGS/collect_proto.log" \
  || { log "FATAL: proto collection failed"; exit 1; }

log "proto h5 + conversion validation (transformers-4.49 passthrough)"
PROTO=/workspace/proto/datasets/dmc/reacher_random.lance
python3 /workspace/stable-worldmodel/scripts/data/build_reacher_h5.py \
  --dataset "$PROTO" --train-out /workspace/proto/proto_train.h5 \
  --eval-out /workspace/proto/proto_eval.h5 --eval-episodes 200 \
  > "$LOGS/proto_h5.log" 2>&1 || { log "FATAL: proto h5 failed"; exit 1; }
CUDA_VISIBLE_DEVICES=0 python3 "$PLAN/convert_reacher_bases.py" \
  --only lejepa,pldm --dataset "$PROTO" --h5 /workspace/proto/proto_train.h5 \
  > "$LOGS/convert_proto.log" 2>&1 \
  || { log "FATAL: conversion failed (see convert_proto.log)"; exit 1; }
grep -E "parity|ratio|wrote|renames" "$LOGS/convert_proto.log" | tee -a "$MASTER"
log "conversion validated"

log "waiting for main collection"
while tmux has-session -t rcollect 2>/dev/null; do sleep 120; done
grep -q "Completed random data collection" "$LOGS/collect_reacher.log" \
  || { log "FATAL: main collection failed"; exit 1; }
log "main collection done"

chain(){ # wm gpu
  local wm=$1 gpu=$2
  log "chain ${wm} gpu${gpu}: phase1"
  bash "$PLAN/reacher_phase1.sh" "$wm" "$gpu" || { log "chain ${wm}: phase1 FAILED"; return 1; }
  log "chain ${wm} gpu${gpu}: LIPv4 amax 2.2 x 3 seeds"
  AMAX_ARMS="2.2" bash "$PLAN/run_reacher_lip4.sh" "$wm" "$gpu" \
    || log "chain ${wm}: lip4 driver returned nonzero"
  log "chain ${wm}: DONE"
}

chain lejepa 0 > "$LOGS/chain_lejepa.log" 2>&1 &
chain pldm 1 > "$LOGS/chain_pldm.log" 2>&1 &
wait
log "MASTER DONE — next: divergence probe (LIP_PROBE_DIR) + Dyna round-1"
