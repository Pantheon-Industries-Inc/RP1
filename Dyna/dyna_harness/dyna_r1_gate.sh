#!/bin/bash
# Dyna round-1 probe-gate. Waits for R1_FINETUNE_DONE, then per arm:
#   export model dir -> probe eval (old actor s0, LIP_PROBE_DIR) -> divergence
#   -> CEM canary (s42, WM0 ref 82).
# All evals SEQUENTIAL (quiet-pod rule). Verdict in logs/driver_gate.log.
# WM0 baselines: divergence 36.1 (imagined 2.6 / reached 38.7), CEM s42 82.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel
LOGS=/workspace/logs
DRV=$LOGS/driver_gate.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
ACTOR=/workspace/actors/lip4_v2_s0.pt
VALUE=/workspace/metrics/lip4_v2_s0_value.pt
mkdir -p "$LOGS"

log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }

log "=== gate driver start (pid $$), waiting for R1_FINETUNE_DONE ==="
for i in $(seq 1 720); do
  grep -q "R1_FINETUNE_DONE" "$LOGS/driver_r1_finetune.log" 2>/dev/null && break
  sleep 60
done
grep -q "R1_FINETUNE_DONE" "$LOGS/driver_r1_finetune.log" || die "arms never finished (12h)"
log "arms finished"

gate_arm(){ # name (dyna_r1_5050 | dyna_r1_8020)
  local name=$1
  local wmdir=/workspace/models/${name}
  local probe=/workspace/probe_${name}
  # export: newest weights + v2 arch config (same architecture)
  if [ ! -f "$wmdir/config.json" ]; then
    mkdir -p "$wmdir" "$probe"
    local w
    w=$(ls -t /workspace/swm_home/checkpoints/${name}/weights_epoch_*.pt 2>/dev/null | head -1)
    [ -n "$w" ] || { log "$name: NO CHECKPOINT FOUND"; return 1; }
    cp "$w" "$wmdir/"
    cp /workspace/models/v2WM/config.json "$wmdir/config.json"
    log "$name: exported $(basename $w)"
  fi
  mkdir -p "$probe"
  cd "$CODE"
  # probe eval (old actor on new WM, seed 42)
  LIP_PROBE_DIR=$probe CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 scripts/plan/eval_wm.py \
    --config-name cube seed=42 eval.dataset_name="$EXPERT" ++bf16=true \
    eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
    policy="$wmdir" solver=lip "solver.actor_path=$ACTOR" \
    output.filename="gate_lip_${name}.txt" > "$LOGS/gate_lip_${name}.log" 2>&1
  local lsr
  lsr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/gate_lip_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  log "$name: probe LIP s42 = ${lsr:-FAIL} (WM0 old-actor ref 66)"
  # divergence
  python3 /workspace/ab_divergence.py "$probe" "$VALUE" > "$LOGS/gate_div_${name}.log" 2>&1
  local dv
  dv=$(grep "DIVERGENCE" "$LOGS/gate_div_${name}.log" | head -1)
  log "$name: ${dv:-divergence FAILED} (WM0 baseline mean 36.10)"
  # CEM canary
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 scripts/plan/eval_wm.py \
    --config-name cube seed=42 eval.dataset_name="$EXPERT" ++bf16=true \
    eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
    policy="$wmdir" solver=cem \
    output.filename="gate_cem_${name}.txt" > "$LOGS/gate_cem_${name}.log" 2>&1
  local csr
  csr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/gate_cem_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  log "$name: CEM canary s42 = ${csr:-FAIL} (WM0 ref 82)"
}

gate_arm dyna_r1_5050
gate_arm dyna_r1_8020
log "GATE_DONE — compare divergences vs 36.1 and canaries vs 82; winner gets the ladder"
