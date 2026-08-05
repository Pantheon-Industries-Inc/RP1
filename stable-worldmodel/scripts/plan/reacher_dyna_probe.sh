#!/bin/bash
# Reacher Dyna stage-1 divergence probe: first measurement of the OGBench
# "action exploitation" signature on reacher. For each WM base, run the LIP
# actor through the standard eval (h25, seed 42, n=50, budget 50 => 2 replan
# rounds) with LIP_PROBE_DIR set, then compute the A/B imagined-vs-reached
# divergence with the actor's own tandem critic (the exact V the planner
# maximized). solver.batch_size=50 => one probe file per replan round, so
# consecutive files pair correctly. OGBench baseline for scale: divergence
# mean 36.1 (imagined 2.6 vs reached 38.7) on v2WM.
# Usage: reacher_dyna_probe.sh   (gated on both LIP chains' .DONE markers)
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=18 MKL_NUM_THREADS=18

CODE=/workspace/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
DRV=$LOGS/driver_reacher_probe.log
PY=python3

log(){ echo "[$(date -u +%m%d-%H:%M:%S)][probe] $*" | tee -a "$DRV"; }

log "waiting for LIP chains"
for wm in lejepa pldm; do
  while [ ! -f "$RES/reacher_lip4_${wm}.DONE" ]; do sleep 300; done
done
log "LIP chains done"

for wm in lejepa pldm; do
  gpu=0; [ "$wm" = "pldm" ] && gpu=1
  for seed in 0 1 2; do
    actor=/workspace/actors/lip4_reacher_${wm}_a22_s${seed}.pt
    value=/workspace/metrics/lip4_reacher_${wm}_a22_s${seed}_value.pt
    pdir=/workspace/probe_reacher/${wm}_s${seed}
    [ -f "$actor" ] || { log "skip ${wm} s${seed}: no actor"; continue; }
    if [ ! -f "$pdir/.done" ]; then
      mkdir -p "$pdir"
      log "probe eval ${wm} s${seed} (gpu${gpu})"
      LIP_PROBE_DIR=$pdir CUDA_VISIBLE_DEVICES=$gpu timeout 7200 $PY "$PLAN/eval_wm.py" \
        --config-name reacher policy="${wm}_reacher" \
        seed=42 eval.goal_offset_steps=25 eval.eval_budget=50 \
        solver=lip "solver.actor_path=$actor" solver.batch_size=50 \
        output.filename="probe_${wm}_s${seed}.txt" \
        > "$LOGS/probe_eval_${wm}_s${seed}.log" 2>&1 \
        || { log "probe eval ${wm} s${seed} FAILED"; continue; }
      touch "$pdir/.done"
    fi
    log "divergence ${wm} s${seed}:"
    $PY /workspace/ab_divergence.py "$pdir" "$value" 2>&1 | tee -a "$DRV"
  done
done
log "PROBE_DONE"
