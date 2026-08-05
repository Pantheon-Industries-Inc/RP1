#!/bin/bash
# Reacher Dyna round-1 probe-gate: for each fine-tuned arm, re-measure the
# A/B imagined-vs-reached divergence with the SAME stage-1 actors + tandem
# critics (baseline: lejepa {45.8, 33.3, 47.7} mean 42.3), plus a 1-cell CEM
# canary (h25 s42; stage-1 reference 66.0 on this draw) to catch general-
# dynamics degradation. Ladder/re-card only the winning arm afterwards.
# Waits for each arm's "done" line independently; 5050 gates on gpu0 while
# 8020 may still be training on gpu1.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

CODE=/workspace/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
CK=/workspace/swm_home/checkpoints
DRV=$LOGS/driver_reacher_r1_gate.log
PY=python3

log(){ echo "[$(date -u +%m%d-%H:%M:%S)][gate] $*" | tee -a "$DRV"; }

prep_final(){ # arm -> echoes final dir
  local arm=$1
  local src="$CK/dyna_reacher_r1_${arm}"
  local dst="$CK/dyna_reacher_r1_${arm}_final"
  mkdir -p "$dst"
  # the training callback writes the TRAINING config; load_pretrained needs the
  # model-ARCHITECTURE config (_target_ ...). Fine-tune preserves architecture,
  # so reuse the converted base's config verbatim.
  cp "$CK/lejepa_reacher/config.json" "$dst/config.json"
  local last
  last=$(ls -v "$src"/weights_epoch_*.pt | tail -1)
  cp "$last" "$dst/weights.pt"
  log "arm ${arm}: final = $(basename "$last")" >&2
  echo "$dst"
}

gate_arm(){ # arm gpu
  local arm=$1 gpu=$2
  log "waiting for arm ${arm} fine-tune"
  while ! grep -q "finetune dyna_reacher_r1_${arm}: done" "$LOGS/driver_reacher_r1_ft.log"; do
    grep -q "finetune dyna_reacher_r1_${arm}: FAILED" "$LOGS/driver_reacher_r1_ft.log" && {
      log "arm ${arm}: fine-tune FAILED — no gate"; return 1; }
    sleep 120
  done
  local wmdir
  wmdir=$(prep_final "$arm")

  # CEM canary (planner-independent general-dynamics check)
  local cn="gate_${arm}_cem_h25_s42"
  CUDA_VISIBLE_DEVICES=$gpu timeout 7200 $PY "$PLAN/eval_wm.py" \
    --config-name reacher policy="$wmdir" \
    seed=42 eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=cem solver.batch_size=10 output.filename="${cn}.txt" \
    > "$LOGS/eval_${cn}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${cn}.log" | tail -1 | grep -oE "[0-9.]+$")
  log "arm ${arm} CEM canary h25 s42: ${sr:-FAIL} (stage-1 ref 66.0)"

  # divergence probes with the stage-1 actors + their tandem critics
  for seed in 0 1 2; do
    local actor=/workspace/actors/lip4_reacher_lejepa_a22_s${seed}.pt
    local value=/workspace/metrics/lip4_reacher_lejepa_a22_s${seed}_value.pt
    local pdir=/workspace/probe_reacher/r1_${arm}_s${seed}
    if [ ! -f "$pdir/.done" ]; then
      mkdir -p "$pdir"
      LIP_PROBE_DIR=$pdir CUDA_VISIBLE_DEVICES=$gpu timeout 7200 $PY "$PLAN/eval_wm.py" \
        --config-name reacher policy="$wmdir" \
        seed=42 eval.goal_offset_steps=25 eval.eval_budget=50 \
        solver=lip "solver.actor_path=$actor" solver.batch_size=50 \
        output.filename="gate_${arm}_lip_s${seed}.txt" \
        > "$LOGS/gate_probe_${arm}_s${seed}.log" 2>&1 \
        || { log "arm ${arm} probe s${seed} FAILED"; continue; }
      touch "$pdir/.done"
    fi
    local lipsr
    lipsr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/gate_probe_${arm}_s${seed}.log" | tail -1 | grep -oE "[0-9.]+$")
    log "arm ${arm} LIP(s${seed}) h25 s42 on fine-tuned WM: ${lipsr:-?}"
    log "arm ${arm} divergence s${seed} (baseline {45.8,33.3,47.7}):"
    $PY /workspace/ab_divergence.py "$pdir" "$value" 2>&1 | tee -a "$DRV"
  done
  log "arm ${arm} GATE DONE"
}

gate_arm 5050 0 &
G1=$!
gate_arm 8020 1 &
G2=$!
wait $G1 || true
wait $G2 || true
log "R1_GATE_ALL_DONE"
