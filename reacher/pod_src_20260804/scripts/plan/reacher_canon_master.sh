#!/bin/bash
# FULL RERUN on the CANONICAL LeWM reacher dataset (quentinll/lewm-reacher).
#
# Why: every earlier reacher number used our own re-collection
# (scripts/data/collect_reacher.py -> dmc/reacher_random.lance). The authors'
# bases were trained on `reacher.h5` from quentinll/lewm-reacher (dated
# 2026-03-27, matching the epoch-10 base checkpoints). All three bases came in
# BELOW published on the authors' 50-step protocol (LeWM 69.3 vs 86, PLDM 74.0
# vs 78, DINO 72.0 vs 79) and the DINO predictor showed a fixed 213-norm feature
# offset that survived a teacher-forced test — both consistent with a domain
# shift. This rerun uses the official data end to end.
#
# STAGE 0 is a GATE: convert-validation open-loop ratios + LeWM CEM at the
# authors' protocol (h25 = goal_offset 25, budget 50). If LeWM CEM lands near
# 86 the domain-shift diagnosis is confirmed and the rest is worth the compute.
# Stage 0 writes results/canon_gate.txt and never blocks later stages, but the
# driver stops before the Dyna stage so the gate can be reviewed.
#
# Old (own-collection) artifacts are NOT deleted — they are the other arm of a
# clean two-dataset domain-shift comparison. Everything here is suffixed _canon.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1 MUJOCO_GL=osmesa
export OMP_NUM_THREADS=18 MKL_NUM_THREADS=18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CODE=/workspace/stable-worldmodel
PLAN=$CODE/scripts/plan
TRM=$CODE/scripts/trm
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
CACHES=/workspace/caches
PY=python3
CANON=/workspace/datasets_canon/lewm-reacher
DRV=$LOGS/driver_canon_master.log
mkdir -p "$LOGS" "$RES" "$MET" "$ACT" "$CACHES"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][canon] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------- locate data
H5=$(find "$CANON" -name "*.h5" -maxdepth 2 | head -1)
LANCE=$(find "$CANON" -name "*.lance" -maxdepth 2 -type d | head -1)
log "canonical h5=${H5:-none} lance=${LANCE:-none}"
[ -n "$H5$LANCE" ] || die "no .h5 or .lance found under $CANON"
# planner/eval dataset id: prefer the h5 (that is what data/dmc.yaml names)
EVALDS="${H5:-$LANCE}"
# cache_latents wants something load_dataset can open; both work
CACHEDS="$EVALDS"
log "using dataset: $EVALDS"

# ---------------------------------------------------------------- stage 0 gate
log "=== STAGE 0: conversion validation on canonical data"
for wm in lejepa pldm dinowmnp; do
  ck=/workspace/swm_home/checkpoints/${wm}_reacher_canon
  if [ ! -d "$ck" ]; then
    # convert again, validating against CANONICAL frames/actions
    CUDA_VISIBLE_DEVICES=0 $PY "$PLAN/convert_reacher_bases.py" --only "$wm" \
      --dataset "$CACHEDS" --h5 "$EVALDS" \
      > "$LOGS/convert_canon_${wm}.log" 2>&1 || log "convert $wm returned nonzero"
    # convert writes <wm>_reacher; clone to _canon so the own-collection ckpt survives
    src=/workspace/swm_home/checkpoints/${wm}_reacher
    [ -d "$src" ] && cp -r "$src" "$ck"
  fi
  grep -hE "ratio|parity|backbone vs" "$LOGS/convert_canon_${wm}.log" 2>/dev/null | sed "s/^/[${wm}] /" | tee -a "$DRV"
done
log "conversion validation done — a ratio < 1 means the WM beats copy-last on THIS data"

log "=== STAGE 0 GATE: LeWM (lejepa) CEM at the authors' protocol (h25, budget 50)"
: > "$RES/canon_gate.txt"
for seed in 42 43 44; do
  nm="canon_gate_lejepa_cem_h25_s${seed}"
  CUDA_VISIBLE_DEVICES=0 timeout 10800 $PY "$PLAN/eval_wm.py" --config-name reacher \
    policy=lejepa_reacher_canon eval.dataset_name="$EVALDS" dataset.stats="$EVALDS" \
    seed="$seed" eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=cem solver.batch_size=10 output.filename="${nm}.txt" \
    > "$LOGS/eval_${nm}.log" 2>&1
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" | tee -a "$RES/canon_gate.txt" | tee -a "$DRV"
done
awk -F, '$2!="FAIL"{s+=$2;n++} END{if(n)printf "GATE: LeWM CEM h25 canonical = %.1f (published 86; own-collection was 69.3)\n", s/n}' \
  "$RES/canon_gate.txt" | tee -a "$DRV"
log "STAGE 0 COMPLETE — review the gate before the Dyna stage"

# ---------------------------------------------------------------- stage 1
# per-base: caches + TD + LIP(amax2.2 x3) + baselines, all on canonical data.
stage1(){ # wm gpu
  local wm=$1 gpu=$2
  local ckpt="${wm}_reacher_canon"
  local c1=$CACHES/canon_${wm}_fs1.pt c5=$CACHES/canon_${wm}_fs5.pt
  local td=$MET/td_canon_${wm}_e0.1_n50.pt
  local sum=$RES/summary_canon_${wm}.csv
  local cap="" iters=8 batch=128
  [ "$wm" = "dinowmnp" ] && { cap="--max-rows 200000"; iters=4; batch=16; }
  touch "$sum"

  [ -f "$c1" ] || CUDA_VISIBLE_DEVICES=$gpu $PY "$TRM/cache_latents.py" --wm "$ckpt" \
    --dataset "$CACHEDS" --out "$c1" --batch-size 256 --state-key qpos $cap \
    > "$LOGS/canon_cache_${wm}_fs1.log" 2>&1 || { log "$wm fs1 FAILED"; return 1; }
  [ -f "$c5" ] || $PY "$TRM/subsample_cache.py" --in "$c1" --out "$c5" --frameskip 5 \
    > "$LOGS/canon_cache_${wm}_fs5.log" 2>&1 || { log "$wm fs5 FAILED"; return 1; }
  [ -f "$td" ] || CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_metric.py" --cache "$c1" \
    --learner td --head quasimetric --expectile 0.1 --n-step 50 --steps 6000 --out "$td" \
    > "$LOGS/canon_td_${wm}.log" 2>&1 || { log "$wm TD FAILED"; return 1; }
  log "$wm: caches + TD done"

  ev(){ # name seed off bud extra...
    local nm=$1 seed=$2 off=$3 bud=$4; shift 4
    grep -q "^${nm}," "$sum" && return 0
    CUDA_VISIBLE_DEVICES=$gpu timeout 14400 $PY "$PLAN/eval_wm.py" --config-name reacher \
      eval.dataset_name="$EVALDS" dataset.stats="$EVALDS" \
      seed="$seed" eval.goal_offset_steps="$off" eval.eval_budget="$bud" \
      solver.batch_size=10 output.filename="${nm}.txt" "$@" \
      > "$LOGS/eval_${nm}.log" 2>&1
    local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
    echo "${nm},${sr:-FAIL}" >> "$sum"; log "$wm eval ${nm}: ${sr:-FAIL}"
  }
  card(){ local tag=$1; shift
    for seed in 42 43 44; do ev "card_canon_${tag}_h25_s${seed}" "$seed" 25 50 "$@"
                             ev "card_canon_${tag}_h50_s${seed}" "$seed" 50 100 "$@"; done
    local t=0 n=0 v
    for seed in 42 43 44; do for h in h25 h50; do
      v=$(grep "^card_canon_${tag}_${h}_s${seed}," "$sum" | tail -1 | cut -d, -f2)
      [ "$v" = "FAIL" ] || [ -z "$v" ] && continue
      t=$(awk "BEGIN{print $t+$v}"); n=$((n+1)); done; done
    [ "$n" -gt 0 ] && log "$wm card ${tag}: 6-cell $(awk "BEGIN{printf \"%.1f\", $t/$n}") (n=$n)"
  }

  # LIP actors (canonical, amax 2.2, 3 seeds)
  for s in 0 1 2; do
    local out="$ACT/lip4_canon_${wm}_s${s}.pt"
    [ -f "$out" ] || CUDA_VISIBLE_DEVICES=$gpu timeout 28800 $PY "$PLAN/train_lip_ac.py" \
      --cache "$c5" --cache-td "$c1" --h5 "$EVALDS" --wm "$ckpt" --init-value "$td" \
      --arch v4 --amax 2.2 --max-delta 12 --iters "$iters" --batch "$batch" \
      --horizon 5 --steps 8000 --n-step 50 --expectile 0.1 --expectile-final 0.03 \
      --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 3e-4 --actor-lr-final 3e-5 \
      --seed "$s" --out "$out" --out-value "$MET/lip4_canon_${wm}_s${s}_value.pt" \
      > "$LOGS/canon_train_lip4_${wm}_s${s}.log" 2>&1 || log "$wm LIP s$s FAILED"
    [ -f "$out" ] && card "${wm}_lip_s${s}" solver=lip "solver.actor_path=$out"
  done

  # baselines on canonical data
  card "${wm}_cem"  policy="$ckpt" solver=cem
  card "${wm}_mppi" policy="$ckpt" solver=mppi
  card "${wm}_gd"   policy="$ckpt" solver=adam
  card "${wm}_tdcem" policy="$ckpt" solver=cem "+metric=$td"
  if [ "$wm" = "lejepa" ]; then card "floor_random" policy=random; card "floor_nomove" policy=nomove; fi
  log "$wm STAGE1 DONE"
}

log "=== STAGE 1: lejepa(gpu0) + pldm(gpu1) in parallel"
stage1 lejepa 0 > "$LOGS/canon_stage1_lejepa.log" 2>&1 &
P1=$!
stage1 pldm 1 > "$LOGS/canon_stage1_pldm.log" 2>&1 &
P2=$!
wait $P1; wait $P2
log "=== STAGE 1: dinowmnp(gpu0)"
stage1 dinowmnp 0 > "$LOGS/canon_stage1_dinowmnp.log" 2>&1
log "CANON_MASTER_DONE — Dyna re-run is deliberately NOT started; review first"
