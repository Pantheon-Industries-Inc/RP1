#!/bin/bash
# Reacher Dyna round-1 downstream, base-parametrized. Given a fine-tuned WM
# for <base>, runs: probe-gate (CEM canary + 3 divergence probes with the
# base's stage-1 actors) -> fresh cache/TD/LIP x3 on the fine-tuned WM ->
# full 6-cell card of fresh actors + same-WM CEM + fresh-critic divergence.
# Mirrors the lejepa round-1 exactly; only path prefixes are parametrized.
# Usage: reacher_r1_downstream.sh <base> <wm_dir> <gpu>
#   base in {lejepa,pldm,dinowmnp}; wm_dir = the fine-tuned checkpoint dir
#   (must have a load_pretrained-compatible config.json).
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1 MUJOCO_GL=osmesa
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=$1
WMDIR=$2
GPU=${3:-0}
CODE=/workspace/stable-worldmodel
PLAN=$CODE/scripts/plan
TRM=$CODE/scripts/trm
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
CACHES=/workspace/caches
PY=python3
H5=/workspace/caches/reacher_random_train.h5
LANCE=dmc/reacher_random.lance
C1=$CACHES/reacher_r1_${BASE}_fs1.pt
C5=$CACHES/reacher_r1_${BASE}_fs5.pt
TD=$MET/td_reacher_r1_${BASE}_e0.1_n50.pt
SUM=$RES/summary_reacher_r1_${BASE}.csv
DRV=$LOGS/driver_reacher_r1_down_${BASE}.log
# DINO token WMs need capped cache + small-batch LIP
CAP=""; ITERS=8; BATCH=128
if [ "$BASE" = "dinowmnp" ]; then CAP="--max-rows 200000"; ITERS=4; BATCH=16; fi

mkdir -p "$LOGS" "$RES" "$MET" "$ACT" "$CACHES"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][down-${BASE}] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

run_eval(){ # name seed off bud extra...
  local name=$1 seed=$2 off=$3 bud=$4; shift 4
  grep -q "^${name}," "$SUM" && { log "eval ${name}: cached ($(sc "$name"))"; return 0; }
  CUDA_VISIBLE_DEVICES=$GPU timeout 10800 $PY "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WMDIR" seed="$seed" eval.goal_offset_steps="$off" eval.eval_budget="$bud" \
    solver.batch_size=10 output.filename="${name}.txt" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${name},${sr:-FAIL}" >> "$SUM"; log "eval ${name}: ${sr:-FAIL}"
}
card_mean(){ # tag -> logs 6-cell mean
  local tag=$1 tot=0 v n=0
  for seed in 42 43 44; do for h in h25 h50; do
    v=$(sc "card_${tag}_${h}_s${seed}"); { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && continue
    tot=$(awk "BEGIN{print $tot+$v}"); n=$((n+1))
  done; done
  [ "$n" -gt 0 ] && log "card ${tag} 6-cell mean: $(awk "BEGIN{printf \"%.1f\", $tot/$n}") (n=$n)"
}

# ---------- 1. probe-gate: CEM canary + divergence with stage-1 actors
log "=== probe-gate on fine-tuned WM"
run_eval "gate_${BASE}_cem_h25_s42" 42 25 50 solver=cem
for s in 0 1 2; do
  actor="$ACT/lip4_reacher_${BASE}_a22_s${s}.pt"
  value="$MET/lip4_reacher_${BASE}_a22_s${s}_value.pt"
  [ -f "$actor" ] || { log "no stage-1 actor s${s}"; continue; }
  pdir=/workspace/probe_reacher/r1down_${BASE}_s${s}
  if [ ! -f "$pdir/.done" ]; then
    mkdir -p "$pdir"
    LIP_PROBE_DIR=$pdir CUDA_VISIBLE_DEVICES=$GPU timeout 7200 $PY "$PLAN/eval_wm.py" \
      --config-name reacher policy="$WMDIR" seed=42 eval.goal_offset_steps=25 eval.eval_budget=50 \
      solver=lip "solver.actor_path=$actor" solver.batch_size=50 \
      output.filename="gate_${BASE}_lip_s${s}.txt" > "$LOGS/gate_probe_${BASE}_s${s}.log" 2>&1 \
      && touch "$pdir/.done" || log "probe s${s} failed"
  fi
  lipsr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/gate_probe_${BASE}_s${s}.log" | tail -1 | grep -oE "[0-9.]+$")
  log "stage-1 actor s${s} LIP on fine-tuned WM (h25 s42): ${lipsr:-?}"
  log "divergence s${s}:"; $PY /workspace/ab_divergence.py "$pdir" "$value" 2>&1 | tee -a "$DRV"
done

# ---------- 2. fresh cache/TD/LIP on the fine-tuned WM
log "=== fresh cache + TD"
[ -f "$C1" ] || CUDA_VISIBLE_DEVICES=$GPU $PY "$TRM/cache_latents.py" --wm "$WMDIR" \
  --dataset "$LANCE" --out "$C1" --batch-size 256 $CAP \
  > "$LOGS/cache_r1_${BASE}_fs1.log" 2>&1 || die "fs1 failed"
[ -f "$C5" ] || $PY "$TRM/subsample_cache.py" --in "$C1" --out "$C5" --frameskip 5 \
  > "$LOGS/cache_r1_${BASE}_fs5.log" 2>&1 || die "fs5 failed"
[ -f "$TD" ] || CUDA_VISIBLE_DEVICES=$GPU $PY "$PLAN/train_metric.py" --cache "$C1" \
  --learner td --head quasimetric --expectile 0.1 --n-step 50 --steps 6000 --out "$TD" \
  > "$LOGS/td_r1_${BASE}.log" 2>&1 || die "TD failed"
log "fresh phase1 done"

log "=== fresh LIP x3"
for s in 0 1 2; do
  out="$ACT/lip4_reacher_r1_${BASE}_s${s}.pt"
  [ -f "$out" ] && { log "fresh actor s${s}: cached"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU timeout 28800 $PY "$PLAN/train_lip_ac.py" \
    --cache "$C5" --cache-td "$C1" --h5 "$H5" --wm "$WMDIR" --init-value "$TD" \
    --arch v4 --amax 2.2 --max-delta 12 --iters "$ITERS" --batch "$BATCH" --horizon 5 --steps 8000 \
    --n-step 50 --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 3e-4 --actor-lr-final 3e-5 \
    --seed "$s" --out "$out" --out-value "$MET/lip4_reacher_r1_${BASE}_s${s}_value.pt" \
    > "$LOGS/train_lip4_r1_${BASE}_s${s}.log" 2>&1 || { log "fresh actor s${s} FAILED"; continue; }
  log "fresh actor s${s}: done"
done

# ---------- 3. card fresh actors + same-WM CEM + fresh divergence
log "=== fresh-actor cards"
for s in 0 1 2; do
  actor="$ACT/lip4_reacher_r1_${BASE}_s${s}.pt"; [ -f "$actor" ] || continue
  for seed in 42 43 44; do for hb in "25 50" "50 100"; do
    set -- $hb
    run_eval "card_r1fresh_${BASE}_s${s}_h${1}_s${seed}" "$seed" "$1" "$2" solver=lip "solver.actor_path=$actor"
  done; done
  card_mean "r1fresh_${BASE}_s${s}"
  pdir=/workspace/probe_reacher/r1fresh_${BASE}_s${s}
  if [ ! -f "$pdir/.done" ]; then
    mkdir -p "$pdir"
    LIP_PROBE_DIR=$pdir CUDA_VISIBLE_DEVICES=$GPU timeout 7200 $PY "$PLAN/eval_wm.py" \
      --config-name reacher policy="$WMDIR" seed=42 eval.goal_offset_steps=25 eval.eval_budget=50 \
      solver=lip "solver.actor_path=$actor" solver.batch_size=50 \
      output.filename="probe_r1fresh_${BASE}_s${s}.txt" > "$LOGS/probe_r1fresh_${BASE}_s${s}.log" 2>&1 \
      && touch "$pdir/.done"
  fi
  log "FRESH divergence s${s}:"; $PY /workspace/ab_divergence.py "$pdir" "$MET/lip4_reacher_r1_${BASE}_s${s}_value.pt" 2>&1 | tee -a "$DRV"
done
for seed in 42 43 44; do for hb in "25 50" "50 100"; do
  set -- $hb; run_eval "card_r1fresh_${BASE}_cem_h${1}_s${seed}" "$seed" "$1" "$2" solver=cem
done; done
card_mean "r1fresh_${BASE}_cem"
log "R1_DOWNSTREAM_${BASE}_DONE"
