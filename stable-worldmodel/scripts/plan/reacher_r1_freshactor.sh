#!/bin/bash
# Reacher Dyna round-1 PROPER actor: fresh cache + TD + LIP trained directly on
# the winning fine-tuned WM (8020), matched to its shifted latent space (the
# handoff's per-round prescription). Then full card + a FRESH-critic divergence
# probe (the clean measurement the stale stage-1 critic could not give).
# Compare vs: CEM on the SAME (8020) WM = 86.0; stage-1 actors on 8020 = 85.0;
# round-0 divergence 42.3.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

ARM=8020
WMDIR=/workspace/swm_home/checkpoints/dyna_reacher_r1_${ARM}_final
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
C1=$CACHES/reacher_r1_${ARM}_fs1.pt
C5=$CACHES/reacher_r1_${ARM}_fs5.pt
TD=$MET/td_reacher_r1_${ARM}_e0.1_n50.pt
SUM=$RES/summary_reacher_r1_${ARM}.csv
DRV=$LOGS/driver_reacher_r1_fresh_${ARM}.log

mkdir -p "$LOGS" "$RES" "$MET" "$ACT" "$CACHES"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][fresh-${ARM}] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

# ---- phase 1: cache + TD on the fine-tuned WM (GPU0)
[ -f "$C1" ] || { log "fs1 cache on 8020 WM"; CUDA_VISIBLE_DEVICES=0 $PY "$TRM/cache_latents.py" \
  --wm "$WMDIR" --dataset "$LANCE" --out "$C1" --batch-size 256 \
  > "$LOGS/cache_r1_${ARM}_fs1.log" 2>&1 || die "fs1 failed"; }
[ -f "$C5" ] || { log "fs5"; $PY "$TRM/subsample_cache.py" --in "$C1" --out "$C5" --frameskip 5 \
  > "$LOGS/cache_r1_${ARM}_fs5.log" 2>&1 || die "fs5 failed"; }
[ -f "$TD" ] || { log "TD"; CUDA_VISIBLE_DEVICES=0 $PY "$PLAN/train_metric.py" --cache "$C1" \
  --learner td --head quasimetric --expectile 0.1 --n-step 50 --steps 6000 --out "$TD" \
  > "$LOGS/td_r1_${ARM}.log" 2>&1 || die "TD failed"; }
log "phase1 done"

# ---- LIP actors (3 seeds, split across GPUs)
train_seed(){ # seed gpu
  local s=$1 g=$2
  local out="$ACT/lip4_reacher_r1_${ARM}_s${s}.pt"
  [ -f "$out" ] && { log "actor s${s}: cached"; return 0; }
  log "actor s${s}: train (gpu${g}, v4 amax2.2 md12)"
  CUDA_VISIBLE_DEVICES=$g timeout 28800 $PY "$PLAN/train_lip_ac.py" \
    --cache "$C5" --cache-td "$C1" --h5 "$H5" --wm "$WMDIR" --init-value "$TD" \
    --arch v4 --amax 2.2 --max-delta 12 --iters 8 --horizon 5 --steps 8000 \
    --n-step 50 --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 3e-4 --actor-lr-final 3e-5 \
    --seed "$s" --out "$out" --out-value "$MET/lip4_reacher_r1_${ARM}_s${s}_value.pt" \
    > "$LOGS/train_lip4_r1_${ARM}_s${s}.log" 2>&1 || { log "actor s${s} FAILED"; return 1; }
  log "actor s${s}: done"
}
train_seed 0 0 & train_seed 1 1 & wait
train_seed 2 0

# ---- card + fresh-critic divergence per seed
run_eval(){ # name seed off bud extra...
  local name=$1 seed=$2 off=$3 bud=$4; shift 4
  grep -q "^${name}," "$SUM" && { log "eval ${name}: cached ($(sc "$name"))"; return 0; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 $PY "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WMDIR" seed="$seed" eval.goal_offset_steps="$off" eval.eval_budget="$bud" \
    solver.batch_size=10 output.filename="${name}.txt" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${name},${sr:-FAIL}" >> "$SUM"; log "eval ${name}: ${sr:-FAIL}"
}
for s in 0 1 2; do
  actor="$ACT/lip4_reacher_r1_${ARM}_s${s}.pt"
  [ -f "$actor" ] || continue
  tot=0; n=0
  for seed in 42 43 44; do for hb in "25 50" "50 100"; do
    set -- $hb
    nm="card_r1fresh_${ARM}_s${s}_h${1}_seed${seed}"
    run_eval "$nm" "$seed" "$1" "$2" solver=lip "solver.actor_path=$actor"
    v=$(sc "$nm"); [ "$v" = "FAIL" ] && continue
    tot=$(awk "BEGIN{print $tot+$v}"); n=$((n+1))
  done; done
  [ "$n" -gt 0 ] && log "FRESH actor s${s} 6-cell mean: $(awk "BEGIN{printf \"%.1f\", $tot/$n}") (n=$n)"
  # fresh-critic divergence
  pdir=/workspace/probe_reacher/r1fresh_${ARM}_s${s}
  if [ ! -f "$pdir/.done" ]; then
    mkdir -p "$pdir"
    LIP_PROBE_DIR=$pdir CUDA_VISIBLE_DEVICES=0 timeout 7200 $PY "$PLAN/eval_wm.py" \
      --config-name reacher policy="$WMDIR" seed=42 eval.goal_offset_steps=25 eval.eval_budget=50 \
      solver=lip "solver.actor_path=$actor" solver.batch_size=50 \
      output.filename="probe_r1fresh_${ARM}_s${s}.txt" > "$LOGS/probe_r1fresh_${ARM}_s${s}.log" 2>&1 \
      && touch "$pdir/.done"
  fi
  log "FRESH divergence s${s} (round-0 baseline 42.3):"
  $PY /workspace/ab_divergence.py "$pdir" "$MET/lip4_reacher_r1_${ARM}_s${s}_value.pt" 2>&1 | tee -a "$DRV"
done
log "R1_FRESH_${ARM}_DONE"
