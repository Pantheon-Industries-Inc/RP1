#!/bin/bash
# Retrain the LEARNED components (TD value + LIP actor) on the SAME data the
# world models were trained on: the canonical quentinll/lewm-reacher reacher.h5.
#
# Why: the first pass trained TD/LIP on caches built from OUR re-collection
# (`dmc/reacher_random.lance`). That data is content-identical to canonical (our
# row t == canonical row t+1, actions matching to 3e-08, 2.00M vs 2.01M rows —
# we dropped each episode's reset frame), but "content-identical" is an argument,
# not a control. This trains on the authors' file end to end so the comparison
# needs no argument.
#
# The latent baselines (Latent+CEM/MPPI/Adam) need no retraining — they have no
# learned parts and are already measured on canonical in summary_matrix_<base>.
#
# Usage: reacher_canon_retrain.sh <base:lejepa|pldm> <gpu>
# Idempotent: caches/values/actors and CSV rows are all skipped if present.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export TQDM_DISABLE=1 MUJOCO_GL=osmesa
export OMP_NUM_THREADS=18 MKL_NUM_THREADS=18
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")

BASE=$1
GPU=$2
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
CODE=/workspace/stable-worldmodel
PLAN=$CODE/scripts/plan
TRM=$CODE/scripts/trm
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
CACHES=/workspace/caches
CK=/workspace/swm_home/checkpoints
WM=$CK/${BASE}_reacher
C1=$CACHES/canon_${BASE}_fs1.pt
C5=$CACHES/canon_${BASE}_fs5.pt
SUM=$RES/summary_canonretrain_${BASE}.csv
DRV=$LOGS/driver_canonretrain_${BASE}.log
mkdir -p "$LOGS" "$RES" "$MET" "$ACT" "$CACHES"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][cr-${BASE}] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

# ---------------------------------------------------------------- caches
# _episode_col() already falls back to the canonical file's `ep_idx`, so the h5
# needs no adaptation. --state-key qpos keeps ground truth in the cache so the
# value audits work later.
if [ ! -f "$C1" ]; then
  log "fs1 cache from CANONICAL h5 (2.01M rows; ~40-60 min)"
  CUDA_VISIBLE_DEVICES=$GPU python3 "$TRM/cache_latents.py" --wm "$WM" \
    --dataset "$CANON" --out "$C1" --batch-size 256 --state-key qpos \
    > "$LOGS/canon_cache_${BASE}_fs1.log" 2>&1 || die "fs1 cache failed"
fi
[ -f "$C5" ] || { log "fs5 subsample"; python3 "$TRM/subsample_cache.py" --in "$C1" \
  --out "$C5" --frameskip 5 > "$LOGS/canon_cache_${BASE}_fs5.log" 2>&1 || die "fs5 failed"; }
log "caches ready: $(python3 -c "
import torch; c=torch.load('$C1',map_location='cpu',weights_only=False)
print('fs1', tuple(c['z'].shape), 'dataset=', c.get('meta',{}).get('dataset'))" 2>/dev/null)"

# ---------------------------------------------------------------- TD x3 seeds
for ts in 0 1 2; do
  td=$MET/td_canon_${BASE}_s${ts}.pt
  [ -f "$td" ] && { log "TD s${ts}: cached"; continue; }
  log "TD s${ts} (tau 0.1, n-step 50, 6k steps)"
  CUDA_VISIBLE_DEVICES=$GPU python3 "$PLAN/train_metric.py" --cache "$C1" \
    --learner td --head quasimetric --expectile 0.1 --n-step 50 --steps 6000 \
    --seed "$ts" --out "$td" > "$LOGS/canon_td_${BASE}_s${ts}.log" 2>&1 \
    || die "TD s${ts} failed"
done

# ---------------------------------------------------------------- LIP x3 seeds
# canonical recipe, amax 2.2 (the sweep winner; re-tuning is a separate question)
for s in 0 1 2; do
  out=$ACT/lip4_canon_${BASE}_s${s}.pt
  [ -f "$out" ] && { log "actor s${s}: cached"; continue; }
  log "LIP actor s${s} (v4 amax2.2 md12 iters8, warm from td_canon_s0)"
  CUDA_VISIBLE_DEVICES=$GPU timeout 28800 python3 "$PLAN/train_lip_ac.py" \
    --cache "$C5" --cache-td "$C1" --h5 "$CANON" --wm "$WM" \
    --init-value "$MET/td_canon_${BASE}_s0.pt" \
    --arch v4 --amax 2.2 --max-delta 12 --iters 8 --horizon 5 --steps 8000 \
    --n-step 50 --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 3e-4 --actor-lr-final 3e-5 \
    --seed "$s" --out "$out" --out-value "$MET/lip4_canon_${BASE}_s${s}_value.pt" \
    > "$LOGS/canon_train_lip4_${BASE}_s${s}.log" 2>&1 || { log "actor s${s} FAILED"; continue; }
done

# ---------------------------------------------------------------- cards
ev(){ # name seed off bud extra...
  local nm=$1 seed=$2 off=$3 bud=$4; shift 4
  grep -q "^${nm}," "$SUM" && { log "eval ${nm}: cached ($(sc "$nm"))"; return 0; }
  CUDA_VISIBLE_DEVICES=$GPU timeout 14400 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    seed="$seed" eval.goal_offset_steps="$off" eval.eval_budget="$bud" \
    solver.batch_size=10 output.filename="${nm}.txt" "$@" \
    > "$LOGS/eval_${nm}.log" 2>&1
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "eval ${nm}: ${sr:-FAIL}"
}
card(){ local tag=$1; shift
  for seed in 42 43 44; do
    ev "cr_${tag}_h25_s${seed}" "$seed" 25 50  "$@"
    ev "cr_${tag}_h50_s${seed}" "$seed" 50 100 "$@"
  done
  local t=0 n=0 v
  for seed in 42 43 44; do for h in h25 h50; do
    v=$(sc "cr_${tag}_${h}_s${seed}"); { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && continue
    t=$(awk "BEGIN{print $t+$v}"); n=$((n+1)); done; done
  [ "$n" -gt 0 ] && log "CARD ${tag}: 6-cell mean $(awk "BEGIN{printf \"%.1f\", $t/$n}") (n=$n)"
}
agg(){ local pfx=$1 t=0 n=0 v   # aggregate a 3-seed family
  for s in 0 1 2; do for seed in 42 43 44; do for h in h25 h50; do
    v=$(sc "cr_${pfx}${s}_${h}_s${seed}"); { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && continue
    t=$(awk "BEGIN{print $t+$v}"); n=$((n+1)); done; done; done
  [ "$n" -gt 0 ] && log "CARD ${pfx}3seed: mean $(awk "BEGIN{printf \"%.1f\", $t/$n}") (n=$n)"
}

for ts in 0 1 2; do card "td_cem_t${ts}" solver=cem "+metric=$MET/td_canon_${BASE}_s${ts}.pt"; done
agg "td_cem_t"
for s in 0 1 2; do
  a=$ACT/lip4_canon_${BASE}_s${s}.pt
  [ -f "$a" ] && card "lip_s${s}" solver=lip "solver.actor_path=$a"
done
agg "lip_s"

log "=== ${BASE} canonical-retrain summary (compare vs summary_matrix_${BASE}: Latent+CEM, and the ours-trained TD/LIP)"
grep -h "CARD " "$DRV" | tail -10
log "CANONRETRAIN_${BASE}_DONE"
