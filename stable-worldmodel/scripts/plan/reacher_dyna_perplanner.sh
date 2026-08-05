#!/bin/bash
# Per-planner Dyna round-1: each planner is fine-tuned on ITS OWN on-policy
# rollouts, then re-evaluated. This is the only comparable form of the
# experiment — the earlier round fine-tuned on LIP rollouts, so that WM is
# adapted to LIP's state distribution and cannot be used to score CEM/MPPI/Adam.
#
# Loop per (base, planner):
#   1. collect ~1500 on-policy episodes with THAT planner on the author base
#      (eval path + SWM_RECORD_PATH, seeds disjoint from eval 42/43/44)
#   2. build the 80/20 mix (canonical data (dup) on-policy, by ROWS)
#   3. fine-tune the base WM on it (lr 1e-5, 3 epochs, action-pin)
#   4. if the planner uses the TD cost, retrain TD on the fine-tuned WM
#      (fresh value, matching what the LIP arm got)
#   5. card that planner on its own fine-tuned WM
#
# Usage: reacher_dyna_perplanner.sh <base:lejepa|pldm> <planner> <gpu>
#   planner in {latent_cem, latent_mppi, latent_adam, td_cem, td_mppi, td_adam}
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export TQDM_DISABLE=1 MUJOCO_GL=osmesa
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")

BASE=$1
PLANNER=$2
GPU=$3
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
CODE=/workspace/stable-worldmodel
PLAN=$CODE/scripts/plan
TRM=$CODE/scripts/trm
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
CACHES=/workspace/caches
DD=/workspace/dyna_data
CK=/workspace/swm_home/checkpoints
TAG=${BASE}_${PLANNER}
SUM=$RES/summary_perplanner_${BASE}.csv
DRV=$LOGS/driver_pp_${TAG}.log
STATS=/workspace/reacher_action_stats.json
mkdir -p "$LOGS" "$RES" "$MET" "$DD"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][pp-${TAG}] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

case "$BASE" in
  lejepa) WM=$CK/lejepa_reacher; TD_PRE=$MET/td_reacher_lejepa_e0.1_n50.pt; TRAINER=lewm_expert.py ;;
  pldm)   WM=$CK/pldm_reacher;   TD_PRE=$MET/td_reacher_pldm_e0.1_n50.pt;   TRAINER=pldm_expert.py ;;
  *) die "unknown base";;
esac
# solver flags + whether this planner needs the TD cost
case "$PLANNER" in
  latent_cem)  SOLVER=(solver=cem);  USES_TD=0 ;;
  latent_mppi) SOLVER=(solver=mppi); USES_TD=0 ;;
  latent_adam) SOLVER=(solver=adam); USES_TD=0 ;;
  td_cem)      SOLVER=(solver=cem);  USES_TD=1 ;;
  td_mppi)     SOLVER=(solver=mppi); USES_TD=1 ;;
  td_adam)     SOLVER=(solver=adam); USES_TD=1 ;;
  *) die "unknown planner";;
esac
TDFLAG=(); [ "$USES_TD" = 1 ] && TDFLAG=("+metric=$TD_PRE")

# ---------------------------------------------------------------- 1. collect
REC=$DD/pp_${TAG}.lance
CALLS=${CALLS:-30}
if [ ! -f "$DD/.pp_${TAG}.collected" ]; then
  log "collect: ${CALLS} calls x 50 envs with $PLANNER on $(basename "$WM")"
  for i in $(seq 0 $((CALLS-1))); do
    lg=$LOGS/pp_collect_${TAG}_c${i}.log
    grep -q "kept=" "$lg" 2>/dev/null && continue
    SWM_RECORD_PATH=$REC CUDA_VISIBLE_DEVICES=$GPU timeout 7200 python3 "$PLAN/eval_wm.py" \
      --config-name reacher policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
      seed=$((3000 + i)) eval.goal_offset_steps=25 eval.eval_budget=50 \
      solver.batch_size=10 "${SOLVER[@]}" "${TDFLAG[@]}" \
      output.filename="pp_collect_${TAG}_c${i}.txt" > "$lg" 2>&1
    log "  call $i: $(grep -oE 'kept=[0-9]+' "$lg" | tail -1 || echo no-record)"
  done
  touch "$DD/.pp_${TAG}.collected"
fi
python3 - "$REC" << 'PY' | tee -a "$DRV"
import sys, lance, numpy as np
ds = lance.dataset(sys.argv[1])
e = np.asarray(ds.take(list(range(ds.count_rows())), columns=["episode_idx"]).to_pydict()["episode_idx"])
print(f"on-policy: rows={ds.count_rows()} episodes={len(set(e.tolist()))}")
PY

# ---------------------------------------------------------------- 2. mix
MIX=$DD/pp_mix_${TAG}.lance
if [ ! -f "$MIX/.done" ]; then
  log "build 80/20 mix (canonical (dup) on-policy, by rows)"
  python3 /workspace/build_dyna_mix.py --expert "$CANON" --onpolicy "$REC" \
    --out "$MIX" --onpolicy-frac 0.2 > "$LOGS/pp_mix_${TAG}.log" 2>&1 || die "mix failed"
  grep -q BUILD_MIX_DONE "$LOGS/pp_mix_${TAG}.log" || die "mix incomplete"
  touch "$MIX/.done"
fi
log "mix: $(grep 'dup K=' "$LOGS/pp_mix_${TAG}.log")"

# ---------------------------------------------------------------- 3. fine-tune
NAME=dyna_pp_${TAG}
if [ ! -d "$CK/${NAME}_final" ]; then
  log "fine-tune $NAME (lr 1e-5, 3 epochs, $TRAINER)"
  cd "$CODE"
  INIT_WEIGHTS=$WM/weights.pt CUDA_VISIBLE_DEVICES=$GPU timeout 86400 python3 \
    "scripts/train/$TRAINER" data=dmc data.dataset.name="$MIX" \
    "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
    num_workers=24 optimizer.lr=1e-5 trainer.max_epochs=3 trainer.devices=1 \
    output_model_name="$NAME" subdir="$NAME" \
    +action_stats_pin="$STATS" wandb.enabled=false \
    > "$LOGS/pp_ft_${TAG}.log" 2>&1 || die "fine-tune failed (see pp_ft_${TAG}.log)"
  # assemble a load_pretrained-compatible dir (arch config + last epoch weights)
  mkdir -p "$CK/${NAME}_final"
  cp "$WM/config.json" "$CK/${NAME}_final/config.json"
  cp "$(ls -v "$CK/$NAME"/weights_epoch_*.pt | tail -1)" "$CK/${NAME}_final/weights.pt"
fi
WMFT=$CK/${NAME}_final
log "fine-tuned WM ready: $WMFT"

# ------------------------------------------------- 4. fresh TD (TD planners only)
# 3 TD training seeds on the fine-tuned WM (same treatment LIP gets)
if [ "$USES_TD" = 1 ]; then
  C1=$CACHES/pp_${TAG}_fs1.pt
  need=0; for ts in 0 1 2; do [ -f "$MET/td_pp_${TAG}_s${ts}.pt" ] || need=1; done
  if [ "$need" = 1 ]; then
    [ -f "$C1" ] || CUDA_VISIBLE_DEVICES=$GPU python3 "$TRM/cache_latents.py" --wm "$WMFT" \
      --dataset "$CANON" --out "$C1" --batch-size 256 --state-key qpos \
      > "$LOGS/pp_cache_${TAG}.log" 2>&1 || die "cache failed"
    for ts in 0 1 2; do
      [ -f "$MET/td_pp_${TAG}_s${ts}.pt" ] && continue
      CUDA_VISIBLE_DEVICES=$GPU python3 "$PLAN/train_metric.py" --cache "$C1" \
        --learner td --head quasimetric --expectile 0.1 --n-step 50 --steps 6000 \
        --seed "$ts" --out "$MET/td_pp_${TAG}_s${ts}.pt" \
        > "$LOGS/pp_td_${TAG}_s${ts}.log" 2>&1 || die "TD s${ts} failed"
    done
    rm -f "$C1"   # 1.5 GB, regenerable; keep disk free
  fi
fi

# ---------------------------------------------------------------- 5. card
log "card $PLANNER on its OWN fine-tuned WM (3 eval draws; x3 TD seeds if TD-based)"
TDSEEDS="none"; [ "$USES_TD" = 1 ] && TDSEEDS="0 1 2"
t=0; n=0
for ts in $TDSEEDS; do
  TDFT_FLAG=(); sfx=""
  if [ "$ts" != "none" ]; then TDFT_FLAG=("+metric=$MET/td_pp_${TAG}_s${ts}.pt"); sfx="_t${ts}"; fi
  for seed in 42 43 44; do
    for hb in "25 50" "50 100"; do
      set -- $hb; off=$1; bud=$2
      nm="pp_${TAG}${sfx}_h${off}_s${seed}"
      if ! grep -q "^${nm}," "$SUM"; then
        CUDA_VISIBLE_DEVICES=$GPU timeout 14400 python3 "$PLAN/eval_wm.py" --config-name reacher \
          policy="$WMFT" eval.dataset_name="$CANON" dataset.stats="$CANON" \
          seed="$seed" eval.goal_offset_steps="$off" eval.eval_budget="$bud" \
          solver.batch_size=10 "${SOLVER[@]}" "${TDFT_FLAG[@]}" \
          output.filename="${nm}.txt" > "$LOGS/eval_${nm}.log" 2>&1
        sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
        echo "${nm},${sr:-FAIL}" >> "$SUM"; log "eval ${nm}: ${sr:-FAIL}"
      fi
      v=$(sc "$nm"); { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && continue
      t=$(awk "BEGIN{print $t+$v}"); n=$((n+1))
    done
  done
done
[ "$n" -gt 0 ] && log "CARD pp_${TAG}: mean $(awk "BEGIN{printf \"%.1f\", $t/$n}") over n=$n cells"
log "PP_${TAG}_DONE"
