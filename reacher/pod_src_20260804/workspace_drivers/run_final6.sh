#!/bin/bash
# The headline run: final recipes, trained on held-out-clean data, 6 seeds.
#
# Everything the campaign converged on, measured once without the two caveats
# that qualify every earlier number:
#
#   1. TRAINED CLEAN. Value and actor see episodes 0..7999 only; the card is on
#      8000:10000. The leak probe put this at +0.2 (lejepa) and +3.6 (pldm)
#      points, so it is small -- but "small" was a measurement, not an
#      assumption, and the headline should not need the caveat at all.
#      Unfixable here: the lejepa/pldm world models were pretrained on all 10k
#      episodes. This is "value + actor clean, world model leaked".
#
#   2. SIX SEEDS. Every card so far is 3 training seeds, and the spread has been
#      wide enough to flip conclusions -- pldm's LR-sweep winner ran 78.4 to
#      95.5 across three seeds, and a half-crashed 3-seed cell produced the
#      false conclusion that pldm preferred gamma 1.0. Three seeds cannot
#      support a +14 point margin claim.
#
# FINAL RECIPES (shared knobs identical, per-base knobs differ only where the
# user allowed: amax, actor-lr, mean-weight):
#   shared   gamma 0.98, expectile 0.05, replay-prob 0.5, expand-weight 0,
#            n-step 50, 1000 steps, batch 128, uniform lambda, iters 8,
#            horizon 5, max-delta 12, pad-context
#   lejepa   amax 2.2, actor-lr 1e-4, mean-weight 0.3
#   pldm     amax 1.8, actor-lr 3e-4, mean-weight 0.5
#
# Bars @0.1 (Latent+CEM, 3000 rollouts, same protocol): lejepa 84.3, pldm 78.3.
# LIP uses ~16 rollouts.
#
# Reference cards to beat, both leaked: lejepa 99.0, pldm 94.1.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
SUM=/workspace/results/summary_final6.csv; touch "$SUM"
L=/workspace/logs/final6; mkdir -p "$L"
REP="42 43 44 45 46 47"
SEEDS="0 1 2 3 4 5"
G=0.98
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][final] $*"; }
die(){ log "FATAL: $*"; exit 1; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
lr_of(){   [ "$1" = "lejepa" ] && echo 1e-4 || echo 3e-4; }
lrf_of(){  [ "$1" = "lejepa" ] && echo 1e-5 || echo 3e-5; }
mw_of(){   [ "$1" = "lejepa" ] && echo 0.3  || echo 0.5; }
amax_of(){ [ "$1" = "lejepa" ] && echo 2.2  || echo 1.8; }
bar_of(){  [ "$1" = "lejepa" ] && echo 84.3 || echo 78.3; }

for B in lejepa pldm; do
  for f in caches/tr_${B}_fs1.pt caches/tr_${B}_fs5.pt metrics/window3_${B}_e005_g098_tr.pt; do
    [ -e "/workspace/$f" ] || die "missing /workspace/$f (run run_cleansplit.sh first)"
  done
done
log "train-only caches and clean values present"

# ---------------------------------------------------------------- train
log "12 trainings: 2 bases x 6 seeds, clean data, gamma ${G}"
i=0
for B in lejepa pldm; do for sd in $SEEDS; do
  A=/workspace/actors/lip4_final_${B}_s${sd}.pt
  [ -f "$A" ] && { i=$((i+1)); continue; }
  CUDA_VISIBLE_DEVICES=$(( i % 6 )) timeout 28800 python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/tr_${B}_fs5.pt \
    --cache-td /workspace/caches/tr_${B}_fs1.pt \
    --h5 "$SLIM" --wm "$(wm_of $B)" --pad-context \
    --init-value /workspace/metrics/window3_${B}_e005_g098_tr.pt \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --gamma "$G" --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight "$(mw_of $B)" --amax "$(amax_of $B)" \
    --replay-prob 0.5 --expand-weight 0 --steps 1000 \
    --actor-lr "$(lr_of $B)" --actor-lr-final "$(lrf_of $B)" --seed "$sd" \
    --out "$A" --out-value "/workspace/metrics/lip4_final_${B}_s${sd}_value.pt" \
    > "$L/train_${B}_s${sd}.log" 2>&1 || log "TRAIN FAILED ${B} s${sd}" &
  i=$((i + 1)); [ $((i % 12)) -eq 0 ] && wait
done; done
wait
log "actors: $(ls /workspace/actors/lip4_final_*_s?.pt 2>/dev/null | wc -l)/12"

# ---------------------------------------------------------------- card
ev(){ local gpu=$1 B=$2 nm=$3 seed=$4 actor=$5
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$(wm_of $B)" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=lip solver.actor_path="$actor" solver.rollout_compat=false \
    solver.batch_size=10 output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && log "  !! ${nm} no score: $(tail -1 "$L/${nm}.log" | cut -c1-60)"
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
meanof(){ local pre=$1 key=$2; local t=0 n=0 e
  for s in $REP; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"; }

log "carding 6 seeds x 6 reporting seeds on held-out episodes 8000:10000"
i=0
for B in lejepa pldm; do for sd in $SEEDS; do
  A=/workspace/actors/lip4_final_${B}_s${sd}.pt; [ -f "$A" ] || continue
  ( for s in $REP; do ev $(( i % 6 )) "$B" "fin_${B}_s${sd}_e${s}" "$s" "$A"; done ) &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done
wait

log "===================== FINAL, CLEAN, 6 SEEDS ====================="
for B in lejepa pldm; do
  vals10=""; vals05=""; nsc=0
  for sd in $SEEDS; do
    a=$(meanof "fin_${B}_s${sd}_e" held10); c=$(meanof "fin_${B}_s${sd}_e" held)
    n=$(grep -c "^fin_${B}_s${sd}_e.*held=[0-9]" "$SUM"); nsc=$((nsc + n))
    [ "$n" -eq 0 ] && { log "  ${B} s${sd}: NOTHING SCORED"; continue; }
    log "  ${B} s${sd}: @0.1 ${a} | @0.05 ${c}  (${n}/6)"
    vals10="${vals10} ${a}"; vals05="${vals05} ${c}"
  done
  read m10 sd10 <<< "$(echo $vals10 | awk '{for(j=1;j<=NF;j++){s+=$j;v[j]=$j;n++} m=s/n; for(j=1;j<=n;j++)q+=(v[j]-m)^2; printf "%.1f %.1f", m, (n>1?sqrt(q/(n-1)):0)}')"
  read m05 sd05 <<< "$(echo $vals05 | awk '{for(j=1;j<=NF;j++){s+=$j;v[j]=$j;n++} m=s/n; for(j=1;j<=n;j++)q+=(v[j]-m)^2; printf "%.1f %.1f", m, (n>1?sqrt(q/(n-1)):0)}')"
  log "  POOLED ${B}: @0.1 ${m10} +/- ${sd10}  |  @0.05 ${m05} +/- ${sd05}   (${nsc}/36 evals)"
  log "  bar $(bar_of $B) -> margin $(awk "BEGIN{printf \"%+.1f\", $m10-$(bar_of $B)}")  [LIP ~16 rollouts vs CEM 3000]"
done
log "FINAL6_DONE"
