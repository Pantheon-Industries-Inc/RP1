#!/bin/bash
# Complete the TD grid at 6 EVAL SEEDS for every arm, then run stage 2.
#
# Why this replaces the original stage-1 logic: the 3-seed screen is noise-
# dominated. Observed so far -- tau 0.05 spans 35.3 (n=25) to 43.3 (n=5) and
# n=5 spans 32.0 (tau 0.1) to 43.3 (tau 0.05), i.e. the within-row spread equals
# the whole grid's spread and is non-monotone in both hypers. A cell at 3 seeds
# x n=50 carries roughly +-8 points, which is every difference in the grid.
# Picking a "winner" from that would just be selecting noise, and stage 2 would
# then build eight LIP actors on it.
#
# 12 arms x 6 seeds, idempotent (cached cells skipped), ~10 min on 4 GPUs.
# A hyper is only called a winner if it clears the canonical tau0.1/n50 arm by
# more than the 6-seed noise band (~5.7 pts); otherwise stage 2 uses the
# canonical TD and we report "no TD hyper beats the default".
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
C1=/workspace/caches/canon_lejepa_fs1.pt
C5=/workspace/caches/canon_lejepa_fs5.pt
TSUM=/workspace/results/summary_tdsweep_lejepa.csv; touch "$TSUM"
LSUM=/workspace/results/summary_lipopt_lejepa.csv; touch "$LSUM"
L=/workspace/logs/tdsweep; mkdir -p "$L"
NOISE=5.7
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][tdgrid6] $*"; }

ev(){ local gpu=$1 sum=$2 nm=$3 seed=$4; shift 4
  grep -q "^${nm}," "$sum" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver.batch_size=10 "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h l
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  l=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  echo "${nm},held=${h:-FAIL},latched=${l:-FAIL}" >> "$sum"
}
mean6(){ local sum=$1 pre=$2; local t=0 n=0 e
  for s in 42 43 44 45 46 47; do
    e=$(grep "^${pre}${s}," "$sum" | tail -1 | grep -oE "held=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

ARMS="t005n5 t005n25 t005n50 t005n100 t01n5 t01n25 t01n50 t01n100 t02n5 t02n25 t02n50 t02n100"
run(){ local gpu=$1; shift
  for tag in "$@"; do
    M=/workspace/metrics/tdsw_${tag}_lejepa.pt; [ -f "$M" ] || continue
    for s in 42 43 44 45 46 47; do ev $gpu "$TSUM" "tds_${tag}_e${s}" $s solver=cem solver.n_steps=10 "+metric=$M"; done
    log "GRID6 ${tag}: HELD $(mean6 "$TSUM" "tds_${tag}_e")"
  done
}
run 0 t005n5 t005n25 t005n50 & run 1 t005n100 t01n5 t01n25 &
run 2 t01n50 t01n100 t02n5 & run 3 t02n25 t02n50 t02n100 &
wait

CANON_V=$(mean6 "$TSUM" "tds_t01n50_e")
BEST=t01n50; BESTV=$CANON_V
log "--- full grid, 6 eval seeds (canonical tau0.1/n50 = ${CANON_V}) ---"
for tag in $ARMS; do
  v=$(mean6 "$TSUM" "tds_${tag}_e"); [ "$v" = "0.0" ] && continue
  log "  ${tag}: HELD ${v}"
  awk "BEGIN{exit !($v > $BESTV)}" && { BEST=$tag; BESTV=$v; }
done
DELTA=$(awk "BEGIN{printf \"%.1f\", $BESTV - $CANON_V}")
if awk "BEGIN{exit !($DELTA > $NOISE)}"; then
  log "TD WINNER: ${BEST} at ${BESTV} (+${DELTA} over canonical, clears the ${NOISE} noise band)"
  TD=/workspace/metrics/tdsw_${BEST}_lejepa.pt
else
  log "NO TD HYPER CLEARS THE NOISE BAND: best ${BEST} ${BESTV} vs canonical ${CANON_V} (+${DELTA} <= ${NOISE})"
  log "=> stage 2 uses the CANONICAL TD; report: TD hypers are not the lever"
  BEST=t01n50; TD=/workspace/metrics/tdsw_t01n50_lejepa.pt
fi
echo "$BEST" > /workspace/_TD_BEST
log "stage-2 critic: $TD"

# ============================================================ STAGE 2
log "STAGE 2: LIP on that critic -- amax x act-penalty, held-ranked, 6-seed cards"
BASE="--cache $C5 --cache-td $C1 --h5 $CANON --wm $WM --pad-context \
 --init-value $TD --arch v4 --iters 8 --horizon 5 --max-delta 12 \
 --steps 8000 --batch 128 --n-step 50 \
 --expectile 0.1 --expectile-final 0.03 \
 --critic-lr 1e-3 --critic-lr-final 1e-4 \
 --actor-lr 3e-4 --actor-lr-final 3e-5"
declare -A CFG=(
  [a22ap0]="--amax 2.2"
  [a22ap05]="--amax 2.2 --act-penalty 0.05"
  [a18ap05]="--amax 1.8 --act-penalty 0.05"
  [a26ap0]="--amax 2.6"
)
i=0
for tag in a22ap0 a22ap05 a18ap05 a26ap0; do
  for seed in 0 1; do
    A=/workspace/actors/lip4_opt_${tag}_s${seed}.pt
    if [ ! -f "$A" ]; then
      CUDA_VISIBLE_DEVICES=$((i % 4)) timeout 43200 python3 "$PLAN/train_lip_ac.py" \
        $BASE ${CFG[$tag]} --seed "$seed" \
        --out "$A" --out-value "/workspace/metrics/lip4_opt_${tag}_s${seed}_value.pt" \
        > "$L/lip_${tag}_s${seed}.log" 2>&1 \
        && log "LIP ${tag} s${seed} DONE" || log "LIP ${tag} s${seed} FAILED" &
    fi
    i=$((i + 1))
  done
done
wait
log "LIP trainings drained"
runarm(){ local gpu=$1 tag=$2
  for seed in 0 1; do
    A=/workspace/actors/lip4_opt_${tag}_s${seed}.pt; [ -f "$A" ] || continue
    for s in 42 43 44 45 46 47; do
      ev $gpu "$LSUM" "opt_${tag}_s${seed}_e${s}" $s solver=lip solver.actor_path="$A" solver.rollout_compat=false
    done
    log "CARD6 LIP ${tag} s${seed}: HELD $(mean6 "$LSUM" "opt_${tag}_s${seed}_e")"
  done
}
runarm 0 a22ap0 & runarm 1 a22ap05 & runarm 2 a18ap05 & runarm 3 a26ap0 &
wait
log "BASELINE to beat: LIP terminal pad s0 57.0 / pooled 48.6 (n=6) | Latent+CEM 41.7 | TD+CEM 35.7"
log "TDGRID6_DONE"
