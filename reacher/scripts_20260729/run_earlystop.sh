#!/bin/bash
# EARLY STOPPING ON HELD@0.05 -- select the actor checkpoint by the deployment
# metric instead of by the training objective or a fixed step count.
#
# Motivation: corr(E_final, HELD) = +0.583 across 20 actors. Training longer /
# reaching a lower imagined cost makes real performance WORSE, because the world
# model's 25-step open-loop error (~0.11 rad) is larger than the 0.05 rad
# tolerance, so past some point the actor is optimising fiction. Seed 0 is simply
# the seed that optimises hardest: E_final 0.78-1.36 and held 4.7-16.3, while
# seeds 1/2 sit at E_final 1.0-1.9 and held 41.7-57.7.
#
# TRAIN/SELECT/TEST HYGIENE. Snapshots are scored on SELECTION eval seeds
# {50,51} -- draws never used anywhere in this campaign -- and only the winner is
# then carded on the REPORTING seeds {42..47}. Selecting and reporting on the
# same draws would be train-on-test and would manufacture a win.
#
# Configs: the centre recipe plus the two levers grid C is testing, each at
# training seeds {0,1,2}, snapshotting every 1000 of 8000 steps (7 snapshots +
# final). Produces a held-vs-step CURVE per actor, which is the direct
# visualisation of the over-optimisation, and a selected checkpoint per actor.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
W3=/workspace/metrics/window3_lejepa.pt
SUM=/workspace/results/summary_earlystop_lejepa.csv; touch "$SUM"
L=/workspace/logs/earlystop; mkdir -p "$L"
SEL_SEEDS="50 51"
REP_SEEDS="42 43 44 45 46 47"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][estop] $*"; }

log "waiting for grid C to release the GPUs"
while ps -eo args --no-headers | grep -q "[r]un_gridC.sh"; do sleep 120; done
log "GPUs free -- early-stopping study starts"

BASE="--cache /workspace/caches/canon_lejepa_fs5.pt \
 --cache-td /workspace/caches/canon_lejepa_fs1.pt \
 --h5 /workspace/reacher_slim.h5 --wm $WM --pad-context --init-value $W3 \
 --arch v4 --amax 2.2 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
 --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
 --mean-weight 0.1 --lambda-base 0.5 --steps 8000 --ckpt-every 1000"

declare -A CFG=(
  [base]="--lambda-schedule uniform --actor-lr 3e-4 --actor-lr-final 3e-5"
  [lr1e4]="--lambda-schedule uniform --actor-lr 1e-4 --actor-lr-final 1e-5"
  [gearly]="--lambda-schedule geom-early --actor-lr 3e-4 --actor-lr-final 3e-5"
)
ORDER="base lr1e4 gearly"

train_one(){ local tag=$1 seed=$2 gpu=$3
  local A=/workspace/actors/lip4_es_${tag}_s${seed}.pt
  [ -f "$A" ] && { log "train ${tag} s${seed}: exists"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$PLAN/train_lip_ac.py" \
    $BASE ${CFG[$tag]} --seed "$seed" \
    --out "$A" --out-value "/workspace/metrics/lip4_es_${tag}_s${seed}_value.pt" \
    > "$L/train_${tag}_s${seed}.log" 2>&1 \
    && log "train ${tag} s${seed} DONE" || log "train ${tag} s${seed} FAILED"
}

ev(){ local gpu=$1 nm=$2 seed=$3 actor=$4
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=lip solver.actor_path="$actor" solver.rollout_compat=false \
    solver.batch_size=10 output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
meanof(){ local pre=$1 key=$2; shift 2; local t=0 n=0 e
  for s in "$@"; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

log "trainings: 3 configs x seeds {0,1,2}, snapshots every 1000 steps"
i=0
for tag in $ORDER; do for seed in 0 1 2; do
  train_one "$tag" "$seed" $((i % 6)) & i=$((i + 1))
done; done
wait
log "trainings drained"

# ---- selection: score every snapshot on the SELECTION seeds only
log "selection pass on held-out seeds {${SEL_SEEDS}} -- serial (concurrent EGL aborts)"
for tag in $ORDER; do
  for seed in 0 1 2; do
    A=/workspace/actors/lip4_es_${tag}_s${seed}.pt
    [ -f "$A" ] || continue
    best=""; bestv=-1; curve=""
    for st in 1000 2000 3000 4000 5000 6000 7000 final; do
      C=$A; [ "$st" != "final" ] && C="${A}.step${st}.pt"
      [ -f "$C" ] || continue
      for s in $SEL_SEEDS; do ev 3 "es_${tag}_s${seed}_${st}_sel${s}" $s "$C"; done
      v=$(meanof "es_${tag}_s${seed}_${st}_sel" held $SEL_SEEDS)
      curve="${curve} ${st}:${v}"
      awk "BEGIN{exit !($v > $bestv)}" && { bestv=$v; best=$C; bests=$st; }
    done
    log "  CURVE ${tag} s${seed}:${curve}"
    log "  SELECTED ${tag} s${seed}: step ${bests} (sel-held ${bestv})"
    echo "${tag}_s${seed},selected=${bests},sel_held=${bestv}" >> /workspace/results/earlystop_selection.csv
    # ---- report the selected checkpoint on the untouched reporting seeds
    for s in $REP_SEEDS; do ev 4 "es_${tag}_s${seed}_SEL_e${s}" $s "$best"; done
    log "  CARD ${tag} s${seed} (early-stopped @${bests}): HELD $(meanof "es_${tag}_s${seed}_SEL_e" held $REP_SEEDS) | @0.1 $(meanof "es_${tag}_s${seed}_SEL_e" held10 $REP_SEEDS)"
    # ---- and the un-stopped final, same seeds, for the paired comparison
    for s in $REP_SEEDS; do ev 5 "es_${tag}_s${seed}_FIN_e${s}" $s "$A"; done
    log "  CARD ${tag} s${seed} (trained to 8000):        HELD $(meanof "es_${tag}_s${seed}_FIN_e" held $REP_SEEDS) | @0.1 $(meanof "es_${tag}_s${seed}_FIN_e" held10 $REP_SEEDS)"
  done
done
log "EARLYSTOP_DONE -- bar: Latent+CEM-window 44.7 @0.05 / 84.3 @0.1"
