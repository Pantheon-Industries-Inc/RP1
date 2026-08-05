#!/bin/bash
# ACT-PENALTY x AMAX sweep -- pushing the one lever that reproducibly moved HELD.
#
# Stage 2 result (terminal critic, held, n=6 x 2 train seeds):
#   amax2.2 + ap0.05 = 50.7   vs   amax2.2 + ap0 = 44.0    => +6.7, both seeds
#   amax1.8 + ap0.05 = 53.8  (best so far; refs: prev best LIP pooled 48.6,
#                             Latent+CEM 41.7, TD+CEM 35.7)
# The penalty asks the FINAL action block to be small, which is the only place
# settling is expressible: the 1-frame quasimetric separates stopped-from-moving
# at AUC 0.505 (chance), so no latent-space value can ask for it.
#
# Grid: amax {1.6, 1.8, 2.0} x act-penalty {0.02, 0.05, 0.1, 0.2}, 1 train seed
# for screening at 6 eval seeds; the top-2 configs then get 2 more train seeds
# so the winner is not a single-seed artifact (the amax-1.8 lesson).
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
TD=/workspace/metrics/tdsw_t01n50_lejepa.pt
C1=/workspace/caches/canon_lejepa_fs1.pt
C5=/workspace/caches/canon_lejepa_fs5.pt
SUM=/workspace/results/summary_actpen_lejepa.csv; touch "$SUM"
L=/workspace/logs/actpen; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][actpen] $*"; }

BASE="--cache $C5 --cache-td $C1 --h5 $CANON --wm $WM --pad-context \
 --init-value $TD --arch v4 --iters 8 --horizon 5 --max-delta 12 \
 --steps 8000 --batch 128 --n-step 50 \
 --expectile 0.1 --expectile-final 0.03 \
 --critic-lr 1e-3 --critic-lr-final 1e-4 \
 --actor-lr 3e-4 --actor-lr-final 3e-5"

tr(){ local amax=$1 ap=$2 seed=$3 gpu=$4
  local tag=a${amax/./}p${ap/./}
  local A=/workspace/actors/lip4_ap_${tag}_s${seed}.pt
  [ -f "$A" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$PLAN/train_lip_ac.py" \
    $BASE --amax "$amax" --act-penalty "$ap" --seed "$seed" \
    --out "$A" --out-value "/workspace/metrics/lip4_ap_${tag}_s${seed}_value.pt" \
    > "$L/train_${tag}_s${seed}.log" 2>&1 \
    && log "train ${tag} s${seed} DONE" || log "train ${tag} s${seed} FAILED"
}
ev(){ local gpu=$1 nm=$2 seed=$3 actor=$4
  grep -q "^${nm}," "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=lip solver.actor_path="$actor" solver.rollout_compat=false \
    solver.batch_size=10 output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h l
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  l=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  echo "${nm},held=${h:-FAIL},latched=${l:-FAIL}" >> "$SUM"
}
mean6(){ local pre=$1; local t=0 n=0 e
  for s in 42 43 44 45 46 47; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "held=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

# ---- wave 1: 12 configs, 1 train seed each, 3 per GPU
log "wave 1: amax {1.6,1.8,2.0} x act-penalty {0.02,0.05,0.1,0.2}, seed 0"
i=0
for amax in 1.6 1.8 2.0; do for ap in 0.02 0.05 0.1 0.2; do
  tr "$amax" "$ap" 0 $((i % 4)) & i=$((i + 1))
  [ $((i % 4)) -eq 0 ] && wait
done; done
wait
log "wave 1 trainings drained"

runcfg(){ local gpu=$1; shift
  for spec in "$@"; do
    set -- $spec; local amax=$1 ap=$2 seed=$3
    local tag=a${amax/./}p${ap/./}
    local A=/workspace/actors/lip4_ap_${tag}_s${seed}.pt
    [ -f "$A" ] || continue
    for s in 42 43 44 45 46 47; do ev $gpu "ap_${tag}_s${seed}_e${s}" $s "$A"; done
    log "CARD6 amax ${amax} ap ${ap} s${seed}: HELD $(mean6 "ap_${tag}_s${seed}_e")"
  done
}
runcfg 0 "1.6 0.02 0" "1.6 0.05 0" "1.6 0.1 0" &
runcfg 1 "1.6 0.2 0" "1.8 0.02 0" "1.8 0.05 0" &
runcfg 2 "1.8 0.1 0" "1.8 0.2 0" "2.0 0.02 0" &
runcfg 3 "2.0 0.05 0" "2.0 0.1 0" "2.0 0.2 0" &
wait

# ---- pick top 2 by held, add 2 more training seeds each
BEST=""; BESTV=0; SEC=""; SECV=0
for amax in 1.6 1.8 2.0; do for ap in 0.02 0.05 0.1 0.2; do
  tag=a${amax/./}p${ap/./}
  v=$(mean6 "ap_${tag}_s0_e"); [ "$v" = "0.0" ] && continue
  log "  amax ${amax} ap ${ap}: HELD ${v}"
  if awk "BEGIN{exit !($v > $BESTV)}"; then SEC=$BEST; SECV=$BESTV; BEST="$amax $ap"; BESTV=$v
  elif awk "BEGIN{exit !($v > $SECV)}"; then SEC="$amax $ap"; SECV=$v; fi
done; done
log "TOP: [${BEST}] ${BESTV} | RUNNER-UP: [${SEC}] ${SECV}"

i=0
for cfg in "$BEST" "$SEC"; do
  [ -z "$cfg" ] && continue
  set -- $cfg
  for seed in 1 2; do tr "$1" "$2" "$seed" $((i % 4)) & i=$((i + 1)); done
done
wait
log "confirmation trainings drained"
for cfg in "$BEST" "$SEC"; do
  [ -z "$cfg" ] && continue
  set -- $cfg; amax=$1; ap=$2; tag=a${amax/./}p${ap/./}
  for seed in 1 2; do
    A=/workspace/actors/lip4_ap_${tag}_s${seed}.pt; [ -f "$A" ] || continue
    for s in 42 43 44 45 46 47; do ev $((seed % 4)) "ap_${tag}_s${seed}_e${s}" $s "$A"; done
  done
  t=0; n=0
  for seed in 0 1 2; do
    v=$(mean6 "ap_${tag}_s${seed}_e"); [ "$v" = "0.0" ] && continue
    t=$(awk "BEGIN{print $t+$v}"); n=$((n+1))
    log "  amax ${amax} ap ${ap} s${seed}: HELD ${v}"
  done
  log "POOLED amax ${amax} ap ${ap}: HELD $(awk "BEGIN{printf \"%.1f\", ($n?$t/$n:0)}") over ${n} train seeds"
done
log "refs: a18ap05 53.8 | prev best LIP pooled 48.6 | Latent+CEM 41.7 | TD+CEM 35.7"
log "ACTPEN_DONE"
