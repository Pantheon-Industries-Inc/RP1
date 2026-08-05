#!/bin/bash
# amax 1.5 companion to the act-penalty sweep (user request 2026-07-29).
# amax 1.6 is already covered by run_actpen_sweep.sh; this adds the 1.5 row so
# the low end of the torque cap is bracketed. Runs CONCURRENTLY with the main
# sweep -- 2 trainings per H100 fits in memory (~15G each of 80G) and only halves
# throughput, which beats waiting ~6h for a free GPU.
#
# Same recipe as the main sweep: terminal critic (canonical tau0.1/n50, since the
# 6-seed TD grid found no hyper clears the noise band), pad-context, iters 8,
# 1 train seed for screening at 6 eval seeds.
# Refs (held, h25): a18ap05 53.8 | prev best LIP pooled 48.6 | Latent+CEM 41.7.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
TD=/workspace/metrics/tdsw_t01n50_lejepa.pt
SUM=/workspace/results/summary_actpen15_lejepa.csv; touch "$SUM"
L=/workspace/logs/actpen15; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][actpen15] $*"; }

BASE="--cache /workspace/caches/canon_lejepa_fs5.pt \
 --cache-td /workspace/caches/canon_lejepa_fs1.pt \
 --h5 $CANON --wm $WM --pad-context --init-value $TD \
 --arch v4 --amax 1.5 --iters 8 --horizon 5 --max-delta 12 \
 --steps 8000 --batch 128 --n-step 50 \
 --expectile 0.1 --expectile-final 0.03 \
 --critic-lr 1e-3 --critic-lr-final 1e-4 \
 --actor-lr 3e-4 --actor-lr-final 3e-5"

tr(){ local ap=$1 gpu=$2
  local A=/workspace/actors/lip4_ap_a15p${ap/./}_s0.pt
  [ -f "$A" ] && { log "train ap${ap}: exists"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$PLAN/train_lip_ac.py" \
    $BASE --act-penalty "$ap" --seed 0 \
    --out "$A" --out-value "/workspace/metrics/lip4_ap_a15p${ap/./}_s0_value.pt" \
    > "$L/train_a15p${ap/./}.log" 2>&1 \
    && log "train amax1.5 ap${ap} DONE" || log "train amax1.5 ap${ap} FAILED"
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

log "training amax 1.5 x act-penalty {0.02,0.05,0.1,0.2}"
i=0
for ap in 0.02 0.05 0.1 0.2; do tr "$ap" $((i % 4)) & i=$((i + 1)); done
wait
log "trainings drained"

runap(){ local gpu=$1 ap=$2
  local tag=a15p${ap/./}
  local A=/workspace/actors/lip4_ap_${tag}_s0.pt
  [ -f "$A" ] || { log "eval ap${ap}: no actor"; return 0; }
  for s in 42 43 44 45 46 47; do ev $gpu "ap_${tag}_s0_e${s}" $s "$A"; done
  log "CARD6 amax 1.5 ap ${ap}: HELD $(mean6 "ap_${tag}_s0_e")"
}
runap 0 0.02 & runap 1 0.05 & runap 2 0.1 & runap 3 0.2 &
wait
log "refs: a18ap05 53.8 | prev best LIP pooled 48.6 | Latent+CEM 41.7 | TD+CEM 35.7"
log "ACTPEN15_DONE"
