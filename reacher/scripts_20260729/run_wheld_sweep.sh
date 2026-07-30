#!/bin/bash
# WINDOW-HELD LIP optimization, wave A (user directive 2026-07-29/30):
#   setting  = h25, HELD@0.05, window3 critic, plain deploy (no restarts)
#   rules    = NO privileged terms (no act-penalty); standard LIPv4 knobs only
#   protocol = ALL configs at training seeds {0,1,2}; card = 3 x 6 eval seeds
#              (single-seed screens are banned -- they misled twice)
#
# References (held, h25, n=6/actor): LIP-window centre pooled 44.2
# (35.0/54.0/43.7), Latent+CEM-window 42.7, TD+CEM-window 11.7.
# Target: clearly above 42.7 -- i.e. pooled >= ~51 to clear the noise band.
#
# Wave A (4 configs x 3 seeds = 12 trainings, 3 per GPU):
#   mw05      --mean-weight 0.5    path-mean emphasis: arrive early + dwell,
#                                  expressed only through the existing loss
#   steps16k  --steps 16000        the 19-pt training-seed spread suggests
#                                  under-training
#   replay03  --replay-prob 0.3    train on own imagined windows (deploy dist)
#   bc01      --bc-weight 0.1      trust region to data actions (under-actuation)
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
W3=/workspace/metrics/window3_lejepa.pt
SUM=/workspace/results/summary_wheld_lejepa.csv; touch "$SUM"
L=/workspace/logs/wheld; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][wheld] $*"; }

BASE="--cache /workspace/caches/canon_lejepa_fs5.pt \
 --cache-td /workspace/caches/canon_lejepa_fs1.pt \
 --h5 $CANON --wm $WM --pad-context --init-value $W3 \
 --arch v4 --amax 2.2 --iters 8 --horizon 5 --max-delta 12 \
 --steps 8000 --batch 128 --n-step 50 \
 --expectile 0.1 --expectile-final 0.03 \
 --critic-lr 1e-3 --critic-lr-final 1e-4 \
 --actor-lr 3e-4 --actor-lr-final 3e-5"
declare -A ARMS=(
  [mw05]="--mean-weight 0.5"
  [steps16k]="--steps 16000"
  [replay03]="--replay-prob 0.3"
  [bc01]="--bc-weight 0.1"
)

tr(){ local tag=$1 seed=$2 gpu=$3
  local A=/workspace/actors/lip4_wh_${tag}_s${seed}.pt
  [ -f "$A" ] && { log "train ${tag} s${seed}: exists"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$PLAN/train_lip_ac.py" \
    $BASE ${ARMS[$tag]} --seed "$seed" \
    --out "$A" --out-value "/workspace/metrics/lip4_wh_${tag}_s${seed}_value.pt" \
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

# 12 trainings: config x seed pinned so each GPU carries 3 (one config each)
log "wave A trainings: 4 configs x seeds {0,1,2}"
g=0
for tag in mw05 steps16k replay03 bc01; do
  for seed in 0 1 2; do tr "$tag" "$seed" $g & done
  g=$((g + 1))
done
wait
log "wave A trainings drained"

runcfg(){ local gpu=$1 tag=$2
  local t=0 n=0
  for seed in 0 1 2; do
    A=/workspace/actors/lip4_wh_${tag}_s${seed}.pt; [ -f "$A" ] || continue
    for s in 42 43 44 45 46 47; do ev $gpu "wh_${tag}_s${seed}_e${s}" $s "$A"; done
    v=$(mean6 "wh_${tag}_s${seed}_e")
    log "CARD6 ${tag} s${seed}: HELD ${v}"
    t=$(awk "BEGIN{print $t+$v}"); n=$((n+1))
  done
  [ "$n" -gt 0 ] && log "POOLED ${tag}: HELD $(awk "BEGIN{printf \"%.1f\", $t/$n}") over ${n} train seeds [centre 44.2 | Latent+CEM-w 42.7]"
}
runcfg 0 mw05 & runcfg 1 steps16k & runcfg 2 replay03 & runcfg 3 bc01 &
wait
log "WHELD_WAVEA_DONE"
