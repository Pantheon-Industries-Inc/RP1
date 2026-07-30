#!/bin/bash
# WAVE A -- window-held LIP optimization on the 6xH200 pod (user directive
# 2026-07-30: factors are AMAX, MEAN-WEIGHT, LR and STEPS only; no bc-weight,
# no replay-prob, no privileged act-penalty).
#
#   setting  : h25, HELD@0.05, window3 critic, 1-frame policy conditioning,
#              plain LIP deploy (no restarts), CEM baselines at 300x10
#   protocol : every config at TRAINING SEEDS {0,1,2}; card = 3 x 6 eval seeds
#              (42-47). Single-seed screens are banned -- they misled twice
#              (amax 1.8 "won" at 1 seed, lost by 11 held points at 3).
#
# Reference row to beat (pre-loss values, being re-measured first):
#   LIP-window pooled 44.2 | Latent+CEM-window 42.7 | TD+CEM-window 11.7
# "A lot better" = pooled >= ~51 (clears the ~6-pt noise band at this n).
#
# OFAT around the centre (amax 2.2, mean-weight 0.1, actor-lr 3e-4->3e-5,
# 8000 steps). 9 configs x 3 seeds = 27 trainings, 3 per GPU on 6 GPUs.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
W3=/workspace/metrics/window3_lejepa.pt
L2W=/workspace/metrics/l2window3.pt
SUM=/workspace/results/summary_waveA_lejepa.csv; touch "$SUM"
L=/workspace/logs/waveA; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][waveA] $*"; }

BASE="--cache /workspace/caches/canon_lejepa_fs5.pt \
 --cache-td /workspace/caches/canon_lejepa_fs1.pt \
 --h5 $SLIM --wm $WM --pad-context --init-value $W3 \
 --arch v4 --iters 8 --horizon 5 --max-delta 12 \
 --batch 128 --n-step 50 \
 --expectile 0.1 --expectile-final 0.03 \
 --critic-lr 1e-3 --critic-lr-final 1e-4"
# centre: amax 2.2, mean-weight 0.1, actor-lr 3e-4->3e-5, steps 8000
declare -A CFG=(
  [centre]="--amax 2.2 --mean-weight 0.1 --actor-lr 3e-4 --actor-lr-final 3e-5 --steps 8000"
  [amax18]="--amax 1.8 --mean-weight 0.1 --actor-lr 3e-4 --actor-lr-final 3e-5 --steps 8000"
  [amax20]="--amax 2.0 --mean-weight 0.1 --actor-lr 3e-4 --actor-lr-final 3e-5 --steps 8000"
  [amax26]="--amax 2.6 --mean-weight 0.1 --actor-lr 3e-4 --actor-lr-final 3e-5 --steps 8000"
  [mw03]="--amax 2.2 --mean-weight 0.3 --actor-lr 3e-4 --actor-lr-final 3e-5 --steps 8000"
  [mw05]="--amax 2.2 --mean-weight 0.5 --actor-lr 3e-4 --actor-lr-final 3e-5 --steps 8000"
  [alr1e4]="--amax 2.2 --mean-weight 0.1 --actor-lr 1e-4 --actor-lr-final 1e-5 --steps 8000"
  [alr1e3]="--amax 2.2 --mean-weight 0.1 --actor-lr 1e-3 --actor-lr-final 1e-4 --steps 8000"
  [steps16k]="--amax 2.2 --mean-weight 0.1 --actor-lr 3e-4 --actor-lr-final 3e-5 --steps 16000"
)
ORDER="centre amax18 amax20 amax26 mw03 mw05 alr1e4 alr1e3 steps16k"

tr(){ local tag=$1 seed=$2 gpu=$3
  local A=/workspace/actors/lip4_wa_${tag}_s${seed}.pt
  [ -f "$A" ] && { log "train ${tag} s${seed}: exists"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$PLAN/train_lip_ac.py" \
    $BASE ${CFG[$tag]} --seed "$seed" \
    --out "$A" --out-value "/workspace/metrics/lip4_wa_${tag}_s${seed}_value.pt" \
    > "$L/train_${tag}_s${seed}.log" 2>&1 \
    && log "train ${tag} s${seed} DONE" || log "train ${tag} s${seed} FAILED"
}
ev(){ local gpu=$1 nm=$2 seed=$3; shift 3
  grep -q "^${nm}," "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver.batch_size=10 "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h l hist
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  l=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  hist=$(grep -c "hist-inject" "$L/${nm}.log")
  echo "${nm},held=${h:-FAIL},latched=${l:-FAIL},hist=${hist}" >> "$SUM"
}
mean6(){ local pre=$1; local t=0 n=0 e
  for s in 42 43 44 45 46 47; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "held=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

# ---------------------------------------------------------------- reference row
log "reference row: the two CEM-window baselines (6 eval seeds each)"
( for s in 42 43 44 45 46 47; do ev 4 "ref_l2_s${s}" $s solver=cem solver.n_steps=10 "+metric=$L2W"; done
  log "REF Latent+CEM-window: HELD $(mean6 "ref_l2_s") [pre-loss 42.7]" ) &
( for s in 42 43 44 45 46 47; do ev 5 "ref_td_s${s}" $s solver=cem solver.n_steps=10 "+metric=$W3"; done
  log "REF TD+CEM-window: HELD $(mean6 "ref_td_s") [pre-loss 11.7]" ) &
wait

# ---------------------------------------------------------------- trainings
log "wave A: 9 configs x seeds {0,1,2} = 27 trainings, 3 per GPU"
i=0
for tag in $ORDER; do
  for seed in 0 1 2; do
    tr "$tag" "$seed" $((i % 6)) &
    i=$((i + 1))
    [ $((i % 18)) -eq 0 ] && wait
  done
done
wait
log "wave A trainings drained"

# ---------------------------------------------------------------- evals
runcfg(){ local gpu=$1 tag=$2
  local t=0 n=0
  for seed in 0 1 2; do
    A=/workspace/actors/lip4_wa_${tag}_s${seed}.pt; [ -f "$A" ] || continue
    for s in 42 43 44 45 46 47; do
      ev $gpu "wa_${tag}_s${seed}_e${s}" $s solver=lip solver.actor_path="$A" solver.rollout_compat=false
    done
    v=$(mean6 "wa_${tag}_s${seed}_e"); log "  CARD6 ${tag} s${seed}: HELD ${v}"
    t=$(awk "BEGIN{print $t+$v}"); n=$((n+1))
  done
  [ "$n" -gt 0 ] && log "POOLED ${tag}: HELD $(awk "BEGIN{printf \"%.1f\", $t/$n}") over ${n} seeds"
}
g=0
for tag in $ORDER; do runcfg $((g % 6)) "$tag" & g=$((g + 1)); [ $((g % 6)) -eq 0 ] && wait; done
wait
log "--- wave A summary (HELD, pooled over 3 training seeds) ---"
for tag in $ORDER; do
  t=0; n=0
  for seed in 0 1 2; do
    v=$(mean6 "wa_${tag}_s${seed}_e"); [ "$v" = "0.0" ] && continue
    t=$(awk "BEGIN{print $t+$v}"); n=$((n+1))
  done
  [ "$n" -gt 0 ] && log "  ${tag}: $(awk "BEGIN{printf \"%.1f\", $t/$n}") (n=${n} seeds)"
done
log "WAVEA_DONE"
