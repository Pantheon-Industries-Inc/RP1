#!/bin/bash
# DYNA round 2, end to end, on the rebuilt H200 pod.
#
# mix -> fine-tune WM -> fresh caches -> fresh window value -> LIP at the winning
# 1000-step recipe -> cards, all compared against the SAME arms on the base WM.
#
# WHY DYNA IS THE RIGHT LEVER HERE. Every cost we can deploy is near-chance at
# seeing settling (stopped-vs-moving AUC 0.505-0.63), and one reason is that the
# random-play dataset barely contains settling: only 0.4% of at-goal states are
# still at goal 25 steps later. A goal-reaching policy produces exactly that
# behaviour, so fine-tuning the WM on it should sharpen the imagined dynamics
# near the goal -- which is where held-at-end is decided. Falsifiable: if held
# does not move, near-goal WM fidelity is not the binding constraint.
#
# SPLIT DISCIPLINE (the round-1 mix leaked 400k eval-pool rows, 10.1%):
#   expert side   = episodes 0:1500   (verified, eval pool excluded)
#   on-policy side= collected with +eval.ep_range=0:8000
#   evaluation    = episodes 8000:10000, untouched by either
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HF_HOME=/root/hf OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
TRM=/workspace/swm_cem/scripts/trm
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
BASE_WM=/workspace/swm_home/checkpoints/lejepa_reacher
NAME=dyna_r2_lejepa
MIX=/workspace/dyna_data/mix_r2_5050.lance
STATS=/workspace/reacher_action_stats.json
SUM=/workspace/results/summary_dyna_lejepa.csv; touch "$SUM"
L=/workspace/logs/dyna; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][dyna] $*"; }
die(){ log "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------- 0. wait for the sweep
# Dyna must run on the configuration the joint sweep actually selected, so it is
# gated on the sweep reaching stage 3. Ceiling 6 h: a dead sweep should fail
# loudly here rather than silently park the GPUs.
# written by the min-margin rescore, NOT by the sweep's cross-base-mean
# stage 3 -- the objective is "both bases above their own Latent+CEM bar"
BESTF=/workspace/_BEST_FINAL_lejepa
if [ ! -f "$BESTF" ]; then
  log "waiting for the min-margin winner at ${BESTF} (ceiling 6 h)"
  waited=0
  while [ ! -f "$BESTF" ]; do
    sleep 120; waited=$((waited + 120))
    [ $((waited % 1800)) -eq 0 ] && log "  still waiting, ${waited}s elapsed"
    [ "$waited" -ge 21600 ] && die "sweep never produced ${BESTF}"
  done
fi
read -r _b SW_AMAX SW_LR SW_EX SW_RP SW_EW < "$BESTF"
case "$SW_EX" in 005) SW_EXV=0.05;; 01) SW_EXV=0.1;; *) die "bad expectile tag ${SW_EX}";; esac
SW_LRF=1e-4; [ "$SW_LR" = "3e-4" ] && SW_LRF=3e-5
log "swept winner for lejepa: amax ${SW_AMAX} lr ${SW_LR} expectile ${SW_EXV} replay ${SW_RP} expand ${SW_EW}"

# ---------------------------------------------------------------- action stats pin
if [ ! -f "$STATS" ]; then
  python3 - <<'PYEOF' || die "stats json failed"
import json, h5py, hdf5plugin, numpy as np
with h5py.File("/workspace/reacher_slim.h5") as h:
    a = h["action"][:]
mu, sd = np.nanmean(a, 0), np.nanstd(a, 0) + 1e-6
json.dump({"mean": mu.tolist(), "std": sd.tolist()},
          open("/workspace/reacher_action_stats.json", "w"))
print("stats:", mu.tolist(), sd.tolist())
PYEOF
fi
log "action stats pinned: $(head -c 90 $STATS)"

# ---------------------------------------------------------------- 1. mix
if [ ! -d "$MIX" ]; then
  log "building 50/50 mix (expert 0:1500 + on-policy, ~1.7x duplication)"
  ONP=$(ls -d /workspace/dyna_data/onp_*.lance 2>/dev/null | tr '\n' ' ')
  [ -n "$ONP" ] || die "no on-policy lances"
  python3 "$PLAN/build_dyna_mix.py" \
    --expert /workspace/dyna_data/expert_0_1500.lance \
    --onpolicy $ONP --out "$MIX" --onpolicy-frac 0.5 \
    > "$L/mix.log" 2>&1 || die "mix build failed (see mix.log)"
  grep -q "BUILD_MIX_DONE" "$L/mix.log" || die "mix incomplete"
fi
log "mix ready: $(grep -oE 'rows=[0-9]+|dup K=[0-9]+' $L/mix.log | tr '\n' ' ')"

# ---------------------------------------------------------------- 2. fine-tune
FT=/workspace/swm_home/checkpoints/${NAME}
if [ ! -f "$FT/weights.pt" ]; then
  log "fine-tuning from the base WM (lr 1e-5, 2 epochs, action-stats pinned)"
  cd /workspace/swm_cem
  INIT_WEIGHTS=$BASE_WM/weights.pt CUDA_VISIBLE_DEVICES=0 timeout 86400 python3 \
    scripts/train/lewm_expert.py data=dmc \
    data.dataset.name="$MIX" \
    "data.dataset.keys_to_load=[pixels,action]" \
    "data.dataset.keys_to_cache=[action]" \
    num_workers=16 optimizer.lr=1e-5 trainer.max_epochs=2 trainer.devices=1 \
    output_model_name="$NAME" subdir="$NAME" \
    +action_stats_pin="$STATS" wandb.enabled=false \
    > "$L/ft.log" 2>&1 || log "fine-tune returned nonzero -- inspect ft.log"
  cd /workspace
  # normalise: trainer writes weights_epoch_N.pt and the loader refuses ambiguity
  if [ ! -f "$FT/weights.pt" ]; then
    latest=$(ls -t "$FT"/weights_epoch_*.pt 2>/dev/null | head -1)
    [ -n "$latest" ] || die "no checkpoint produced"
    mkdir -p "$FT/_extra"; for f in "$FT"/weights_epoch_*.pt; do mv "$f" "$FT/_extra/"; done
    cp "$FT/_extra/$(basename $latest)" "$FT/weights.pt"
    log "normalised $(basename $latest) -> weights.pt"
  fi
  # planners need the ARCHITECTURE config, not the training config
  cp "$BASE_WM/config.json" "$FT/config.json"
fi
log "fine-tuned WM: $FT"

# ---------------------------------------------------------------- 3. caches + window value
C1=/workspace/caches/dyna_lejepa_fs1.pt
C5=/workspace/caches/dyna_lejepa_fs5.pt
# read at point of use, not at the gate: the value-steps ladder is still running
# when Dyna starts, and Dyna does not need this until after the fine-tune
SW_VSTEPS=6000
[ -f /workspace/_VSTEPS ] && SW_VSTEPS=$(cat /workspace/_VSTEPS)
W3=/workspace/metrics/window3_dyna_lejepa_e${SW_EX}_st${SW_VSTEPS}.pt
[ -f "$C1" ] || { log "fs1 cache on the fine-tuned WM"; CUDA_VISIBLE_DEVICES=0 python3 \
  "$TRM/cache_latents.py" --wm "$FT" --dataset "$CANON" --out "$C1" \
  --state-key qpos --batch-size 512 > "$L/cache1.log" 2>&1 || die "fs1 cache failed"; }
[ -f "$C5" ] || CUDA_VISIBLE_DEVICES=0 python3 "$TRM/subsample_cache.py" \
  --in "$C1" --out "$C5" --frameskip 5 > "$L/cache5.log" 2>&1 || die "fs5 failed"
[ -f "$W3" ] || { log "window3 value on fine-tuned latents (${SW_VSTEPS} steps)"; CUDA_VISIBLE_DEVICES=0 python3 \
  /workspace/train_window.py --cache "$C1" --lag 5 --frames 3 --expectile "$SW_EXV" \
  --n-step 50 --steps "$SW_VSTEPS" --seed 0 --out "$W3" > "$L/w3.log" 2>&1 || die "window value failed"; }
log "caches + window value ready"
grep -oE "\[tiled-goal diag\].*" "$L/w3.log" 2>/dev/null | tail -1

# ---------------------------------------------------------------- 4. LIP at the winning recipe
log "LIP x3 seeds at the SWEPT recipe: 1000 steps, amax ${SW_AMAX}, lr ${SW_LR}, expectile ${SW_EXV}, replay ${SW_RP}, expand ${SW_EW}"
for s in 0 1 2; do
  A=/workspace/actors/lip4_dyna_sw_s${s}.pt
  [ -f "$A" ] && continue
  CUDA_VISIBLE_DEVICES=$((s % 6)) timeout 21600 python3 "$PLAN/train_lip_ac.py" \
    --cache "$C5" --cache-td "$C1" --h5 "$SLIM" --wm "$FT" \
    --pad-context --init-value "$W3" \
    --arch v4 --amax "$SW_AMAX" --iters 8 --horizon 5 --max-delta 12 \
    --steps 1000 --batch 128 --n-step 50 \
    --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr "$SW_LR" --actor-lr-final "$SW_LRF" \
    --replay-prob "$SW_RP" --expand-weight "$SW_EW" \
    --lambda-schedule uniform --mean-weight 0.1 --seed "$s" \
    --out "$A" --out-value "/workspace/metrics/lip4_dyna_sw_s${s}_value.pt" \
    > "$L/lip_s${s}.log" 2>&1 && log "  LIP s${s} DONE" || log "  LIP s${s} FAILED" &
done
wait

# ---------------------------------------------------------------- 5. cards
ev(){ local gpu=$1 nm=$2 seed=$3; shift 3
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$FT" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver.batch_size=10 "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
meanof(){ local pre=$1 key=$2; local t=0 n=0 e
  for s in 42 43 44 45 46 47; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

log "cards on the fine-tuned WM (eval pool 8000:10000, never trained on)"
( for s in 42 43 44 45 46 47; do ev 0 "dy_l2_s${s}" $s solver=cem solver.n_steps=10 "+metric=/workspace/metrics/l2window3.pt"; done
  log "CARD Latent+CEM-window (Dyna WM): HELD $(meanof dy_l2_s held) | @0.1 $(meanof dy_l2_s held10)  [base 44.7 / 84.3]" ) &
( for s in 42 43 44 45 46 47; do ev 1 "dy_td_s${s}" $s solver=cem solver.n_steps=10 "+metric=$W3"; done
  log "CARD TD+CEM-window (Dyna WM): HELD $(meanof dy_td_s held) | @0.1 $(meanof dy_td_s held10)  [base 17.7 / 45.3]" ) &
( for sd in 0 1 2; do
    A=/workspace/actors/lip4_dyna_sw_s${sd}.pt; [ -f "$A" ] || continue
    for s in 42 43 44 45 46 47; do ev $((sd + 2)) "dy_lip${sd}_s${s}" $s solver=lip solver.actor_path="$A" solver.rollout_compat=false; done
    log "CARD LIP-window s${sd} (Dyna WM): HELD $(meanof dy_lip${sd}_s held) | @0.1 $(meanof dy_lip${sd}_s held10)"
  done ) &
wait
t=0; k=0
for sd in 0 1 2; do
  v=$(meanof "dy_lip${sd}_s" held); [ "$v" = "0.0" ] && continue
  t=$(awk "BEGIN{print $t+$v}"); k=$((k+1))
done
[ "$k" -gt 0 ] && log "POOLED LIP-window (Dyna WM): HELD $(awk "BEGIN{printf \"%.1f\", $t/$k}") over ${k} seeds  [base WM: 48.9]"
log "DYNA_DONE -- compare against the sweep's POOLED lejepa card, not the old 48.9"
