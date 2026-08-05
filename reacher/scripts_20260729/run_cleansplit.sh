#!/bin/bash
# How much of 99.0 / 93.1 is memorisation?
#
# THE LEAK. The window value and the LIP actor both train on the FULL latent
# cache -- episodes 0..9999, verified, with no episode filtering in either
# trainer (train_window.py: 0 filters; the only "8000" in train_lip_ac.py is the
# --steps default). We then evaluate on 8000:10000. So 20.0% of every training
# batch is drawn from the evaluation episodes:
#     canon_lejepa_fs1  2010000 rows, 402000 in the eval pool
#     canon_lejepa_fs5   410000 rows,  82000 in the eval pool
# The headline numbers are therefore UPPER BOUNDS, not held-out results.
#
# This rebuilds the value and the actor on episodes 0..7999 only and re-cards on
# 8000:10000, which is then genuinely held out for both. The gap between the two
# cards is the size of the leak.
#
# WHAT THIS DOES NOT FIX: the lejepa/pldm world models were themselves
# pretrained on all 10k episodes. That is not repairable without retraining the
# world model, so the result here is "value + actor clean, world model still
# leaked" -- a strict improvement in validity, and it isolates how much of the
# gain lived in the two components we control.
#
# WHY THE COMPARISON SURVIVES EITHER WAY. Latent+CEM is measured on the same
# leaked world model, and its L2 window cost is parameter-free -- it has no
# capacity to memorise anything. So "LIP beats Latent+CEM" is not a leak
# artifact. What the leak threatens is the ABSOLUTE number, not the margin.
#
# Filtering the cache alone is sound: the actor takes its episode list from the
# cache (c.episodes()) and reaches actions through the h5's ep_offset index, so
# rows belonging to excluded episodes are unreachable. Residual, and stated
# rather than engineered around: the action z-score stats are still computed
# over all 2.01M rows. That is a mean/std of a random-play action distribution
# over 10000 vs 8000 episodes -- far too coarse to carry per-episode
# information, but it is not literally zero.
#
# Runs on GPU 0 with capped concurrency: phase 2 owns GPUs 1-5, and four CEM
# arms in this campaign have core-dumped under contention.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
SUM=/workspace/results/summary_cleansplit.csv; touch "$SUM"
L=/workspace/logs/cleansplit; mkdir -p "$L"
REP="42 43 44 45 46 47"
G=0.98
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][clean] $*"; }
die(){ log "FATAL: $*"; exit 1; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
# the current per-base winners (gamma 0.98 grid, carded 99.0 / 93.1)
lr_of(){   [ "$1" = "lejepa" ] && echo 1e-4  || echo 3e-4; }
lrf_of(){  [ "$1" = "lejepa" ] && echo 1e-5  || echo 3e-5; }
mw_of(){   [ "$1" = "lejepa" ] && echo 0.3   || echo 0.1; }
amax_of(){ [ "$1" = "lejepa" ] && echo 2.2   || echo 1.8; }
bar_of(){  [ "$1" = "lejepa" ] && echo 84.3  || echo 78.3; }
leak_of(){ [ "$1" = "lejepa" ] && echo 99.0  || echo 93.1; }

# ---------------------------------------------------------------- 1. filtered caches
log "building train-only caches (episodes 0..7999)"
python3 - <<'PYEOF' || die "cache filtering failed"
import torch, numpy as np
for B in ["lejepa", "pldm"]:
    for fs in ["fs1", "fs5"]:
        src = f"/workspace/caches/canon_{B}_{fs}.pt"
        dst = f"/workspace/caches/tr_{B}_{fs}.pt"
        import os
        if os.path.exists(dst):
            print(f"{dst} exists", flush=True); continue
        c = torch.load(src, map_location="cpu", weights_only=False)
        e = np.asarray(c["episode_idx"])
        m = e < 8000
        out = {}
        for k, v in c.items():
            if hasattr(v, "shape") and len(v) == len(e):
                out[k] = v[torch.from_numpy(m)] if torch.is_tensor(v) else v[m]
            else:
                out[k] = v
        ne = np.asarray(out["episode_idx"])
        assert ne.max() < 8000, "eval pool survived the filter"
        torch.save(out, dst)
        print(f"{B} {fs}: {len(e)} -> {len(ne)} rows, ep max {ne.max()}", flush=True)
print("FILTER_DONE", flush=True)
PYEOF
for B in lejepa pldm; do for fs in fs1 fs5; do
  [ -f "/workspace/caches/tr_${B}_${fs}.pt" ] || die "missing tr_${B}_${fs}.pt"
done; done

# ---------------------------------------------------------------- 2. values on train-only data
log "window values at gamma ${G}, expectile 0.05, on episodes 0..7999"
for B in lejepa pldm; do
  W=/workspace/metrics/window3_${B}_e005_g098_tr.pt
  [ -f "$W" ] && continue
  CUDA_VISIBLE_DEVICES=0 python3 /workspace/train_window.py \
    --cache /workspace/caches/tr_${B}_fs1.pt --lag 5 --frames 3 \
    --expectile 0.05 --n-step 50 --gamma "$G" --steps 6000 --seed 0 --out "$W" \
    > "$L/w3_${B}.log" 2>&1 || die "value ${B} failed"
  log "  ${B}: loss $(grep -oE 'final loss=[0-9.]+' "$L/w3_${B}.log" | tail -1 | cut -d= -f2)"
done

# ---------------------------------------------------------------- 3. actors on train-only data
log "LIP actors at each base's carded winner, trained on episodes 0..7999"
for B in lejepa pldm; do for sd in 0 1 2; do
  A=/workspace/actors/lip4_clean_${B}_s${sd}.pt
  [ -f "$A" ] && continue
  CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$PLAN/train_lip_ac.py" \
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
    --out "$A" --out-value "/workspace/metrics/lip4_clean_${B}_s${sd}_value.pt" \
    > "$L/train_${B}_s${sd}.log" 2>&1 || log "TRAIN FAILED ${B} s${sd}"
done; done
log "actors: $(ls /workspace/actors/lip4_clean_*_s?.pt 2>/dev/null | wc -l)/6"

# ---------------------------------------------------------------- 4. card on held-out episodes
ev(){ local B=$1 nm=$2 seed=$3 actor=$4
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
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

log "carding on 8000:10000 -- held out from the value and the actor"
for B in lejepa pldm; do
  ta=0; tb=0; k=0; nsc=0
  for sd in 0 1 2; do
    A=/workspace/actors/lip4_clean_${B}_s${sd}.pt; [ -f "$A" ] || continue
    for s in $REP; do ev "$B" "cl_${B}_s${sd}_e${s}" "$s" "$A"; done
    a=$(meanof "cl_${B}_s${sd}_e" held10); c=$(meanof "cl_${B}_s${sd}_e" held)
    n=$(grep -c "^cl_${B}_s${sd}_e.*held=[0-9]" "$SUM")
    log "  CARD ${B} s${sd}: @0.1 ${a} | @0.05 ${c}  (${n}/6 scored)"
    ta=$(awk "BEGIN{print $ta+$a}"); tb=$(awk "BEGIN{print $tb+$c}"); k=$((k+1)); nsc=$((nsc+n))
  done
  [ "$k" -eq 0 ] && { log "  ${B}: nothing carded"; continue; }
  m10=$(awk "BEGIN{printf \"%.1f\", $ta/$k}"); m05=$(awk "BEGIN{printf \"%.1f\", $tb/$k}")
  log "POOLED ${B} CLEAN: @0.1 ${m10} | @0.05 ${m05}   (${nsc}/18 evals scored)"
  log "  leaked card was @0.1 $(leak_of $B) -> leak = $(awk "BEGIN{printf \"%+.1f\", $(leak_of $B)-$m10}") points"
  log "  bar $(bar_of $B) -> clean margin $(awk "BEGIN{printf \"%+.1f\", $m10-$(bar_of $B)}")"
done
log "CLEANSPLIT_DONE"
