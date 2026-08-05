#!/bin/bash
# Is the 3-frame window value earning its keep? 1-frame vs 3-frame, current recipe.
#
# The window has been the default since early in the campaign, but it was chosen
# under a recipe that has since changed in every component: gamma 1.0 -> 0.98,
# expectile 0.1 -> 0.05, replay 0 -> 0.5, mean-weight 0.1 -> per-base, actor-lr
# 1e-3 -> per-base. It has never been re-tested against a plain 1-frame value on
# the recipe actually being reported.
#
# THREE REASONS TO DOUBT IT
#
# 1. A measured train/deploy mismatch. At TRAIN time the goal window is real --
#    critic_step uses _wrow(g_idx), three genuine consecutive goal frames. At
#    DEPLOY only one goal frame exists, so _wpair TILES it three times.
#    train_window.py prints exactly this gap: d(win -> same-location tiled)
#    = 7.02 against a random-pair baseline of 82.19. Small relative to random,
#    but not zero, and it is a bias the 1-frame value does not have at all.
#
# 2. Complexity cost. The window is why vframes plumbing exists across the
#    trainer, seven deploy call sites, the expand path, and the PWM port. It is
#    also what breaks DINO-WM: its metric hook wants 3 frames but that base
#    supplies 2 (index -3 out of bounds for dimension 1 with size 2). A 1-frame
#    value would unblock that base outright.
#
# 3. It may simply not be needed at gamma 0.98. Discounting sharpened the value
#    substantially on its own (TD+CEM 38.3 -> 51.3 lejepa, 33.7 -> 51.3 pldm),
#    and part of what the window was compensating for may have been the
#    undiscounted value's poor conditioning.
#
# WHAT IS COMPARED, everything else held at the reported recipe:
#     LIP        1-frame value   vs   3-frame window value   (3 seeds, 2 bases)
#     Latent+CEM 1-frame L2      vs   3-frame window L2      (the bar itself)
# The bar is included because if the window helps CEM but not LIP (or vice
# versa) that is a different conclusion than if it helps both or neither.
#
# Incumbents to beat, @0.1, leaked, reporting seeds:
#     LIP-window        lejepa 98.2   pldm 94.2
#     Latent+CEM-window lejepa 84.3   pldm 78.3
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
SUM=/workspace/results/summary_winabl.csv; touch "$SUM"
L=/workspace/logs/winabl; mkdir -p "$L"
REP="42 43 44 45 46 47"; G=0.98
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][winabl] $*"; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
lr_of(){   [ "$1" = "lejepa" ] && echo 1e-4 || echo 3e-4; }
lrf_of(){  [ "$1" = "lejepa" ] && echo 1e-5 || echo 3e-5; }
mw_of(){   [ "$1" = "lejepa" ] && echo 0.3  || echo 0.5; }
amax_of(){ [ "$1" = "lejepa" ] && echo 2.2  || echo 1.8; }
inc_of(){  [ "$1" = "lejepa" ] && echo 98.2 || echo 94.2; }
bar_of(){  [ "$1" = "lejepa" ] && echo 84.3 || echo 78.3; }

# ---------------------------------------------------------------- 1-frame value + L2 stub
log "stage0: 1-frame values (gamma ${G}, expectile 0.05, n-step 50, 6000 steps)"
i=0
for B in lejepa pldm; do
  W=/workspace/metrics/window1_${B}_e005_g098.pt
  [ -f "$W" ] && { i=$((i+1)); continue; }
  CUDA_VISIBLE_DEVICES=$((i % 6)) python3 /workspace/train_window.py \
    --cache /workspace/caches/canon_${B}_fs1.pt --lag 5 --frames 1 \
    --expectile 0.05 --n-step 50 --gamma "$G" --steps 6000 --seed 0 --out "$W" \
    > "$L/w1_${B}.log" 2>&1 &
  i=$((i + 1))
done
wait
for B in lejepa pldm; do
  f="$L/w1_${B}.log"
  log "  ${B} 1-frame: loss $(grep -oE 'final loss=[0-9.]+' "$f" 2>/dev/null | tail -1 | cut -d= -f2) (3-frame was ~0.55)"
done
L2ONE=/workspace/metrics/l2window1.pt
[ -f "$L2ONE" ] || python3 -c "
import torch, collections
torch.save({'learner':'l2','latent_dim':192,
            'arch':{'window_frames':1,'window_lag':5,'unlearned':True},
            'state_dict':collections.OrderedDict()}, '$L2ONE')
print('wrote 1-frame L2 stub')"

# ---------------------------------------------------------------- LIP on the 1-frame value
log "stage1: 6 LIP trainings (2 bases x 3 seeds) on the 1-frame value"
i=0
for B in lejepa pldm; do for sd in 0 1 2; do
  A=/workspace/actors/lip4_w1_${B}_s${sd}.pt
  [ -f "$A" ] && { i=$((i+1)); continue; }
  CUDA_VISIBLE_DEVICES=$((i % 6)) timeout 28800 python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/canon_${B}_fs5.pt \
    --cache-td /workspace/caches/canon_${B}_fs1.pt \
    --h5 "$SLIM" --wm "$(wm_of $B)" --pad-context \
    --init-value /workspace/metrics/window1_${B}_e005_g098.pt \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --gamma "$G" --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight "$(mw_of $B)" --amax "$(amax_of $B)" \
    --replay-prob 0.5 --expand-weight 0 --steps 1000 \
    --actor-lr "$(lr_of $B)" --actor-lr-final "$(lrf_of $B)" --seed "$sd" \
    --out "$A" --out-value "/workspace/metrics/lip4_w1_${B}_s${sd}_value.pt" \
    > "$L/train_${B}_s${sd}.log" 2>&1 || log "TRAIN FAILED ${B} s${sd}" &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done
wait
log "actors: $(ls /workspace/actors/lip4_w1_*_s?.pt 2>/dev/null | wc -l)/6"
for B in lejepa pldm; do
  grep -m1 -oE "\[vframes\].*" "$L/train_${B}_s0.log" 2>/dev/null \
    && log "  NOTE ${B}: printed a [vframes] line -- expected NONE for a 1-frame value" \
    || log "  ${B}: no [vframes] line, as expected for 1-frame"
done

# ---------------------------------------------------------------- eval
ev(){ local gpu=$1 B=$2 nm=$3 seed=$4; shift 4
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$(wm_of $B)" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
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

log "stage2: cards -- LIP 1-frame, and the 1-frame Latent+CEM bar"
i=0
for B in lejepa pldm; do
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_w1_${B}_s${sd}.pt; [ -f "$A" ] || continue
      for s in $REP; do ev $((i % 5)) "$B" "w1_${B}_s${sd}_e${s}" "$s" \
        solver=lip solver.actor_path="$A" solver.rollout_compat=false; done
    done ) &
  i=$((i + 1))
  ( for s in $REP; do ev $((i % 5)) "$B" "w1cem_${B}_s${s}" "$s" \
      solver=cem solver.n_steps=10 "+metric=$L2ONE"; done ) &
  i=$((i + 1))
done
wait

log "==================== WINDOW vs 1-FRAME (@0.1, reporting seeds) ===================="
for B in lejepa pldm; do
  ta=0; k=0; nsc=0
  for sd in 0 1 2; do
    a=$(meanof "w1_${B}_s${sd}_e" held10)
    n=$(grep -c "^w1_${B}_s${sd}_e.*held=[0-9]" "$SUM"); nsc=$((nsc + n))
    [ "$n" -eq 0 ] && continue
    ta=$(awk "BEGIN{print $ta+$a}"); k=$((k+1))
  done
  M=$([ "$k" -gt 0 ] && awk "BEGIN{printf \"%.1f\", $ta/$k}" || echo "n/a")
  log "--- ${B} ---"
  log "  LIP        1-frame ${M}   |  3-frame $(inc_of $B)   (${nsc}/18 evals)"
  log "  Latent+CEM 1-frame $(meanof "w1cem_${B}_s" held10)   |  3-frame $(bar_of $B)   ($(grep -c "^w1cem_${B}_s.*held=[0-9]" "$SUM")/6)"
done
log "WINABL_DONE -- if 1-frame holds up, the window's plumbing and its tiled-goal"
log "bias can both go, and DINO-WM stops being blocked by the 3-frame hook."
