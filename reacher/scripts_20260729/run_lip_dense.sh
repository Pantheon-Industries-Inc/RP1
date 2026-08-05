#!/bin/bash
# LIP with a dense per-timestep path term. Everything else at the reported recipe.
#
# PWM's ablation made the objective the single largest lever measured on this
# task, at matched budget and matched critic:
#     lejepa  dense 81.2  terminal 64.1   (and terminal collapses to 0.3 at 8k)
#     pldm    dense 91.2  terminal 20.4   (terminal collapses to 1.8 at 8k)
# LIP is terminal with respect to the rollout at every refinement, and LIP also
# degrades with more steps (1000 beat 3000 in all 12 pairings). Same signature,
# milder -- plausibly because refinement plus the mean-over-iterations term act
# as brakes. So the question is whether LIP gains from the term outright, or
# whether its existing machinery already substitutes for it.
#
# NOT the same as --mean-weight: that averages TERMINAL cost over refinement
# iterations k; this averages cost over rollout timesteps t. Orthogonal axes.
#
# ARMS: dense-weight {0, 0.3, 1.0} x 2 bases x 3 seeds. The 0 arm is retrained
# rather than inherited so the comparison is exactly matched at 3 seeds -- the
# reported incumbents (lejepa 98.2, pldm 94.2) are 6-seed numbers and would be a
# different n. Both are printed.
#
# Everything else pinned to the reported recipe: gamma 0.98, expectile 0.05,
# replay 0.5, expand 0, n-step 50, 1000 steps, batch 128, 3-frame window value
# (the window ablation showed 1-frame costs 8-22 points, so it stays), per-base
# amax / actor-lr / mean-weight.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
SUM=/workspace/results/summary_lipdense.csv; touch "$SUM"
L=/workspace/logs/lipdense; mkdir -p "$L"
REP="42 43 44 45 46 47"; G=0.98
DW="0 0.3 1.0"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][dense] $*"; }
die(){ log "FATAL: $*"; exit 1; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
lr_of(){   [ "$1" = "lejepa" ] && echo 1e-4 || echo 3e-4; }
lrf_of(){  [ "$1" = "lejepa" ] && echo 1e-5 || echo 3e-5; }
mw_of(){   [ "$1" = "lejepa" ] && echo 0.3  || echo 0.5; }
amax_of(){ [ "$1" = "lejepa" ] && echo 2.2  || echo 1.8; }
inc_of(){  [ "$1" = "lejepa" ] && echo 98.2 || echo 94.2; }
bar_of(){  [ "$1" = "lejepa" ] && echo 84.3 || echo 78.3; }
tag(){ echo "${1}_d${2//./}"; }

grep -q "dense_weight" "$PLAN/train_lip_ac.py" || die "train_lip_ac.py is not dense-patched"

log "18 trainings: 2 bases x dense-weight{${DW}} x 3 seeds"
i=0
for B in lejepa pldm; do for dw in $DW; do for sd in 0 1 2; do
  t=$(tag "$B" "$dw"); A=/workspace/actors/lip4_dn_${t}_s${sd}.pt
  [ -f "$A" ] && { i=$((i+1)); continue; }
  CUDA_VISIBLE_DEVICES=$((i % 6)) timeout 28800 python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/canon_${B}_fs5.pt \
    --cache-td /workspace/caches/canon_${B}_fs1.pt \
    --h5 "$SLIM" --wm "$(wm_of $B)" --pad-context \
    --init-value /workspace/metrics/window3_${B}_e005_g098.pt \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --gamma "$G" --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight "$(mw_of $B)" --amax "$(amax_of $B)" \
    --replay-prob 0.5 --expand-weight 0 --steps 1000 --dense-weight "$dw" \
    --actor-lr "$(lr_of $B)" --actor-lr-final "$(lrf_of $B)" --seed "$sd" \
    --out "$A" --out-value "/workspace/metrics/lip4_dn_${t}_s${sd}_value.pt" \
    > "$L/train_${t}_s${sd}.log" 2>&1 || log "TRAIN FAILED ${t} s${sd}" &
  i=$((i + 1)); [ $((i % 12)) -eq 0 ] && wait
done; done; done
wait
log "actors: $(ls /workspace/actors/lip4_dn_*_s?.pt 2>/dev/null | wc -l)/18"

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

log "carding on reporting seeds"
i=0
for B in lejepa pldm; do for dw in $DW; do
  t=$(tag "$B" "$dw")
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_dn_${t}_s${sd}.pt; [ -f "$A" ] || continue
      for s in $REP; do ev $((i % 5)) "$B" "dn_${t}_s${sd}_e${s}" "$s" "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % 5)) -eq 0 ] && wait
done; done
wait

log "=============== LIP dense path term (@0.1, reporting seeds) ==============="
for B in lejepa pldm; do
  log "--- ${B} (bar $(bar_of $B); 6-seed terminal incumbent $(inc_of $B)) ---"
  for dw in $DW; do
    t=$(tag "$B" "$dw"); ta=0; tb=0; k=0; nsc=0
    for sd in 0 1 2; do
      a=$(meanof "dn_${t}_s${sd}_e" held10); c=$(meanof "dn_${t}_s${sd}_e" held)
      n=$(grep -c "^dn_${t}_s${sd}_e.*held=[0-9]" "$SUM"); nsc=$((nsc + n))
      [ "$n" -eq 0 ] && continue
      ta=$(awk "BEGIN{print $ta+$a}"); tb=$(awk "BEGIN{print $tb+$c}"); k=$((k+1))
    done
    [ "$k" -eq 0 ] && { log "  dense-weight ${dw}: NOTHING SCORED"; continue; }
    mark=""; [ "$dw" = "0" ] && mark="   <- control (terminal-only)"
    log "  $(printf 'dense-weight %-4s @0.1 %-6s @0.05 %-6s (%s/18)' "$dw" \
        "$(awk "BEGIN{printf \"%.1f\", $ta/$k}")" "$(awk "BEGIN{printf \"%.1f\", $tb/$k}")" "$nsc")${mark}"
  done
done
log "LIPDENSE_DONE"
