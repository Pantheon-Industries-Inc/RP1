#!/bin/bash
# Does discounting help? A clean A/B at the incumbent recipe.
#
# Nothing in this stack has ever discounted: train_window.py and train_lip_ac.py
# both default gamma=1.0 and neither was overridden, so every campaign number is
# an UNDISCOUNTED steps-to-go. The decision was to keep it that way, on the
# reasoning that
#   - the head is quasimetric, and (1-g^n)/(1-g) is a saturating transform of
#     step count that breaks the metric/triangle-inequality interpretation;
#   - the value saturates at 1/(1-g) (= 50 at g=0.98), collapsing the distinction
#     between 50 and 200 steps out, while the planner needs resolution in the
#     near field it actually operates over (h25);
#   - the usual argument FOR discounting -- unbounded targets -- does not apply,
#     since n_eff is capped by the 50-step backup and by episode length.
# This measures that reasoning instead of trusting it.
#
# DESIGN: vary ONLY gamma. Recipe pinned at the incumbent (expectile 0.05,
# replay 0.5, expand 0, n-step 50, lr 3e-4, steps 1000, mw 0.1, batch 128,
# amax 2.2 lejepa / 1.8 pldm) so the comparison is like-for-like against the
# already-carded gamma=1.0 numbers:
#     lejepa  @0.1 89.0   @0.05 49.8      (bar 84.3)
#     pldm    @0.1 90.4   @0.05 48.9      (bar 78.3)
#
# Two gammas, to see a TREND rather than a single point: 0.98 (the value the
# user named) and 0.995 (a third of the way to undiscounted in 1/(1-g) terms:
# saturation moves 50 -> 200). If discounting hurts, the ordering should be
# monotone 0.98 < 0.995 < 1.0. If 0.98 wins, the intuition above is wrong.
#
# The window value is retrained per gamma: initialising a discounted critic from
# an undiscounted value would mix two objectives on different scales.
#
# No selection happens here -- three fixed configs are measured -- so all cards
# go straight to the reporting seeds. Nothing is being chosen on them.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
SUM=/workspace/results/summary_gamma.csv; touch "$SUM"
L=/workspace/logs/gamma; mkdir -p "$L"
REP="42 43 44 45 46 47"
GAMMAS="0.98 0.995"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][gamma] $*"; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
amax_of(){ [ "$1" = "lejepa" ] && echo 2.2 || echo 1.8; }
bar_of(){ [ "$1" = "lejepa" ] && echo 84.3 || echo 78.3; }
gtag(){ echo "g${1//./}"; }        # 0.98 -> g098, 0.995 -> g0995

# ---------------------------------------------------------------- values per gamma
log "stage0: window values per gamma (expectile 0.05, n-step 50, 6000 steps)"
i=0
for g in $GAMMAS; do for B in lejepa pldm; do
  W=/workspace/metrics/window3_${B}_e005_$(gtag $g).pt
  [ -f "$W" ] && { i=$((i+1)); continue; }
  CUDA_VISIBLE_DEVICES=$(( i % 6 )) python3 /workspace/train_window.py \
    --cache /workspace/caches/canon_${B}_fs1.pt --lag 5 --frames 3 \
    --expectile 0.05 --n-step 50 --gamma "$g" --steps 6000 --seed 0 --out "$W" \
    > "$L/w3_${B}_$(gtag $g).log" 2>&1 &
  i=$((i + 1))
done; done
wait
for g in $GAMMAS; do for B in lejepa pldm; do
  f="$L/w3_${B}_$(gtag $g).log"
  log "  ${B} gamma ${g}: loss $(grep -oE 'final loss=[0-9.]+' "$f" 2>/dev/null | tail -1 | cut -d= -f2)  (gamma 1.0: $([ "$B" = lejepa ] && echo 1.5939 || echo 1.5751))"
  grep -oE "\[tiled-goal diag\].*" "$f" 2>/dev/null | tail -1
done; done

# ---------------------------------------------------------------- LIP per gamma
log "stage1: 12 trainings (2 gammas x 2 bases x 3 seeds) at the incumbent recipe"
i=0
for g in $GAMMAS; do for B in lejepa pldm; do for sd in 0 1 2; do
  A=/workspace/actors/lip4_gam_${B}_$(gtag $g)_s${sd}.pt
  [ -f "$A" ] && { i=$((i+1)); continue; }
  CUDA_VISIBLE_DEVICES=$(( i % 6 )) timeout 28800 python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/canon_${B}_fs5.pt \
    --cache-td /workspace/caches/canon_${B}_fs1.pt \
    --h5 "$SLIM" --wm "$(wm_of $B)" --pad-context \
    --init-value /workspace/metrics/window3_${B}_e005_$(gtag $g).pt \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --gamma "$g" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight 0.1 --amax "$(amax_of $B)" \
    --replay-prob 0.5 --expand-weight 0 --steps 1000 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --seed "$sd" \
    --out "$A" --out-value "/workspace/metrics/lip4_gam_${B}_$(gtag $g)_s${sd}_value.pt" \
    > "$L/train_${B}_$(gtag $g)_s${sd}.log" 2>&1 || log "TRAIN FAILED ${B} ${g} s${sd}" &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done; done
wait
log "trainings drained ($(ls /workspace/actors/lip4_gam_*.pt 2>/dev/null | grep -c "_s[0-9]\.pt$")/12)"

# ---------------------------------------------------------------- cards
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
  [ -z "$h" ] && log "  !! ${nm} no score: $(tail -1 "$L/${nm}.log" | cut -c1-64)"
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
meanof(){ local pre=$1 key=$2; local t=0 n=0 e
  for s in $REP; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"; }

log "stage2: cards on reporting seeds {${REP}}"
i=0
for g in $GAMMAS; do for B in lejepa pldm; do
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_gam_${B}_$(gtag $g)_s${sd}.pt; [ -f "$A" ] || continue
      for s in $REP; do ev $(( i % 6 )) "$B" "gam_${B}_$(gtag $g)_s${sd}_e${s}" "$s" "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
done; done
wait

log "================== GAMMA A/B (reporting seeds) =================="
log "base     gamma     @0.1    @0.05    bar     margin@0.1"
for B in lejepa pldm; do
  inc10=$([ "$B" = lejepa ] && echo 89.0 || echo 90.4)
  inc05=$([ "$B" = lejepa ] && echo 49.8 || echo 48.9)
  for g in $GAMMAS; do
    ta=0; tb=0; k=0
    for sd in 0 1 2; do
      v=$(meanof "gam_${B}_$(gtag $g)_s${sd}_e" held10)
      w=$(meanof "gam_${B}_$(gtag $g)_s${sd}_e" held)
      [ "$v" = "0.0" ] && continue
      ta=$(awk "BEGIN{print $ta+$v}"); tb=$(awk "BEGIN{print $tb+$w}"); k=$((k+1))
    done
    [ "$k" -eq 0 ] && { log "  ${B} ${g}: no seeds scored"; continue; }
    m10=$(awk "BEGIN{printf \"%.1f\", $ta/$k}"); m05=$(awk "BEGIN{printf \"%.1f\", $tb/$k}")
    log "  $(printf '%-8s %-8s %6s %8s %7s %+11.1f' "$B" "$g" "$m10" "$m05" "$(bar_of $B)" "$(awk "BEGIN{print $m10-$(bar_of $B)}")")  (${k} seeds)"
  done
  log "  $(printf '%-8s %-8s %6s %8s %7s' "$B" "1.0" "$inc10" "$inc05" "$(bar_of $B)")  <- incumbent, undiscounted"
done
log "GAMMA_AB_DONE -- expected if the metric argument holds: 0.98 < 0.995 < 1.0"
