#!/bin/bash
# Value+CEM looks understated at 51.3/51.3. Three concrete reasons it might be.
#
# The value fed to CEM is the STANDALONE window value: a pure TD fit on cached
# transitions, expectile 0.05, gamma 0.98, 6000 steps. Three things about that
# are chosen for LIP, not for CEM:
#
#  B. EXPECTILE. 0.05 was selected because it won the LIP sweep. For CEM the
#     earlier evidence points the other way: at gamma 1.0, lejepa Value+CEM
#     scored 45.3 with the expectile-0.1 value against 38.3 with expectile 0.05.
#     Nobody has ever tried expectile 0.1 AT gamma 0.98 -- the two knobs were
#     only ever moved one at a time, and the current cell is the intersection
#     that happens to favour LIP.
#
#  C. THE WRONG VALUE ENTIRELY. LIP does not deploy the standalone value -- it
#     CO-TRAINS its critic during actor training (expectile annealed 0.1 -> 0.03,
#     replay-prob 0.5, EMA teacher) and writes the result to --out-value. That
#     critic has seen the states the actor actually visits, which is exactly the
#     distribution CEM will query. Feeding CEM the initial value instead of the
#     co-trained one may simply be scoring the wrong object, and would understate
#     what "the learned value" is worth.
#
#  D. REFINEMENT BUDGET. Our protocol pins solver.n_steps=10, overriding
#     cem.yaml's default of 30. Latent+CEM is fine with 10 because plain L2 is
#     smooth; a learned quasimetric may need more refinement to exploit. Same
#     300 samples, 3x the iterations.
#
# Control (A) is the current number and is not re-run: 51.3 / 51.3 @0.1.
#
# If (C) wins, the honest table row becomes "Value+CEM using LIP's co-trained
# critic", and the gap between LIP and its own critic shrinks -- which changes
# the story about where LIP's advantage comes from.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_vcem.csv; touch "$SUM"
L=/workspace/logs/vcem; mkdir -p "$L"
REP="42 43 44 45 46 47"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][vcem] $*"; }

# ---------------------------------------------------------------- expectile-0.1 value at gamma 0.98
log "building expectile-0.1 values at gamma 0.98 (the untried intersection)"
i=0
for B in lejepa pldm; do
  W=/workspace/metrics/window3_${B}_e01_g098.pt
  [ -f "$W" ] && { i=$((i+1)); continue; }
  CUDA_VISIBLE_DEVICES=$((i % 6)) python3 /workspace/train_window.py \
    --cache /workspace/caches/canon_${B}_fs1.pt --lag 5 --frames 3 \
    --expectile 0.1 --n-step 50 --gamma 0.98 --steps 6000 --seed 0 --out "$W" \
    > "$L/w3_${B}_e01.log" 2>&1 &
  i=$((i + 1))
done
wait

ev(){ local gpu=$1 B=$2 nm=$3 seed=$4 metric=$5 nsteps=$6
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="/workspace/swm_home/checkpoints/${B}_reacher" \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=cem solver.n_steps=$nsteps solver.batch_size=10 \
    "+metric=${metric}" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
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

log "3 arms x 2 bases x 6 seeds, 4-wide"
i=0
for B in lejepa pldm; do
  CO=/workspace/metrics/lip4_leak6_${B}_s0_value.pt
  ( for s in $REP; do ev $((i % 6)) "$B" "vc_e01_${B}_s${s}" $s \
      "/workspace/metrics/window3_${B}_e01_g098.pt" 10; done ) &
  i=$((i + 1))
  if [ -f "$CO" ]; then
    ( for s in $REP; do ev $((i % 6)) "$B" "vc_co_${B}_s${s}" $s "$CO" 10; done ) &
    i=$((i + 1))
  else
    log "  NOTE ${B}: no co-trained value at ${CO}"
  fi
  ( for s in $REP; do ev $((i % 6)) "$B" "vc_n30_${B}_s${s}" $s \
      "/workspace/metrics/window3_${B}_e005_g098.pt" 30; done ) &
  i=$((i + 1))
  [ $((i % 4)) -eq 0 ] && wait
done
wait

log "==================== Value+CEM retry (@0.1, reporting seeds) ===================="
for B in lejepa pldm; do
  log "--- ${B} ---"
  log "  A expectile 0.05, n_steps 10   @0.1 51.3   [current table row]"
  log "  B expectile 0.10, n_steps 10   @0.1 $(meanof "vc_e01_${B}_s" held10)   @0.05 $(meanof "vc_e01_${B}_s" held)   ($(grep -c "^vc_e01_${B}_s.*held=[0-9]" "$SUM")/6)"
  log "  C LIP co-trained critic        @0.1 $(meanof "vc_co_${B}_s" held10)   @0.05 $(meanof "vc_co_${B}_s" held)   ($(grep -c "^vc_co_${B}_s.*held=[0-9]" "$SUM")/6)"
  log "  D expectile 0.05, n_steps 30   @0.1 $(meanof "vc_n30_${B}_s" held10)   @0.05 $(meanof "vc_n30_${B}_s" held)   ($(grep -c "^vc_n30_${B}_s.*held=[0-9]" "$SUM")/6)"
  log "  reference: Latent+CEM (parameter-free L2) $([ "$B" = lejepa ] && echo 84.3 || echo 78.3)"
done
log "VCEM_DONE"
