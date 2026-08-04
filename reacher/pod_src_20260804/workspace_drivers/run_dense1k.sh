#!/bin/bash
# Is `dense` alignment worth 6000 steps, or does it work at the REPORTED 1000?
#
# The rh=1 winner (dense) differs from the reported recipe in TWO ways: the
# added path term, and 6000 steps instead of 1000. The terminal control already
# shows the step count alone is harmful at rh=1 (lejepa 79.0 -> 51.0 @0.1 going
# 1000 -> 6000 under a terminal loss), so dense is winning despite the longer
# budget, not because of it. If dense also works at 1000 steps then the rh=1
# recipe is the reported recipe plus ONE loss term at an identical budget --
# which is a far cleaner thing to publish than a two-variable change.
#
# 2 trainings, seed 0, each base at its reported amax (lejepa 2.2, pldm 1.8 --
# which is also what the 6000-step screen picked), then carded at rh=1 on the
# selection seeds against the 6000-step numbers.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_rh1_screen.csv
L=/workspace/logs/dense1k; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][d1k] $*"; }
amax_of(){ [ "$1" = lejepa ] && echo 2.2 || echo 1.8; }
lr_of(){   [ "$1" = lejepa ] && echo 1e-4 || echo 3e-4; }
lrf_of(){  [ "$1" = lejepa ] && echo 1e-5 || echo 3e-5; }
mw_of(){   [ "$1" = lejepa ] && echo 0.3  || echo 0.5; }

log "training dense @1000 steps, both bases"
i=0
for B in lejepa pldm; do
  A=/workspace/actors/rh1k_${B}_dense.pt
  [ -f "$A" ] && { log "  $B exists"; i=$((i+1)); continue; }
  CUDA_VISIBLE_DEVICES=$i timeout 21600 python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/canon_${B}_fs5.pt \
    --cache-td /workspace/caches/canon_${B}_fs1.pt \
    --h5 /workspace/reacher_slim.h5 --wm /workspace/swm_home/checkpoints/${B}_reacher \
    --pad-context --init-value /workspace/metrics/window3_${B}_e005_g098.pt \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --gamma 0.98 --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight "$(mw_of $B)" --amax "$(amax_of $B)" \
    --replay-prob 0.5 --expand-weight 0 --steps 1000 \
    --align-mode dense --align-weight 0.3 \
    --actor-lr "$(lr_of $B)" --actor-lr-final "$(lrf_of $B)" --seed 0 \
    --out "$A" --out-value "/workspace/metrics/rh1k_${B}_dense_value.pt" \
    > "$L/${B}.log" 2>&1 && log "  ${B} DONE" || log "  ${B} FAILED" &
  i=$((i + 1))
done
wait
log "E_final: $(for f in $L/*.log; do printf '%s=%s ' "$(basename $f .log)" "$(grep -oE 'E_final [0-9.]+' $f | tail -1 | cut -d' ' -f2)"; done)"

# ------------------------------------------------------------------- card it
ev(){ local gpu=$1 B=$2 nm=$3 seed=$4 act=$5
  grep -q "^${nm}," "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 5400 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy=/workspace/swm_home/checkpoints/${B}_reacher \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    plan_config.receding_horizon=1 \
    solver=lip solver.actor_path="$act" solver.rollout_compat=false \
    output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && { log "  !! ${nm} no score"; h=FAIL; t10=FAIL; }
  echo "${nm},held=${h},held10=${t10},i3=0" >> "$SUM"
}
log "carding at rh=1, I3 off, selection seeds"
i=0
for B in lejepa pldm; do for s in 50 51; do
  ev $((i % 4)) "$B" "${B}_dense1k_off_e${s}" "$s" /workspace/actors/rh1k_${B}_dense.pt &
  i=$((i + 1))
done; done
wait

mean(){ grep "^${1}" "$SUM" | grep -oE "${2}=[0-9.]+" | cut -d= -f2 \
        | awk '{t+=$1;n++}END{printf "%5.1f(n=%d)", (n?t/n:0), n}'; }
log "===== dense @1000 vs @6000 at rh=1, selection seeds, I3 off ====="
for B in lejepa pldm; do
  AM=$([ "$B" = lejepa ] && echo 22 || echo 18)
  log "  $B  1000 steps: @0.1 $(mean ${B}_dense1k_off_ held10)  @0.05 $(mean ${B}_dense1k_off_ held)"
  log "  $B  6000 steps: @0.1 $(mean ${B}_dense_a${AM}_off_ held10)  @0.05 $(mean ${B}_dense_a${AM}_off_ held)"
done
log "DENSE1K_DONE"
