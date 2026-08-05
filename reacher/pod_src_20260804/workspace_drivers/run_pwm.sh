#!/bin/bash
# PWM joins the baseline table, on equal terms (leaked configuration).
#
# PWM is the reactive-policy arm: one encoder pass + one MLP pass per action
# block, no sampling and no refinement -- so it sits at the opposite end of the
# compute axis from CEM (3000 rollouts) with LIP (~16) in between. It was
# blocked until now because train_pwm_ac.py hard-asserted
# latent_dim == cache_dim and so could not consume the 3-frame window value
# every other arm uses (patch_pwm_vframes.py fixes that).
#
# EQUAL TERMS means it gets exactly what LIP gets:
#   - the same FULL caches (episodes 0..9999) and the same gamma-0.98 window
#     value the reported LIP numbers use. The campaign reports the leaked
#     configuration (user, 2026-08-02); the clean split was run separately and
#     measured the leak at +0.2 / +3.6 points, so the two differ little -- but
#     PWM must match whatever LIP is compared against, or the arms are not
#     comparable at all.
#   - the same shared knobs: gamma 0.98, n-step 50, max-delta 12, horizon 5,
#     batch 128, expectile 0.1 -> 0.03
#   - the same per-base amax (2.2 lejepa / 1.8 pldm)
# PWM-specific settings stay at the trainer's own defaults (width 512, layers 3,
# ema-tau 0.005, dense objective, 8000 steps, actor-lr 5e-4) -- tuning those
# would be tuning PWM, and the point here is a baseline, not a contest between
# my tuning effort on each arm.
#
# PWM needs no --pad-context: it already builds z_hist as z0 repeated hs times
# with zero action history, which IS the 1-frame deployment interface.
#
# Gated behind run_final6.sh so the two do not contend -- four CEM/eval arms in
# this campaign have core-dumped when too many shared the GPUs.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
SUM=/workspace/results/summary_pwm.csv; touch "$SUM"
L=/workspace/logs/pwm; mkdir -p "$L"
REP="42 43 44 45 46 47"
G=0.98
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][pwm] $*"; }
die(){ log "FATAL: $*"; exit 1; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
amax_of(){ [ "$1" = "lejepa" ] && echo 2.2 || echo 1.8; }
bar_of(){ [ "$1" = "lejepa" ] && echo 84.3 || echo 78.3; }

grep -q "_wpair" "$PLAN/train_pwm_ac.py" || die "train_pwm_ac.py is not vframes-patched"

if ! grep -q "FINAL6_DONE" /workspace/logs/final6.log 2>/dev/null; then
  log "waiting for run_final6.sh (ceiling 6 h)"
  w=0
  while ! grep -q "FINAL6_DONE" /workspace/logs/final6.log 2>/dev/null; do
    sleep 120; w=$((w + 120))
    [ $((w % 1800)) -eq 0 ] && log "  still waiting, ${w}s"
    [ "$w" -ge 21600 ] && die "final6 never finished"
  done
fi

for B in lejepa pldm; do
  for f in caches/canon_${B}_fs1.pt caches/canon_${B}_fs5.pt metrics/window3_${B}_e005_g098.pt; do
    [ -e "/workspace/$f" ] || die "missing /workspace/$f"
  done
done

# ---------------------------------------------------------------- train
log "6 PWM trainings (2 bases x 3 seeds), clean caches, gamma ${G}"
i=0
for B in lejepa pldm; do for sd in 0 1 2; do
  A=/workspace/actors/pwm_${B}_s${sd}.pt
  [ -f "$A" ] && { i=$((i+1)); continue; }
  CUDA_VISIBLE_DEVICES=$(( i % 6 )) timeout 28800 python3 "$PLAN/train_pwm_ac.py" \
    --cache /workspace/caches/canon_${B}_fs5.pt \
    --cache-td /workspace/caches/canon_${B}_fs1.pt \
    --wm "$(wm_of $B)" \
    --init-value /workspace/metrics/window3_${B}_e005_g098.pt \
    --gamma "$G" --amax "$(amax_of $B)" --horizon 5 --max-delta 12 \
    --n-step 50 --batch 128 --steps 8000 \
    --expectile 0.1 --expectile-final 0.03 --seed "$sd" \
    --out "$A" --out-value "/workspace/metrics/pwm_${B}_s${sd}_value.pt" \
    > "$L/train_${B}_s${sd}.log" 2>&1 || log "TRAIN FAILED ${B} s${sd}" &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done
wait
log "actors: $(ls /workspace/actors/pwm_*_s?.pt 2>/dev/null | wc -l)/6"
for B in lejepa pldm; do
  grep -m1 -oE "\[vframes\].*" "$L/train_${B}_s0.log" 2>/dev/null \
    || log "  NOTE ${B}: no [vframes] line -- check it used the 3-frame value"
done

# ---------------------------------------------------------------- card
ev(){ local gpu=$1 B=$2 nm=$3 seed=$4 actor=$5
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$(wm_of $B)" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=pwm solver.actor_path="$actor" solver.batch_size=10 \
    output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
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

log "carding PWM on episodes 8000:10000"
i=0
for B in lejepa pldm; do for sd in 0 1 2; do
  A=/workspace/actors/pwm_${B}_s${sd}.pt; [ -f "$A" ] || continue
  ( for s in $REP; do ev $(( i % 4 )) "$B" "pwm_${B}_s${sd}_e${s}" "$s" "$A"; done ) &
  i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
done; done
wait

log "========================= PWM (leaked) ========================="
for B in lejepa pldm; do
  ta=0; tb=0; k=0; nsc=0
  for sd in 0 1 2; do
    a=$(meanof "pwm_${B}_s${sd}_e" held10); c=$(meanof "pwm_${B}_s${sd}_e" held)
    n=$(grep -c "^pwm_${B}_s${sd}_e.*held=[0-9]" "$SUM"); nsc=$((nsc + n))
    [ "$n" -eq 0 ] && { log "  ${B} s${sd}: NOTHING SCORED"; continue; }
    log "  ${B} s${sd}: @0.1 ${a} | @0.05 ${c}  (${n}/6)"
    ta=$(awk "BEGIN{print $ta+$a}"); tb=$(awk "BEGIN{print $tb+$c}"); k=$((k+1))
  done
  [ "$k" -eq 0 ] && continue
  log "  POOLED ${B} PWM: @0.1 $(awk "BEGIN{printf \"%.1f\", $ta/$k}") | @0.05 $(awk "BEGIN{printf \"%.1f\", $tb/$k}")  (${nsc}/18 evals)"
  log "  bar $(bar_of $B) (Latent+CEM, 3000 rollouts) | LIP ~16 rollouts is the other reference"
done
log "PWM_DONE"
