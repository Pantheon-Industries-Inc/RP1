#!/bin/bash
# Re-run the TD+CEM arm on the Dyna world model, and retrain the collapsed seed.
#
# Two defects in the round-2 Dyna cards, both of which produced a number rather
# than an error:
#
# 1. TD+CEM reported "HELD 0.0 | @0.1 0.0". That arm did not score zero, it
#    CORE-DUMPED under timeout. The driver's ev() writes held=FAIL when the
#    grep finds nothing, meanof() skips FAIL rows, and an all-FAIL arm averages
#    to 0.0 -- indistinguishable in the log from a genuine total failure to
#    reach the goal. Re-run here, one eval at a time on the now-idle GPU 0, so a
#    repeat crash is visible as a crash.
#
# 2. LIP seed 0 collapsed to 1.0 (0-4 on every eval seed) while s1/s2 gave
#    39.3/66.7. On the BASE world model the identical recipe made s0 the best of
#    the three (57.0), so this is specific to the fine-tuned model, not the
#    recipe. One dead seed dragged the pooled figure to 35.7 and buried the fact
#    that the healthy seeds average 53.0 -- above the 48.9 base-WM incumbent.
#    Retrained here on two fresh seeds (3,4) to establish whether the collapse
#    is a rare event or a property of this world model. Reported either way.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
FT=/workspace/swm_home/checkpoints/dyna_r2_lejepa
W3=/workspace/metrics/window3_dyna_lejepa_e005_st6000.pt
C1=/workspace/caches/dyna_lejepa_fs1.pt
C5=/workspace/caches/dyna_lejepa_fs5.pt
SUM=/workspace/results/summary_dyna_lejepa.csv; touch "$SUM"
L=/workspace/logs/dyna; mkdir -p "$L"
REP="42 43 44 45 46 47"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][dynafix] $*"; }

ev(){ local gpu=$1 nm=$2 seed=$3; shift 3
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$FT" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver.batch_size=10 "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local rc=$? h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  if [ -z "$h" ]; then
    log "  !! ${nm} produced no score (exit ${rc}) -- $(tail -1 "$L/${nm}.log" | cut -c1-70)"
    echo "${nm},held=FAIL,held10=FAIL" >> "$SUM"
  else
    echo "${nm},held=${h},held10=${t10:-FAIL}" >> "$SUM"
  fi
}
meanof(){ local pre=$1 key=$2; local t=0 n=0 e
  for s in $REP; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"; }

# ---------------------------------------------------------------- 1. TD+CEM, serial
log "TD+CEM on the Dyna WM, serial on GPU 0 (previous attempt core-dumped)"
for s in $REP; do ev 0 "dy_td_s${s}" $s solver=cem solver.n_steps=10 "+metric=$W3"; done
n_ok=$(grep -c "^dy_td_s.*,held=[0-9]" "$SUM")
log "TD+CEM (Dyna WM): HELD $(meanof dy_td_s held) | @0.1 $(meanof dy_td_s held10)  [${n_ok}/6 scored, base 17.7/45.3]"

# ---------------------------------------------------------------- 2. two more LIP seeds
log "retraining LIP seeds 3,4 at the swept recipe to test the s0 collapse"
for sd in 3 4; do
  A=/workspace/actors/lip4_dyna_sw_s${sd}.pt
  [ -f "$A" ] && continue
  CUDA_VISIBLE_DEVICES=0 timeout 21600 python3 "$PLAN/train_lip_ac.py" \
    --cache "$C5" --cache-td "$C1" --h5 "$SLIM" --wm "$FT" \
    --pad-context --init-value "$W3" \
    --arch v4 --amax 2.2 --iters 8 --horizon 5 --max-delta 12 \
    --steps 1000 --batch 128 --n-step 50 \
    --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 \
    --replay-prob 0.5 --expand-weight 0 \
    --lambda-schedule uniform --mean-weight 0.1 --seed "$sd" \
    --out "$A" --out-value "/workspace/metrics/lip4_dyna_sw_s${sd}_value.pt" \
    > "$L/lip_s${sd}.log" 2>&1 && log "  LIP s${sd} trained" || log "  LIP s${sd} FAILED"
done
for sd in 3 4; do
  A=/workspace/actors/lip4_dyna_sw_s${sd}.pt; [ -f "$A" ] || continue
  for s in $REP; do ev 0 "dy_lip${sd}_s${s}" $s solver=lip solver.actor_path="$A" solver.rollout_compat=false; done
  log "  CARD LIP s${sd} (Dyna WM): HELD $(meanof dy_lip${sd}_s held) | @0.1 $(meanof dy_lip${sd}_s held10)"
done

log "--- all LIP seeds on the Dyna WM ---"
t=0; k=0; t10=0
for sd in 0 1 2 3 4; do
  v=$(meanof "dy_lip${sd}_s" held); w=$(meanof "dy_lip${sd}_s" held10)
  [ "$v" = "0.0" ] && [ "$w" = "0.0" ] && continue
  log "  s${sd}: HELD ${v} | @0.1 ${w}"
  t=$(awk "BEGIN{print $t+$v}"); t10=$(awk "BEGIN{print $t10+$w}"); k=$((k+1))
done
[ "$k" -gt 0 ] && log "POOLED over ${k} seeds: HELD $(awk "BEGIN{printf \"%.1f\", $t/$k}") | @0.1 $(awk "BEGIN{printf \"%.1f\", $t10/$k}")"
log "bars on the Dyna WM: Latent+CEM 53.0 / 86.3   |  base-WM LIP 49.8 / 89.0"
log "DYNAFIX_DONE"
