#!/bin/bash
# Stage 1b: screen the I3 readout on the SAMPLING arms, selection seeds only.
#
# WHY THIS IS NOT OPTIONAL. The readout is part of the plan cost, so it has to
# be identical for every arm in the rh=1 table -- RLP and its baselines alike.
# Stage 1 screened it on RLP only. If I3 helps RLP and were adopted on that
# evidence, the baselines would be carded under a cost they never got the
# benefit of, and the margin would be an artefact of the readout.
#
# So: measure it on the bar (Latent+CEM) and on Value+CEM, and adopt the readout
# that is BEST FOR THE BASELINES. That is the choice that is conservative for
# the claim -- if RLP still wins under the baselines' own best readout, the
# margin is not a readout artefact.
#
# Both sampling arms reach the cost through _MetricCost's _m>=3 branch, which
# is a different code path from LIPSolver._V and needs its own evidence that I3
# fires (it crashed outright until the goal-broadcast fix).
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_rh1_bars.csv; touch "$SUM"
L=/workspace/logs/rh1bars; mkdir -p "$L"
SEL="50 51"
L2=/workspace/metrics/l2window3.pt
GPUS=${GPUS:-"0 1 2 3"}
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][bars] $*"; }
val_of(){ echo "/workspace/metrics/window3_${1}_e005_g098.pt"; }

ev(){ local gpu=$1 B=$2 nm=$3 seed=$4 mode=$5 dl=$6; shift 6
  grep -q "^${nm}," "$SUM" && return 0
  local DLARG=""; [ "$dl" = "1" ] && DLARG="+plan_config.deadline=50"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu I3_MODE=$mode \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy=/workspace/swm_home/checkpoints/${B}_reacher \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    plan_config.receding_horizon=1 $DLARG \
    "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10 i3
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  i3=$(grep -c "^\[I3\]" "$L/${nm}.log")
  if [ "$dl" = "1" ] && [ "$i3" = "0" ]; then
    log "  !! ${nm}: deadline set, I3 never fired -- not a valid I3 cell"; h=NOFIRE; t10=NOFIRE
  fi
  if [ -z "$h" ]; then
    # crash-vs-zero: a core-dumped sampling eval used to average in as a real 0.0
    log "  !! ${nm} NO SCORE: $(tail -2 "$L/${nm}.log" | head -1 | cut -c1-70)"; h=FAIL; t10=FAIL
  fi
  echo "${nm},held=${h},held10=${t10},i3=${i3}" >> "$SUM"
}

Q=$L/queue.txt; : > "$Q"
for B in lejepa pldm; do for R in off deadline min; do
  DL=1; [ "$R" = "off" ] && DL=0
  for s in $SEL; do
    echo "$B|l2cem_${B}_${R}_e${s}|$s|$R|$DL|solver=cem solver.n_steps=10 +metric=$L2" >> "$Q"
    echo "$B|tdcem_${B}_${R}_e${s}|$s|$R|$DL|solver=cem solver.n_steps=10 +metric=$(val_of $B)" >> "$Q"
  done
done; done
N=$(wc -l < "$Q"); W=0; for _ in $GPUS; do W=$((W + 1)); done
log "queue: $N sampling cells over GPUs [$GPUS] ($W concurrent -- the crash cap is ~4)"

k=0
for g in $GPUS; do
  ( idx=0
    while IFS='|' read -r B nm s R DL ARGS; do
      if [ $((idx % W)) -eq $k ]; then ev "$g" "$B" "$nm" "$s" "$R" "$DL" $ARGS; fi
      idx=$((idx + 1))
    done < "$Q" ) &
  k=$((k + 1))
done
wait

mean(){ grep "^${1}" "$SUM" | grep -oE "${2}=[0-9.]+" | cut -d= -f2 \
        | awk '{t+=$1;n++}END{printf "%5.1f(n=%d)", (n?t/n:0), n}'; }
log "===== sampling arms at rh=1, selection seeds, mean of 2 ====="
for M in held10 held; do
  lbl=$([ "$M" = held10 ] && echo "@0.1" || echo "@0.05")
  log "  --- $lbl ---           I3=off            I3=deadline       I3=min"
  for B in lejepa pldm; do
    printf "      %-22s %s   %s   %s\n" "Latent+CEM $B" \
      "$(mean l2cem_${B}_off_ $M)" "$(mean l2cem_${B}_deadline_ $M)" "$(mean l2cem_${B}_min_ $M)"
    printf "      %-22s %s   %s   %s\n" "Value+CEM $B" \
      "$(mean tdcem_${B}_off_ $M)" "$(mean tdcem_${B}_deadline_ $M)" "$(mean tdcem_${B}_min_ $M)"
  done
done
log "failures: $(grep -cE "NOFIRE|FAIL" "$SUM") of $N"
log "BARS_DONE -- adopt whichever readout is best for the BASELINES"
