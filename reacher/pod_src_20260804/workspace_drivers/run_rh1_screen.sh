#!/bin/bash
# Stage 1: screen the rh=1 retrained actors AND the I3 readout, on SELECTION
# seeds only (50, 51). Nothing here is reportable -- reporting seeds are 42-47
# and are not touched until the winner is fixed.
#
# TWO CHOICES ARE OPEN AND BOTH ARE SCREENED HERE, CROSSED:
#
#   alignment mode  which rollout timesteps entered the TRAINING loss
#                   {terminal, dense, prefix, randhorizon} x amax {1.8, 2.2}
#                   (16 actors from run_rh1_retrain.sh, 6000 steps, seed 0)
#
#   I3 readout      which imagined chunk the PLAN COST scores at eval time
#     off           terminal chunk -- overshoots the deadline by design
#     deadline      the chunk that lands on the graded step (cr-1)
#     min           the best still-reachable chunk (min_j, j < cr)
#
# The reported rh=5 actor (lip4_leak6_*_s0) is carded under all three readouts
# as the reference: it is the artifact the paper's table already contains, so
# its rh=1 row is the honest "same artifact, published cadence" ablation, while
# the retrained actors answer "and if you train for that cadence".
#
# I3 is inert without plan_config.deadline and inert while chunks_remaining >= 5
# (the plan length). At rh=5 the count is 10 then 5 -- verified: the rh=5 cell
# scores 66.0/100.0 with a deadline set, bit-identical to the reported number.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_rh1_screen.csv; touch "$SUM"
L=/workspace/logs/rh1screen; mkdir -p "$L"
SEL="50 51"
GPUS=${GPUS:-"4 5"}
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][screen] $*"; }

ev(){ local gpu=$1 B=$2 nm=$3 seed=$4 mode=$5 act=$6 dl=$7
  grep -q "^${nm}," "$SUM" && return 0
  local DLARG=""; [ "$dl" = "1" ] && DLARG="+plan_config.deadline=50"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu I3_MODE=$mode \
  timeout 3600 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy=/workspace/swm_home/checkpoints/${B}_reacher \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    plan_config.receding_horizon=1 $DLARG \
    solver=lip solver.actor_path="$act" solver.rollout_compat=false \
    output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10 i3
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  i3=$(grep -c "^\[I3\]" "$L/${nm}.log")
  # Guard 5 from the handoff: a dead I3 returns the unaligned baseline verbatim
  # and looks like a legitimate result. Record whether it actually fired.
  if [ "$dl" = "1" ] && [ "$i3" = "0" ]; then
    log "  !! ${nm}: deadline set but I3 NEVER FIRED -- not a valid I3 cell"
    t10="NOFIRE"; h="NOFIRE"
  fi
  [ -z "$h" ] && { log "  !! ${nm} no score: $(tail -2 "$L/${nm}.log" | head -1 | cut -c1-70)"; h=FAIL; t10=FAIL; }
  echo "${nm},held=${h},held10=${t10},i3=${i3}" >> "$SUM"
}

# ------------------------------------------------------------- build the queue
Q=/workspace/logs/rh1screen/queue.txt; : > "$Q"
for B in lejepa pldm; do
  for R in off deadline min; do
    DL=1; [ "$R" = "off" ] && DL=0
    for s in $SEL; do
      # reference: the actor the reported table uses, at rh=1
      echo "$B|ref_${B}_${R}_e${s}|$s|$R|/workspace/actors/lip4_leak6_${B}_s0.pt|$DL" >> "$Q"
      for M in terminal dense prefix randhorizon; do for AM in 18 22; do
        A=/workspace/actors/rh1_${B}_${M}_a${AM}.pt
        [ -f "$A" ] || continue
        echo "$B|${B}_${M}_a${AM}_${R}_e${s}|$s|$R|$A|$DL" >> "$Q"
      done; done
    done
  done
done
N=$(wc -l < "$Q"); log "queue: $N cells over GPUs [$GPUS]"

# --------------------------------------------------------------- run, sharded
# Fixed sharding, one sequential worker pinned per GPU: the cells are
# homogeneous (~60 s each) so imbalance is small, and every render keeps its own
# MUJOCO_EGL_DEVICE_ID instead of two evals racing on one device.
W=0; for _ in $GPUS; do W=$((W + 1)); done
k=0
for g in $GPUS; do
  ( idx=0
    while IFS='|' read -r B nm s R A DL; do
      if [ $((idx % W)) -eq $k ]; then ev "$g" "$B" "$nm" "$s" "$R" "$A" "$DL"; fi
      idx=$((idx + 1))
    done < "$Q" ) &
  k=$((k + 1))
done
wait

# --------------------------------------------------------------------- report
mean(){ grep "^${1}" "$SUM" | grep -oE "${2}=[0-9.]+" | cut -d= -f2 \
        | awk '{t+=$1;n++}END{printf "%5.1f(n=%d)", (n?t/n:0), n}'; }
log "===== rh=1 screen, SELECTION seeds 50-51, mean over 2 cells ====="
log "  reported rh=5 reference for this actor family: lejepa 98.2 / pldm 94.2 @0.1"
for B in lejepa pldm; do
  log "  --- $B ---            I3=off            I3=deadline       I3=min"
  printf "      %-22s %s   %s   %s\n" "rh5-trained (reported)" \
    "$(mean ref_${B}_off_ held10)" "$(mean ref_${B}_deadline_ held10)" "$(mean ref_${B}_min_ held10)"
  for M in terminal dense prefix randhorizon; do for AM in 18 22; do
    printf "      %-22s %s   %s   %s\n" "${M}_a${AM}" \
      "$(mean ${B}_${M}_a${AM}_off_ held10)" \
      "$(mean ${B}_${M}_a${AM}_deadline_ held10)" \
      "$(mean ${B}_${M}_a${AM}_min_ held10)"
  done; done
done
log "same table @0.05 (tighter tolerance, where the readout choice hurt on seed 42)"
for B in lejepa pldm; do
  log "  --- $B @0.05 ---      I3=off            I3=deadline       I3=min"
  printf "      %-22s %s   %s   %s\n" "rh5-trained (reported)" \
    "$(mean ref_${B}_off_ held)" "$(mean ref_${B}_deadline_ held)" "$(mean ref_${B}_min_ held)"
  for M in terminal dense prefix randhorizon; do for AM in 18 22; do
    printf "      %-22s %s   %s   %s\n" "${M}_a${AM}" \
      "$(mean ${B}_${M}_a${AM}_off_ held)" \
      "$(mean ${B}_${M}_a${AM}_deadline_ held)" \
      "$(mean ${B}_${M}_a${AM}_min_ held)"
  done; done
done
log "cells with a deadline that never fired I3: $(grep -c NOFIRE "$SUM")"
log "SCREEN_DONE"
