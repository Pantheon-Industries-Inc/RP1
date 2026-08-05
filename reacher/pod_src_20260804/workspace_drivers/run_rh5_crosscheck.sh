#!/bin/bash
# Stage 1c: is the alignment retrain cadence-SPECIFIC or a strict improvement?
#
# The rh=1 screen says dense/prefix alignment is worth ~20 points @0.1 at rh=1.
# That leaves one question the rh=1 table cannot answer on its own: what do those
# same actors do at rh=5, the REPORTED cadence?
#
#   dense >= terminal at rh=5  -> the alignment is a strict improvement, and the
#                                 reported table understates RLP. That has to be
#                                 said out loud rather than quietly banked.
#   dense <  terminal at rh=5  -> it is cadence-specific: match the training cost
#                                 to the deployed replanning cadence. Cleaner
#                                 story, and it leaves the reported table intact.
#
# Selection seeds only. I3 off throughout -- at rh=5 it is inert anyway
# (chunks_remaining is 10 then 5, never below the plan length 5), so this also
# double-checks that inertness on the retrained actors.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_rh5_cross.csv; touch "$SUM"
L=/workspace/logs/rh5cross; mkdir -p "$L"
SEL="50 51"
GPUS=${GPUS:-"4 5"}
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][cross] $*"; }

ev(){ local gpu=$1 B=$2 nm=$3 seed=$4 act=$5
  grep -q "^${nm}," "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 5400 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy=/workspace/swm_home/checkpoints/${B}_reacher \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    plan_config.receding_horizon=5 \
    solver=lip solver.actor_path="$act" solver.rollout_compat=false \
    output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && { log "  !! ${nm} no score"; h=FAIL; t10=FAIL; }
  echo "${nm},held=${h},held10=${t10}" >> "$SUM"
}

Q=$L/queue.txt; : > "$Q"
for B in lejepa pldm; do for s in $SEL; do
  echo "$B|x_ref_${B}_e${s}|$s|/workspace/actors/lip4_leak6_${B}_s0.pt" >> "$Q"
  for M in terminal dense prefix randhorizon; do for AM in 18 22; do
    A=/workspace/actors/rh1_${B}_${M}_a${AM}.pt
    [ -f "$A" ] && echo "$B|x_${B}_${M}_a${AM}_e${s}|$s|$A" >> "$Q"
  done; done
done; done
N=$(wc -l < "$Q"); W=0; for _ in $GPUS; do W=$((W + 1)); done
log "queue: $N cells at rh=5 over GPUs [$GPUS]"

k=0
for g in $GPUS; do
  ( idx=0
    while IFS='|' read -r B nm s A; do
      if [ $((idx % W)) -eq $k ]; then ev "$g" "$B" "$nm" "$s" "$A"; fi
      idx=$((idx + 1))
    done < "$Q" ) &
  k=$((k + 1))
done
wait

mean(){ grep "^${1}" "$SUM" | grep -oE "${2}=[0-9.]+" | cut -d= -f2 \
        | awk '{t+=$1;n++}END{printf "%5.1f(n=%d)", (n?t/n:0), n}'; }
log "===== SAME actors at rh=5 (reported cadence), selection seeds ====="
for B in lejepa pldm; do
  log "  --- $B ---              @0.1              @0.05"
  printf "      %-22s %s   %s\n" "rh5-trained (reported)" \
    "$(mean x_ref_${B}_ held10)" "$(mean x_ref_${B}_ held)"
  for M in terminal dense prefix randhorizon; do for AM in 18 22; do
    printf "      %-22s %s   %s\n" "${M}_a${AM}" \
      "$(mean x_${B}_${M}_a${AM}_ held10)" "$(mean x_${B}_${M}_a${AM}_ held)"
  done; done
done
log "CROSS_DONE"
