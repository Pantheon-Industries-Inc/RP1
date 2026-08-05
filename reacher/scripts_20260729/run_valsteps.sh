#!/bin/bash
# Does the shared window value just need more training?
#
# The learned TD value is the weak link on BOTH bases: as a CEM cost it scores
# 14.3 (pldm) / 17.7 (lejepa), while the PARAMETER-FREE L2 window cost scores
# 39.3 / 44.7 on the same protocol. A learned value losing 2.5-3x to plain L2 is
# not a subtle miscalibration, and the trainer's budget explains why it might be
# undertrained: batch 1024 x 6000 steps = 6.1M samples over a 2.01M-row cache,
# i.e. about 3 epochs, in 127 seconds.
#
# So: hold everything else fixed and walk --steps up. This is a SHARED knob --
# the identical ladder runs on both bases and the winner must be one value of
# --steps for both, per the user's constraint that TD/dataset knobs not differ.
#
# READ-OUT is TD+CEM on the value itself, not LIP. That isolates value quality
# from planner quality, and it has two known reference points:
#     control  (6000 steps)  14.3 pldm / 17.7 lejepa
#     L2 ceiling             39.3 pldm / 44.7 lejepa
# If more steps close that gap, the value was undertrained and LIP should
# inherit the gain. If TD+CEM is flat across a 25x step range, the value is not
# step-limited and the problem is the objective or the head, not the budget --
# equally useful to know, and it kills the "just train longer" hypothesis
# cheaply rather than after another full LIP sweep.
#
# Both expectiles are carried because stage 2 has not yet fixed the shared
# expectile, and step count may interact with it.
#
# Controls are re-run on the SELECTION seeds so the comparison is like-for-like
# (the 14.3/17.7 figures are reporting seeds; the ladder screens on selection).
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_valsteps.csv; touch "$SUM"
L=/workspace/logs/valsteps; mkdir -p "$L"
SEL="50 51"
LADDER="20000 60000 150000"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][vsteps] $*"; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }

# ---------------------------------------------------------------- 1. train the ladder
# Ascending order so the cheap rungs report before the expensive one finishes.
# ~127 s per 6000 steps => 20k ~ 7 min, 60k ~ 21 min, 150k ~ 53 min.
log "training values: {${LADDER}} steps x 2 bases x 2 expectiles"
for st in $LADDER; do
  i=0
  for B in lejepa pldm; do for EX in 005 01; do
    case "$EX" in 005) EXV=0.05;; 01) EXV=0.1;; esac
    W=/workspace/metrics/window3_${B}_e${EX}_st${st}.pt
    [ -f "$W" ] && { i=$((i + 1)); continue; }
    CUDA_VISIBLE_DEVICES=$((i % 6)) python3 /workspace/train_window.py \
      --cache /workspace/caches/canon_${B}_fs1.pt --lag 5 --frames 3 \
      --expectile "$EXV" --n-step 50 --steps "$st" --seed 0 --out "$W" \
      > "$L/w3_${B}_e${EX}_st${st}.log" 2>&1 \
      && log "  value ${B} e${EX} ${st} steps DONE (loss $(grep -oE 'final loss=[0-9.]+' "$L/w3_${B}_e${EX}_st${st}.log" | tail -1 | cut -d= -f2))" \
      || log "  value ${B} e${EX} ${st} steps FAILED" &
    i=$((i + 1))
  done; done
  wait
  log "rung ${st} trained"
done

# ---------------------------------------------------------------- 2. TD+CEM on each value
ev(){ local gpu=$1 B=$2 nm=$3 seed=$4 metric=$5
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$(wm_of $B)" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=cem solver.n_steps=10 solver.batch_size=10 \
    "+metric=${metric}" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
# cut -d= -f2, NOT a bare [0-9.]+ grep: "held10=80.0" would otherwise yield the
# 10 from the key name, which silently printed a constant 10.0 in earlier drivers
meanof(){ local pre=$1 key=$2; local t=0 n=0 e
  for s in $SEL; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

log "TD+CEM on every rung (plus the 6000 control on the SAME selection seeds)"
i=0
for B in lejepa pldm; do for EX in 005 01; do
  ( # control first so the baseline lands early
    ev $((i % 6)) "$B" "vs_${B}_e${EX}_st6000_sel50" 50 "/workspace/metrics/window3_${B}_e${EX}.pt"
    ev $((i % 6)) "$B" "vs_${B}_e${EX}_st6000_sel51" 51 "/workspace/metrics/window3_${B}_e${EX}.pt"
    for st in $LADDER; do
      W=/workspace/metrics/window3_${B}_e${EX}_st${st}.pt; [ -f "$W" ] || continue
      for s in $SEL; do ev $((i % 6)) "$B" "vs_${B}_e${EX}_st${st}_sel${s}" $s "$W"; done
    done ) &
  i=$((i + 1))
done; done
wait

# ---------------------------------------------------------------- 3. report
log "================= VALUE STEPS LADDER (TD+CEM, selection seeds) ================="
log "reference: L2 window cost = 39.3 pldm / 44.7 lejepa (reporting seeds)"
: > /workspace/results/valsteps_table.txt
for B in lejepa pldm; do for EX in 005 01; do
  row="${B} e${EX}:"
  for st in 6000 $LADDER; do
    v=$(meanof "vs_${B}_e${EX}_st${st}_sel" held)
    row="${row}  ${st}=${v}"
    echo "$v $B $EX $st" >> /workspace/results/valsteps_table.txt
  done
  log "  $row"
done; done

log "--- best shared step count by cross-base mean of each base's best expectile ---"
for st in 6000 $LADDER; do
  tot=0; k=0
  for B in lejepa pldm; do
    b=$(awk -v b="$B" -v s="$st" '$2==b && $4==s {print $1}' \
        /workspace/results/valsteps_table.txt | sort -rn | head -1)
    [ -z "$b" ] && continue
    tot=$(awk "BEGIN{print $tot+$b}"); k=$((k+1))
  done
  [ "$k" -lt 2 ] && continue
  log "  steps ${st}: cross-base mean $(awk "BEGIN{printf \"%.1f\", $tot/$k}")"
done
log "VALSTEPS_DONE"
