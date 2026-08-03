#!/bin/bash
# Dyna on DINO-WM, stage 1: gate on a LIP actor, evaluate it, then collect
# on-policy rollouts with it.
#
# WHY LIP-ONLY: collection needs ~1800 episodes. CEM costs 2166 s per 50 tasks
# => ~22 h, which is infeasible. A LIP actor plans in one forward pass per step,
# so collection is expected to be ~1-2 h. The actual rate is MEASURED from the
# eval below and projected before the collection starts, rather than assumed.
#
# SPLIT: collection draws from episodes 0-7999 only (train side); every eval in
# this campaign draws 8000-9999. terminate_at_goal=False so all episodes run the
# full 50 steps -- the campaign's key fix, since the WM needs a 25-frame window
# and early-terminating (successful) episodes were otherwise silently filtered
# out, biasing the fine-tune set toward failures.
#
# CAVEAT specific to this WM: DINOv2 is FROZEN, so a later fine-tune can only
# adapt the 6-block predictor and the action embedder (~10% of parameters). The
# LeWM Dyna result (+3.6) fine-tuned a fully trainable WM, so a weaker effect
# here would not be surprising -- this tests Dyna when the representation cannot
# move.
#
# Stops after collection. The fine-tune (stage 2) needs a PreJEPA warm-start
# config, which is set up separately rather than fired blind.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_dino
EXPERT=/root/datasets/ogb_cube_single/ogb_cube_single.lance
CKPT=/workspace/ckpts/dinowm_noprop_cube
SUM=$R/summary_lipdino.csv
NCALL=${NCALL:-12}          # calls x 50 tasks = episodes collected
mkdir -p "$L" "$R" "$D" /workspace/videos_scratch; touch "$SUM"; cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_dyna_dino.log"; }

# ---------------------------------------------------------------- 1. the gate
log "=== waiting for a LIP actor checkpoint ==="
ACTOR=""
for i in $(seq 1 240); do
  for cand in /workspace/actors/lipdino_ck_s1500.pt /workspace/actors/lipdino_ck_s1250.pt \
              /workspace/actors/lipdino_ck_s1000.pt /workspace/actors/lipdino_ck_s750.pt \
              /workspace/actors/lipdino_ck_s500.pt /workspace/actors/lipdino_ck_s250.pt \
              /workspace/actors/lipdino_b32.pt; do
    [ -f "$cand" ] && { ACTOR="$cand"; break; }
  done
  [ -n "$ACTOR" ] && break
  sleep 60
done
[ -n "$ACTOR" ] || { log "FATAL no LIP actor appeared"; exit 1; }
log "  using actor $ACTOR"

# ------------------------------------------------- 2. evaluate it (held-out)
log "=== LIP eval, held-out eps 8000-9999 (refs: plain CEM 80.0, TD+CEM 76.0) ==="
for d in 44 42 43; do
  nm="lipdino_e${d}"
  grep -q "^${nm}," "$SUM" && { log "  $nm cached"; continue; }
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 14400 python3 "$P/eval_wm_dino.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" eval.img_size=196 eval.goal_offset_steps=25 \
    eval.eval_budget=50 "+eval.ep_range=8000:10000" policy="$CKPT" solver=lip \
    "solver.actor_path=$ACTOR" "+video_dir=/workspace/videos_scratch/$nm" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  t1=$(date +%s); dt=$((t1-t0))
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL},${dt}" >> "$SUM"
  log "  $nm = ${sr:-FAIL}  (${dt}s for 50 tasks)"
  if [ -z "$sr" ]; then
    log "    ERR: $(grep -E 'Error|Traceback' "$L/eval_${nm}.log" | tail -2 | tr '\n' ' ')"
    log "    LIP eval failed -- not collecting on a broken actor"
    exit 1
  fi
done

# project the collection cost from the MEASURED eval rate
RATE=$(grep "^lipdino_e" "$SUM" | tail -1 | cut -d, -f3)
log "  measured $RATE s per 50 tasks => ${NCALL} calls ~ $(( RATE * NCALL / 60 )) min for $(( NCALL * 50 )) episodes"

# ------------------------------------------------------------- 3. collection
log "=== collecting $(( NCALL * 50 )) on-policy episodes, eps 0-7999, terminate_at_goal=False ==="
REC=$D/onpolicy_lipdino.lance
for i in $(seq 0 $((NCALL-1))); do
  lg="$L/dynacol_c${i}.log"
  grep -q "success_rate" "$lg" 2>/dev/null && { log "  call $i cached"; continue; }
  SWM_RECORD_PATH=$REC SWM_RECORD_OUTCOME=qpos_in_ball \
  CUDA_VISIBLE_DEVICES=0 timeout 14400 python3 "$P/eval_wm_dino.py" --config-name cube \
    seed=$((1000 + i)) eval.dataset_name="$EXPERT" eval.img_size=196 \
    eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=50 \
    "+eval.ep_range=0:8000" world.terminate_at_goal=False \
    policy="$CKPT" solver=lip "solver.actor_path=$ACTOR" \
    output.filename="dynacol_c${i}.txt" > "$lg" 2>&1
  grep -q "ep_range 0:8000" "$lg" || { log "  call $i: ep_range NOT applied -- split violated, aborting"; exit 1; }
  log "  call $i done: $(grep -oE 'success_rate[^0-9]*[0-9.]+' "$lg" | tail -1)"
done

log "=== collection composition ==="
python3 - "$REC" <<'PY' 2>&1 | tee -a "$L/driver_dyna_dino.log"
import sys, lance, numpy as np
d = lance.dataset(sys.argv[1]); n = d.count_rows()
t = d.take(list(range(n)), columns=["episode_idx", "step_idx"]).to_pydict()
e = np.asarray(t["episode_idx"]).reshape(-1)
print(f"rows {n} | episodes {len(np.unique(e))} | rows/ep {n/max(len(np.unique(e)),1):.1f}")
PY
log "DYNA_DINO_STAGE1_DONE (fine-tune is stage 2, set up separately)"
