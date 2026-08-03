#!/bin/bash
# Dyna on DINO-WM, stage 1 (v2): gate on a LIP actor, eval it, collect with it.
#
# v2 fixes every failure mode found in the v1 pre-flight audit:
#   - actor candidate list now includes the names the trainer ACTUALLY writes
#     (lipdino_v2_sN.pt from --ckpt-every, plus the existing lipdino_ck.pt).
#     v1 listed names that never appear, so it would have waited 4 h and exited.
#   - SWM_RECORD_OUTCOME removed. v1 hardcoded `qpos_in_ball`, which is the
#     REACHER success flag; world.py raises if the named column is not
#     published, so every cube collection call would have aborted. Default
#     'auto' records `success`, verified by smoke test (cols=...,success).
#   - gate waits up to 8 h (was 4) and re-scans for a LATER checkpoint each pass,
#     so it picks up the best actor available rather than the first.
#   - no pkill anywhere: `pkill -f <pat>` matches this script's own cmdline.
#   - every collection call is verified two ways (ep_range applied AND a
#     [record] kept= line present) before the next one starts.
#
# Measured facts this plan relies on:
#   LIP solve 1.08 s => ~3 min per 50 tasks (CEM was 37 min, i.e. 22 h for 1800
#   episodes -- which is why this is LIP-only).
#   LIP step-250 actor scored 70.0 on held-out draw 44 (plain CEM 78.0).
#
# CAVEAT: DINOv2 is frozen, so a later fine-tune can only move the predictor and
# action embedder (~10% of params). LeWM's Dyna +3.6 moved a fully trainable WM.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_dino
EXPERT=/root/datasets/ogb_cube_single/ogb_cube_single.lance
CKPT=/workspace/ckpts/dinowm_noprop_cube
SUM=$R/summary_lipdino.csv
NCALL=${NCALL:-12}
MINSTEP=${MINSTEP:-500}     # prefer an actor trained at least this many steps
mkdir -p "$L" "$R" "$D" /workspace/videos_scratch; touch "$SUM"; cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_dyna_v2.log"; }

pick_actor(){
  local best="" bs=-1 f s
  for f in /workspace/actors/lipdino_v2_s*.pt; do
    [ -f "$f" ] || continue
    s=$(basename "$f" .pt); s=${s##*_s}
    case "$s" in (*[!0-9]*) continue;; esac
    [ "$s" -gt "$bs" ] && { bs=$s; best=$f; }
  done
  if [ -z "$best" ] && [ -f /workspace/actors/lipdino_v2.pt ]; then
    best=/workspace/actors/lipdino_v2.pt; bs=9999
  fi
  if [ -z "$best" ] && [ -f /workspace/actors/lipdino_ck.pt ]; then
    best=/workspace/actors/lipdino_ck.pt; bs=250
  fi
  echo "$best $bs"
}

log "=== gate: waiting for a LIP actor with >= $MINSTEP steps (8 h max) ==="
ACTOR=""; STEPS=0
for i in $(seq 1 480); do
  read -r cand cs <<<"$(pick_actor)"
  if [ -n "$cand" ]; then
    ACTOR="$cand"; STEPS="$cs"
    [ "$cs" -ge "$MINSTEP" ] && break
  fi
  sleep 60
done
[ -n "$ACTOR" ] || { log "FATAL no LIP actor appeared in 8 h"; exit 1; }
log "  actor $ACTOR (step $STEPS)"
python3 - "$ACTOR" <<'PY' 2>&1 | tee -a "$L/driver_dyna_v2.log"
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
bad = [k for k in ("kind", "value", "amu5", "ast5", "z_dim") if ck.get(k) is None]
print(f"[gate] kind={ck.get('kind')} z_dim={ck.get('z_dim')} amax={ck.get('amax')!r}")
assert not bad, f"actor missing fields {bad}"
assert ck.get("amax") is not None, "amax is None -> LIPSolver crashes on -amax"
print("[gate] actor checkpoint OK")
PY
grep -q "actor checkpoint OK" "$L/driver_dyna_v2.log" || { log "FATAL actor blob invalid"; exit 1; }

log "=== LIP eval, held-out 8000-9999 (refs: plain CEM 80.0, TD+CEM 76.0) ==="
RATE=0
for d in 44 42 43; do
  nm="lipv2_s${STEPS}_e${d}"
  grep -q "^${nm}," "$SUM" && { log "  $nm cached"; continue; }
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm_dino.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" eval.img_size=196 eval.goal_offset_steps=25 \
    eval.eval_budget=50 "+eval.ep_range=8000:10000" policy="$CKPT" solver=lip \
    "solver.actor_path=$ACTOR" "+video_dir=/workspace/videos_scratch/$nm" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  t1=$(date +%s); RATE=$((t1-t0))
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL},${RATE}" >> "$SUM"
  log "  $nm = ${sr:-FAIL}  (${RATE}s / 50 tasks)"
  [ -n "$sr" ] || { log "FATAL LIP eval failed; not collecting on a broken actor"; exit 1; }
done
log "  projection: ${NCALL} calls x ~${RATE}s = ~$(( RATE * NCALL / 60 )) min for $(( NCALL*50 )) episodes"

log "=== collect $(( NCALL*50 )) episodes, eps 0-7999, terminate_at_goal=False ==="
REC=$D/onpolicy_lipdino.lance
for i in $(seq 0 $((NCALL-1))); do
  lg="$L/dynacol_v2_c${i}.log"
  grep -q "\[record\] kept=" "$lg" 2>/dev/null && { log "  call $i cached"; continue; }
  SWM_RECORD_PATH=$REC CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm_dino.py" \
    --config-name cube seed=$((1000 + i)) eval.dataset_name="$EXPERT" \
    eval.img_size=196 eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=50 \
    "+eval.ep_range=0:8000" world.terminate_at_goal=False \
    policy="$CKPT" solver=lip "solver.actor_path=$ACTOR" \
    output.filename="dynacol_v2_c${i}.txt" > "$lg" 2>&1
  grep -q "ep_range 0:8000" "$lg" || { log "FATAL call $i: ep_range NOT applied (split violated)"; exit 1; }
  grep -q "\[record\] kept=" "$lg" || { log "FATAL call $i: nothing recorded"; tail -3 "$lg" | tee -a "$L/driver_dyna_v2.log"; exit 1; }
  log "  call $i: $(grep -oE '\[record\] kept=[0-9]+ dropped=[0-9]+[^-]*' "$lg" | tail -1) $(grep -oE 'success_rate[^0-9]*[0-9.]+' "$lg" | tail -1)"
done

log "=== composition ==="
python3 - "$REC" <<'PY' 2>&1 | tee -a "$L/driver_dyna_v2.log"
import sys, lance, numpy as np
d = lance.dataset(sys.argv[1]); n = d.count_rows()
t = d.take(list(range(n)), columns=["episode_idx", "success"]).to_pydict()
e = np.asarray(t["episode_idx"]).reshape(-1)
s = np.asarray(t["success"]).reshape(-1).astype(float)
ne = len(np.unique(e))
print(f"rows {n} | episodes {ne} | rows/ep {n/max(ne,1):.1f} | "
      f"success rows {s.sum():.0f} ({100*s.mean():.1f}%)")
PY
log "DYNA_DINO_V2_STAGE1_DONE (fine-tune = stage 2, set up separately)"
