#!/bin/bash
# Unified work-queue scheduler for the LIPv4-dino grid.
#
# Replaces the gated round-3 script, which waited for ALL trainers to exit --
# including the 6000-step cell (10 h). That left two GPUs idle ~12:45-17:45 for
# no reason. Here a cell launches the moment ANY slot frees, so the only
# serialization left is the physical one.
#
# Slot detection is by GPU FREE MEMORY, not PID bookkeeping: round 2's trainers
# were launched by a different script, and pattern-based process checks
# (pgrep/pkill -f) alias onto the invoking ssh shell -- which produced a false
# "already alive" and three killed sessions tonight.
#
# Capacity facts (measured, not assumed):
#   ~6 s/step at batch 32 / iters 8  => 2000 steps = 3.3 h, 3000 = 5 h, 6000 = 10 h
#   each trainer holds ~127 GB of a 143 GB card => exactly ONE trainer per GPU
#   GPU 0 is reserved: EGL evals + concurrent training on one GPU is the
#   combination that corrupts runs, so the eval daemon owns it alone.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1
CODE=/workspace/code/stable-worldmodel; L=/workspace/logs
mkdir -p "$L" /workspace/actors /workspace/metrics
cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_sched_fx.log"; }

STEPS=3000   # match the pre-fix control (72.0) for a like-for-like read
FREE_MB=8000        # a live trainer sits at ~127 GB; <8 GB means the slot is idle

# tag|extra args (amax 1.6, actor-lr 3e-4, iters 8 unless overridden)
CELLS=(
  "ctrl|"
  "a12|--amax 1.2"
  "a20|--amax 2.0"
  "exp0.5|--expand-weight 0.5"
  "rep0.25|--replay-prob 0.25"
  "exp1.0|--expand-weight 1.0"
  "rep0.5|--replay-prob 0.5"
  "exp0.5rep0.25|--expand-weight 0.5 --replay-prob 0.25"
  "lr1e3|--actor-lr 1e-3 --actor-lr-final 1e-4"
  "iters16|--iters 16"
)

free_gpu(){    # echo an idle GPU id from 1..3, or nothing
  local g mb
  for g in 1 2 3; do
    mb=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -n "$mb" ] && [ "$mb" -lt "$FREE_MB" ] && { echo "$g"; return; }
  done
}

log "=== scheduler up: ${#CELLS[@]} cells, $STEPS steps, slots 1-3 (0 = evals) ==="
for cell in "${CELLS[@]}"; do
  tag="fx_${cell%%|*}"; extra="${cell#*|}"
  if [ -f "/workspace/actors/${tag}_s${STEPS}.pt" ]; then log "  $tag done, skip"; continue; fi
  g=""
  while [ -z "$g" ]; do g=$(free_gpu); [ -z "$g" ] && sleep 120; done
  # shellcheck disable=SC2086
  nohup env CUDA_VISIBLE_DEVICES=$g python3 scripts/plan/train_lip_ac_dino.py \
    --cache /workspace/caches/dinopool_tr8000_fs5.pt \
    --cache-td /workspace/caches/dinopool_tr8000_fs1.pt \
    --dataset /root/datasets/ogb_cube_single/ogb_cube_single.lance \
    --h5 /workspace/datasets/expert_actions.h5 \
    --wm /workspace/ckpts/dinowm_noprop_cube \
    --init-value /workspace/metrics/dinopool_td_24k.pt \
    --horizon 5 --steps $STEPS --batch 32 --n-step 50 --amax 1.6 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed 0 --iters 8 \
    --ckpt-every 1000 $extra \
    --out "/workspace/actors/${tag}.pt" \
    --out-value "/workspace/metrics/${tag}_value.pt" \
    > "$L/${tag}.log" 2>&1 &
  log "  $tag -> GPU $g $extra"
  # let it claim the card before polling again, and surface argparse deaths
  sleep 90
  if grep -qE "error:|Traceback" "$L/${tag}.log" 2>/dev/null; then
    log "  FATAL $tag: $(grep -E 'error:|Error' "$L/${tag}.log" | tail -1)"
  fi
done

log "  all cells dispatched; waiting for the last trainers"
while [ -n "$(free_gpu)" ] || true; do
  busy=0
  for g in 1 2 3; do
    mb=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -n "$mb" ] && [ "$mb" -ge "$FREE_MB" ] && busy=$((busy+1))
  done
  [ "$busy" -eq 0 ] && break
  sleep 180
done
log "SCHED_DONE"
