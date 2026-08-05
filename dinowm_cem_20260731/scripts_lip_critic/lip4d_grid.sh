#!/bin/bash
# LIPv4-dino round 3: expand x replay x critic-lr x iters, work-queue scheduled.
#
# Rounds 1-2 covered amax {1.2,1.6,2.0} and actor-lr {1e-4,3e-4,1e-3} + a 6k
# length cell. This round adds the two feedback-loop features (now that replay is
# ported to token space) plus the critic axes, OFAT from the round-1 winner
# (amax 1.6, actor-lr 3e-4) with two interaction cells.
#
# Why OFAT and not full factorial: a full 3x3x3x3 is 81 cells x ~5 h / 4 GPUs =
# 100+ h. OFAT + targeted interactions is the methodology the PLDM/Dyna campaigns
# used (26-arm OFAT screen, then an adaptive 2^4 factorial on what interacted).
#
# SCREEN LENGTH: 2000 steps, not 3000. Round 1 measured s2000 -> s3000 worth only
# +2..+4, so 2000 screens the same ordering at 2/3 the cost. Winners get extended.
#
# EXPECTATION TO SET, from the trainer's own docstring: expand "lets the pair
# co-exploit WM errors, so compare against expand-weight 0 before trusting" --
# and on THIS substrate we measured teacher(imagined,goal) agreement decaying to
# ~0 by h=5, so expand may well distill unreliable terminal evaluations into the
# values of real states. That is exactly why it is being MEASURED, with the
# round-1 amax-1.6 cell (expand 0, replay 0, mean 72.0) as the control.
#
# Work-queue: 3 GPU slots (0 stays free for the eval daemon), next cell starts the
# moment a slot frees -- no wave barriers.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1
CODE=/workspace/code/stable-worldmodel; L=/workspace/logs
mkdir -p "$L" /workspace/actors /workspace/metrics
cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_lip4d_grid.log"; }

STEPS=2000
# tag|extra args   (amax 1.6, actor-lr 3e-4 unless overridden)
CELLS=(
  "exp0.1|--expand-weight 0.1"
  "exp0.5|--expand-weight 0.5"
  "exp1.0|--expand-weight 1.0"
  "rep0.25|--replay-prob 0.25"
  "rep0.5|--replay-prob 0.5"
  "exp0.5rep0.25|--expand-weight 0.5 --replay-prob 0.25"
  "exp0.1rep0.5|--expand-weight 0.1 --replay-prob 0.5"
  "iters16|--iters 16"
  "clr3e4|--critic-lr 3e-4 --critic-lr-final 3e-5"
  "ratio2|--critic-ratio 2"
)

log "=== round 3 gate: waiting for rounds 1-2 trainers to exit (14 h max) ==="
for i in $(seq 1 840); do
  pgrep -f "train_lip_ac_din[o]" >/dev/null || break
  sleep 60
done
pgrep -f "train_lip_ac_din[o]" >/dev/null && { log "FATAL prior round still running"; exit 1; }
log "  prior rounds done; ${#CELLS[@]} cells over 3 slots, $STEPS steps each"

declare -A SLOT_PID
free_slot(){
  while true; do
    for g in 1 2 3; do
      p=${SLOT_PID[$g]:-}
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then echo "$g"; return; fi
    done
    sleep 60
  done
}

for cell in "${CELLS[@]}"; do
  tag="lip4d_${cell%%|*}"; extra="${cell#*|}"
  if [ -f "/workspace/actors/${tag}_s${STEPS}.pt" ]; then log "  $tag done, skip"; continue; fi
  g=$(free_slot)
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
  SLOT_PID[$g]=$!
  log "  $tag -> GPU $g (pid ${SLOT_PID[$g]}) $extra"
  sleep 40
  if grep -qE "error:|Traceback" "$L/${tag}.log" 2>/dev/null; then
    log "  FATAL $tag died: $(grep -E 'error:|Error' "$L/${tag}.log" | tail -1)"
  fi
done

for g in 1 2 3; do p=${SLOT_PID[$g]:-}; [ -n "$p" ] && wait "$p" 2>/dev/null; done
log "GRID_DONE"
