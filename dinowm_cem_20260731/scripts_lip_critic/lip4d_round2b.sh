#!/bin/bash
# LIPv4-dino round 2 (relaunch). Fixes three faults in the first attempt:
#   1. --actor-lr-final was computed with a nested-quoted python one-liner that
#      collapsed to '' and killed all three cells instantly. lr pairs are now
#      HARDCODED -- no computation, no quoting.
#   2. amax is pinned to 1.6, the ACTUAL round-1 winner (mean 72.0 over 3 draws
#      vs a20 70.7, a12 67.3). The gated picker had read the CSV before the eval
#      daemon's final s3000 sweep landed and chose a20 on incomplete data.
#   3. the eval daemon is started unconditionally here; last time the "only if
#      none alive" check saw the round-1 daemon still finishing its final sweep,
#      skipped the relaunch, and then that daemon exited -- leaving nothing to
#      score round 2.
#
# Grid completed by this round: amax (round 1) x actor-lr (here) x length (s6k).
# s6k closes the last uncontrolled gap vs LeWM's 87.6 recipe (6000 steps; round
# 1 ran 3000 and gained +2..+6 per 1000).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1
CODE=/workspace/code/stable-worldmodel; L=/workspace/logs
mkdir -p "$L" /workspace/actors /workspace/metrics
cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_lip4d_r2b.log"; }

A=1.6                     # round-1 winner, from the data

launch(){ # gpu tag steps lr lr_final
  local g=$1 tag=$2 st=$3 lr=$4 lrf=$5
  [ -f "/workspace/actors/${tag}_s${st}.pt" ] && { log "  $tag complete, skip"; return 0; }
  nohup env CUDA_VISIBLE_DEVICES=$g python3 scripts/plan/train_lip_ac_dino.py \
    --cache /workspace/caches/dinopool_tr8000_fs5.pt \
    --cache-td /workspace/caches/dinopool_tr8000_fs1.pt \
    --dataset /root/datasets/ogb_cube_single/ogb_cube_single.lance \
    --h5 /workspace/datasets/expert_actions.h5 \
    --wm /workspace/ckpts/dinowm_noprop_cube \
    --init-value /workspace/metrics/dinopool_td_24k.pt \
    --horizon 5 --iters 8 --steps "$st" --batch 32 --n-step 50 --amax "$A" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr "$lr" --actor-lr-final "$lrf" \
    --arch v4 --seed 0 --ckpt-every 1000 \
    --out "/workspace/actors/${tag}.pt" \
    --out-value "/workspace/metrics/${tag}_value.pt" \
    > "$L/${tag}.log" 2>&1 &
  log "  $tag -> GPU $g (amax $A, lr $lr -> $lrf, steps $st)"
}

log "=== round 2b: actor-lr axis + length cell at amax $A ==="
launch 1 lip4d_a16_lr1e4 3000 1e-4 1e-5
launch 2 lip4d_a16_lr1e3 3000 1e-3 1e-4
launch 3 lip4d_a16_s6k   6000 3e-4 3e-5

# verify they SURVIVED argparse (last round died in <1 s), before trusting the run
sleep 45
alive=$(pgrep -fc "train_lip_ac_din[o]" || true)
log "  trainers alive after 45 s: ${alive:-0} (expect 3)"
for t in lip4d_a16_lr1e4 lip4d_a16_lr1e3 lip4d_a16_s6k; do
  if grep -qE "error:|Traceback" "$L/$t.log" 2>/dev/null; then
    log "  FATAL $t: $(grep -E 'error:|Error' "$L/$t.log" | tail -1)"
  fi
done

(nohup bash /workspace/lip4d_evald.sh > "$L/nohup_lip4d_evald_r2b.log" 2>&1 &)
log "  eval daemon started (unconditional)"
log "ROUND2B_LAUNCHED"
