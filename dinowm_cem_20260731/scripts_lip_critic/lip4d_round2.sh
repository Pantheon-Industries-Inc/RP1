#!/bin/bash
# LIPv4-dino sweep, round 2: the actor-lr axis + a training-length cell.
#
# Round 1 swept amax {1.2,1.6,2.0} at actor-lr 3e-4. This completes the grid the
# original request asked for (amax x lr) and settles the one remaining deviation
# from LeWM's canonical recipe (3000 steps vs 6000).
#
# Gated: waits for round 1 trainers to exit, then reads summary_lip4d.csv and
# picks the best amax from the DATA (highest mean across whatever draws exist at
# the deepest checkpoint), rather than hardcoding my guess. Cells:
#   lr 1e-4   at the winning amax   (does a slower actor help? LIP is lr-sensitive)
#   lr 1e-3   at the winning amax   (does a faster one?)
#   6000 steps at winning amax, lr 3e-4  (round 1 gained +2..+6 per 1000 steps;
#            this tests whether training length is the binding constraint, and it
#            is the last uncontrolled difference vs LeWM's 87.6 recipe)
#
# Also relaunches the eval daemon, which exits when round 1's trainers die.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1
CODE=/workspace/code/stable-worldmodel; L=/workspace/logs; R=/workspace/results
SUM=$R/summary_lip4d.csv
mkdir -p "$L" /workspace/actors /workspace/metrics
cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_lip4d_r2.log"; }

log "=== round 2 gate: waiting for round 1 trainers to exit (12 h max) ==="
for i in $(seq 1 720); do
  pgrep -f "train_lip_ac_din[o]" >/dev/null || break
  sleep 60
done
if pgrep -f "train_lip_ac_din[o]" >/dev/null; then
  log "FATAL round 1 still running after 12 h"; exit 1
fi
log "  round 1 done"

# ---- pick the winning amax from the data (deepest checkpoint, mean over draws)
BEST=$(python3 - "$SUM" <<'PY'
import sys, re, collections
rows = collections.defaultdict(list)
try:
    for line in open(sys.argv[1]):
        line = line.strip()
        if not line:
            continue
        nm, sr = line.split(',')[0], line.split(',')[1]
        if sr == 'FAIL':
            continue
        m = re.match(r'lip4d_a(\d+)_s(\d+)_e(\d+)$', nm)
        if m:
            rows[(int(m.group(2)), m.group(1))].append(float(sr))
except FileNotFoundError:
    pass
if not rows:
    print('16'); raise SystemExit
deepest = max(k[0] for k in rows)
cand = {a: sum(v) / len(v) for (s, a), v in rows.items() if s == deepest}
print(max(cand, key=cand.get))
PY
)
case "$BEST" in (12) A=1.2;; (16) A=1.6;; (20) A=2.0;; (*) A=1.6; BEST=16;; esac
log "  best amax from data: $A (tag a$BEST)"

launch(){ # gpu tag steps lr
  local g=$1 tag=$2 st=$3 lr=$4
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
    --actor-lr "$lr" --actor-lr-final "$(python3 -c "print(f'{float('$lr')/10:.1e}')")" \
    --arch v4 --seed 0 --ckpt-every 1000 \
    --out "/workspace/actors/${tag}.pt" \
    --out-value "/workspace/metrics/${tag}_value.pt" \
    > "$L/${tag}.log" 2>&1 &
  log "  $tag launched on GPU $g (amax $A, lr $lr, steps $st)"
}

launch 1 "lip4d_a${BEST}_lr1e4"  3000 1e-4
launch 2 "lip4d_a${BEST}_lr1e3"  3000 1e-3
launch 3 "lip4d_a${BEST}_s6k"    6000 3e-4

sleep 20
if ! pgrep -f "lip4d_evald" >/dev/null; then
  (nohup bash /workspace/lip4d_evald.sh > "$L/nohup_lip4d_evald_r2.log" 2>&1 &)
  log "  eval daemon relaunched"
fi
log "ROUND2_LAUNCHED"
