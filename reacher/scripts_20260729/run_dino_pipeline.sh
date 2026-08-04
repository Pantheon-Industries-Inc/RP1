#!/bin/bash
# DINO-WM, uncompressed, 3-frame: value -> RLP -> cards.
#
# Everything at the native 75,264-d patch-token latent. No random projection and
# no mean-pooling anywhere, so the learned value and the parameter-free
# Latent+CEM bar are fit and scored on exactly the same representation -- the
# arms then differ only in the cost, which is the whole point of the comparison.
#
# Two things had to be fixed to make this possible, both already verified:
#   keep_pred_frames 2 -> 3 in the checkpoint config, so predicted_emb carries
#     the 3 frames the window hook indexes. Smoked: the hook reports
#     "pred (5, 300, 3, 75264)" where it previously raised
#     "index -3 out of bounds for dimension 1 with size 2".
#   train_window's window is now GATHERED PER BATCH rather than materialized.
#     torch.cat([Z[i] for i in idxs]) costs 7x the base cache (F indexed copies
#     held alongside the output) = ~4.2 TB at this width against 1.9 TB of RAM.
#     The lazy view costs base + one batch = 605 GB + 925 MB, verified
#     bit-identical to the materialized path, and keeps all 2.01M rows.
#
# COST NOTE. Each process must torch.load the 605 GB fs1 cache from a 204 MB/s
# volume: ~50 min before a single step runs. So this deliberately trains ONE RLP
# seed first to validate the path end to end, rather than paying that load three
# times in parallel and discovering a shape error at the end of all of them.
#
# RLP hypers are NOT swept for this base. They are the lejepa settings
# (amax 2.2, actor-lr 3e-4, mean-weight 0.3) plus the shared recipe. Any number
# out of this run is a first read, not a tuned result.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
TRM=/workspace/swm_cem/scripts/trm
CANON=/workspace/datasets_canon/reacher.h5
WM=/workspace/swm_home/checkpoints/dinowmnp_reacher
C1=/workspace/caches/canon_dinowmnp_fs1.pt
C5=/workspace/caches/canon_dinowmnp_fs5.pt
W3=/workspace/metrics/window3_dinowmnp_e005_g098.pt
L2=/workspace/metrics/l2window3_dino.pt
SUM=/workspace/results/summary_dino.csv; touch "$SUM"
L=/workspace/logs/dino; mkdir -p "$L"
G=0.98
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][dino] $*"; }
die(){ log "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------- 0. gate on the cache
if ! grep -q "CACHE_EXIT=0" /workspace/logs/cache_dino_full.log 2>/dev/null; then
  log "waiting for the 605 GB cache (ceiling 4 h)"
  w=0
  while ! grep -q "CACHE_EXIT=0" /workspace/logs/cache_dino_full.log 2>/dev/null; do
    sleep 120; w=$((w + 120))
    [ $((w % 1800)) -eq 0 ] && log "  still waiting, ${w}s, file $(du -sh $C1 2>/dev/null | cut -f1)"
    grep -q "CACHE_EXIT=[1-9]" /workspace/logs/cache_dino_full.log 2>/dev/null && die "cache failed"
    [ "$w" -ge 14400 ] && die "cache never finished"
  done
fi
log "cache present: $(du -sh $C1 | cut -f1)"
python3 - <<'PYEOF' || exit 1
import torch
c = torch.load("/workspace/caches/canon_dinowmnp_fs1.pt", map_location="cpu",
               weights_only=False, mmap=True)
z = c["z"]
print(f"[verify] z {tuple(z.shape)} {z.dtype}  rows={len(c['episode_idx'])}", flush=True)
assert z.shape[1] == 75264, f"expected full width 75264, got {z.shape[1]}"
assert z.shape[0] == 2010000, f"expected all rows, got {z.shape[0]}"
print("[verify] full width, all rows, uncompressed", flush=True)
PYEOF

# ---------------------------------------------------------------- 1. fs5 (actor contexts)
if [ ! -f "$C5" ]; then
  log "fs5 subsample (stride 5, one latent per action block)"
  CUDA_VISIBLE_DEVICES=1 python3 "$TRM/subsample_cache.py" --in "$C1" --out "$C5" \
    --frameskip 5 > "$L/fs5.log" 2>&1 || die "fs5 failed (see fs5.log)"
fi
log "fs5: $(du -sh $C5 | cut -f1)"

# ---------------------------------------------------------------- 2. 3-frame value
if [ ! -f "$W3" ]; then
  log "3-frame window value at 3x75264 = 225792 input (lazy gather, ~50 min cache load first)"
  CUDA_VISIBLE_DEVICES=1 python3 /workspace/train_window.py \
    --cache "$C1" --lag 5 --frames 3 --expectile 0.05 --n-step 50 \
    --gamma "$G" --steps 6000 --seed 0 --out "$W3" > "$L/w3.log" 2>&1 \
    || die "value failed (see w3.log)"
fi
log "value: $(grep -oE 'final loss=[0-9.]+' "$L/w3.log" | tail -1)"
grep -oE "lazy window:.*" "$L/w3.log" | tail -1
grep -oE "\[tiled-goal diag\].*" "$L/w3.log" | tail -1
python3 -c "
import torch; b=torch.load('$W3',map_location='cpu',weights_only=False)
print('[verify] value latent_dim', b['latent_dim'], 'frames', b['arch']['window_frames'])
assert b['latent_dim']==225792, b['latent_dim']" || die "value has the wrong width"

# ---------------------------------------------------------------- 3. one RLP seed
A0=/workspace/actors/rlp_dinowmnp_s0.pt
if [ ! -f "$A0" ]; then
  log "RLP seed 0 (validation run; hypers are lejepa's, unswept for this base)"
  CUDA_VISIBLE_DEVICES=2 timeout 43200 python3 "$PLAN/train_lip_ac.py" \
    --cache "$C5" --cache-td "$C1" --h5 "$CANON" --wm "$WM" --pad-context \
    --init-value "$W3" \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --gamma "$G" --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight 0.3 --amax 2.2 \
    --replay-prob 0.5 --expand-weight 0 --steps 1000 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --seed 0 \
    --out "$A0" --out-value "/workspace/metrics/rlp_dinowmnp_s0_value.pt" \
    > "$L/rlp_s0.log" 2>&1 || die "RLP failed (see rlp_s0.log)"
fi
log "RLP seed 0 trained; $(grep -oE '\[vframes\].*' "$L/rlp_s0.log" | tail -1)"

# ---------------------------------------------------------------- 4. cards
ev(){ local gpu=$1 nm=$2 seed=$3; shift 3
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 21600 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=5 \
    "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && log "  !! ${nm} no score: $(tail -1 "$L/${nm}.log" | cut -c1-70)"
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
  log "  ${nm}: @0.05 ${h:-FAIL} | @0.1 ${t10:-FAIL}"
}
mean(){ grep "^${1}" "$SUM" | grep -oE "${2}=[0-9.]+" | cut -d= -f2 \
        | awk '{t+=$1;n++}END{printf "%.1f (n=%d)", (n?t/n:0), n}'; }

log "cards on episodes 8000:10000 (3 seeds; full-width rollouts are expensive)"
for s in 42 43 44; do ev 1 "dn_l2cem_s${s}"  $s solver=cem solver.n_steps=10 "+metric=$L2"; done
for s in 42 43 44; do ev 1 "dn_tdcem_s${s}"  $s solver=cem solver.n_steps=10 "+metric=$W3"; done
for s in 42 43 44; do ev 2 "dn_rlp_s${s}"    $s solver=lip solver.actor_path="$A0" solver.rollout_compat=false; done

log "==================== DINO-WM (3-frame, uncompressed) ===================="
log "  Latent + CEM : @0.1 $(mean dn_l2cem_s held10)  | @0.05 $(mean dn_l2cem_s held)"
log "  Value  + CEM : @0.1 $(mean dn_tdcem_s held10)  | @0.05 $(mean dn_tdcem_s held)"
log "  RLP (1 seed) : @0.1 $(mean dn_rlp_s held10)  | @0.05 $(mean dn_rlp_s held)"
log "  reference bars: lejepa 84.3 / pldm 78.3 @0.1 (both 3-frame, 224 px)"
log "  CAVEAT: this base renders at 196 px, not 224 -- not pixel-identical to those"
log "DINO_PIPELINE_DONE"
