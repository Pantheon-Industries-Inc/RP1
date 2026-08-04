#!/bin/bash
# dinowmnp's Latent+CEM bar, at full token width, no cache and no compression.
#
# WHY THIS RUNS BEFORE ANYTHING ELSE ON THIS BASE. The DINO-WM latent is
# 75264-d -- DinoWMTokens keeps all 196 DINOv2 patch tokens unpooled (196 x 384).
# A full-dataset cache would be 2.01M x 75264 x 4B = 605 GB against 389 GB of
# free disk, which is what actually killed run_dinowm.sh's cache stage (the log
# ended right after checkpoint load with no traceback; I guessed host OOM and
# checked free memory AFTER the crash, when it had already been released -- the
# latent width was the thing to look at, and I did not).
#
# The BAR, however, needs no cache at all: CEM plans through the world model
# directly and the L2 window cost is parameter-free. So it can be measured at
# the native 75264-d width, exactly as lejepa/pldm bars are measured at their
# native 192-d. That makes this the faithful, compression-free reference the
# later LIP number has to be judged against.
#
# The value and LIP DO need the cache, hence --compress rp1024 (8 GB), which the
# cache_latents docstring documents for this exact checkpoint and prefers over
# mean-pooling because a random projection "preserves the full-vector L2
# geometry CEM already exploits". Mean-pooling would have silently destroyed the
# geometry this very bar depends on. That stage needs --compress support added
# to train_window.py and is deliberately separate from this measurement.
#
# The L2 stub is generated at 3 x 75264 = 225792 so the eval hook infers a
# 3-frame window, matching every other base's l2window3.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/dinowmnp_reacher
L2=/workspace/metrics/l2window3_dino.pt
SUM=/workspace/results/summary_dinobar.csv; touch "$SUM"
L=/workspace/logs/dinobar; mkdir -p "$L"
REP="42 43 44 45 46 47"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][dinobar] $*"; }

[ -f "$L2" ] || python3 -c "
import torch, collections
D = 75264
torch.save({'learner':'l2','latent_dim':3*D,
            'arch':{'window_frames':3,'window_lag':5,'unlearned':True},
            'state_dict':collections.OrderedDict()}, '$L2')
print('wrote $L2 at 3x%d = %d' % (D, 3*D))"
log "L2 stub ready at full token width"

ev(){ local gpu=$1 nm=$2 seed=$3
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 14400 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=cem solver.n_steps=10 solver.batch_size=10 \
    "+metric=$L2" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && log "  !! ${nm} no score: $(tail -2 "$L/${nm}.log" | tr '\n' ' ' | cut -c1-90)"
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
  log "  ${nm}: @0.05 ${h:-FAIL} | @0.1 ${t10:-FAIL}"
}
meanof(){ local key=$1; local t=0 n=0 e
  for s in $REP; do
    e=$(grep "^dbar_s${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"; }

log "Latent+CEM on dinowmnp, 6 reporting seeds, 2-wide (75264-d rollouts are heavy)"
( for s in 42 43 44; do ev 4 "dbar_s${s}" $s; done ) &
( for s in 45 46 47; do ev 5 "dbar_s${s}" $s; done ) &
wait

n=$(grep -c "^dbar_s.*held=[0-9]" "$SUM")
log "=================================================================="
log "Latent+CEM (dinowmnp): @0.05 $(meanof held) | @0.1 $(meanof held10)   (${n}/6 scored)"
log "  other bases' bars: lejepa 44.7/84.3, pldm 39.3/78.3"
log "  prior campaign anchor for this base: 65.3 latched at the paper config"
log "DINOBAR_DONE"
