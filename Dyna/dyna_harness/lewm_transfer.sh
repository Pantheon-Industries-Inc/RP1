#!/bin/bash
# DOES IT TRANSFER? Test the PLDM factorial's winning configuration on LeWM.
#
# A knob that helps on a weak base may be repairing a weakness the strong base
# does not have -- in which case it is a PLDM fix, not a LIP improvement. The
# only way to tell them apart is to run the same configuration on LeWM, where
# the recipe is already at 87.8 and has resisted every optimisation arm tried
# (iters16 = PRE exactly, s12k +0.9, alr -2.2).
#
# LeWM assets died with the AP-JP-1 volume, so this rebuilds the minimum: v2WM
# (uploaded), its ViT keys inverted to the pod's transformers 4.49 layout, fs1
# cache -> train-split filter -> fs5, and a TD teacher. ~30 min.
#
# ANCHOR: LeWM PRE at amax 1.6 was 87.8 (3 seeds) / 87.1 (6 seeds). The rebuilt
# baseline arm must land near that or the LeWM side is not trustworthy and no
# transfer claim can be made from it.
#
# Note amax: LeWM's optimum is 1.6, PLDM's is 4.5. The transfer test keeps each
# base at ITS OWN amax and varies only the factorial's winning knobs -- porting
# PLDM's amax to LeWM would confound the knob with a 7-point regression.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_lewmtx.log
V2WM=/workspace/models/v2WM
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
LF1F=/workspace/caches/v2_full_fs1.pt
LF1=/workspace/caches/v2_tr8000_fs1.pt
LF5=/workspace/caches/v2_tr8000_fs5.pt
LTD=/workspace/metrics/v2_tr8000_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
LEWM_AMAX=1.6; NGPU=8
mkdir -p "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== LeWM TRANSFER TEST (pid $$) ==="
log "P0: waiting for the PLDM factorial"
T0=$(date +%s)
until grep -q "PLDM_FACTORIAL_DONE" "$L/driver_factorial.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && die "factorial never finished in 12h"
  sleep 180
done
while pgrep -f "train_lip_a[c]|train_metri[c]|eval_w[m]" >/dev/null; do sleep 60; done
log "P0: factorial done, box quiet"

# ---- winning cell = lowest E_final in the factorial
WIN=$(python3 - "$L" <<'PY'
import sys, re, glob, os
L = sys.argv[1]; best = (None, 1e9)
for f in glob.glob(f"{L}/fx_*_s0.log"):
    tag = os.path.basename(f)[3:-7]
    try: t = open(f).read()
    except OSError: continue
    m = re.findall(r"E_final ([0-9.]+)", t)
    if m and float(m[-1]) < best[1]: best = (tag, float(m[-1]))
print(best[0] or "")
PY
)
[ -n "$WIN" ] || die "could not identify the factorial winner"
log "P1: factorial winner = $WIN"
# recover its flags from the arm table used by the factorial
WIN_FLAGS=$(python3 - "$WIN" <<'PY'
import sys
FLAG = {
 "expand01":"--expand-weight 0.1", "expand10":"--expand-weight 1.0",
 "replay25":"--replay-prob 0.25",  "replay50":"--replay-prob 0.5",
 "mw003":"--mean-weight 0.03",     "mw000":"--mean-weight 0.0",
 "itemb":"--iter-mode emb",        "headprec":"--head-mode precond",
 "wide":"--width 512",             "deep":"--layers 4",
 "sdim512":"--s-dim 512",          "rec1024":"--rec-hidden 1024",
 "zeroinit":"--zero-init",         "gdinit":"--gd-init 0.1",
 "featnorm":"--feat-norm",         "s0z0":"--s0-mode z0",
 "feedend":"--feed end",           "feedtraj":"--feed traj",
 "pcross0":"--p-cross 0.0",        "pcross6":"--p-cross 0.6",
 "maxd20":"--max-delta 20",        "batch256":"--batch 256",
}
tag = sys.argv[1].replace("fx_", "")
print(" ".join(FLAG[p] for p in tag.split(".") if p in FLAG))
PY
)
log "P1: winning flags: ${WIN_FLAGS:-<none>}"
[ -n "$WIN_FLAGS" ] || { log "winner is the all-off cell -- nothing to transfer"; exit 0; }

# ------------------------------------------------- P2 rebuild LeWM assets
[ -f "$V2WM/weights_epoch_22.pt" ] || die "v2WM not uploaded"
if [ ! -f /workspace/.v2wm_keys_ok ]; then
  log "P2: inverting v2WM ViT keys to the pod's transformers 4.49 layout"
  python3 /workspace/invert_pldm_keys.py "$V2WM" weights_epoch_22.pt \
    > "$L/lewmtx_invert.log" 2>&1
  grep -qE "INVERT_OK|nothing to invert" "$L/lewmtx_invert.log" \
    || { tail -6 "$L/lewmtx_invert.log"; die "v2WM key inversion failed"; }
  touch /workspace/.v2wm_keys_ok
fi
if [ ! -f "$LF1" ]; then
  log "P2: caching fs1 under v2WM (~25 min)"
  [ -f "$LF1F" ] || CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$TRM/cache_latents.py" \
    --wm "$V2WM" --dataset "$EXPERT" --out "$LF1F" --state-key privileged_block_0_pos \
    > "$L/lewmtx_cache.log" 2>&1 || { tail -6 "$L/lewmtx_cache.log"; die "cache failed"; }
  python3 /workspace/filter_cache_eprange.py "$LF1F" "$LF1" --lo 0 --hi "$EPHI" \
    > "$L/lewmtx_filter.log" 2>&1 || die "filter failed"
  grep -q FILTER_CACHE_DONE "$L/lewmtx_filter.log" || die "filter incomplete"
fi
[ -f "$LF5" ] || python3 "$TRM/subsample_cache.py" --in "$LF1" --out "$LF5" --frameskip 5 \
  > "$L/lewmtx_fs5.log" 2>&1 || die "fs5 failed"
[ -f "$LTD" ] || { log "P2: TD teacher on v2WM";
  CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_metric.py" \
    --cache "$LF1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
    --steps 6000 --seed 0 --out "$LTD" > "$L/lewmtx_td.log" 2>&1 || die "TD failed"; }
log "P2: LeWM assets ready"

# ------------------- P3 baseline + winner on LeWM, 3 seeds, LeWM's own amax
log "P3: LeWM arms at amax $LEWM_AMAX -- baseline (anchor 87.8) and winner"
g=0
for arm in lewmbase lewmwin; do
  case "$arm" in lewmbase) extra="";; *) extra="$WIN_FLAGS";; esac
  for s in $SEEDS; do
    out=/workspace/actors/lip4_${arm}_s${s}.pt
    [ -f "$out" ] && continue
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$g timeout 43200 python3 "$P/train_lip_ac.py" \
      --cache "$LF5" --cache-td "$LF1" --h5 "$AH5" --wm "$V2WM" --init-value "$LTD" \
      --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$LEWM_AMAX" \
      --actor-lr 3e-4 --actor-lr-final 3e-5 \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --arch v4 --seed "$s" --freeze-critic-frac 0.05 $extra \
      --out "$out" --out-value "${out%.pt}_value.pt" \
      > "$L/lewmtx_${arm}_s${s}.log" 2>&1 &
    g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
  done
done
wait
log "P3: trains done"

ev(){ # gpu arm seed draw
  local gpu=$1 arm=$2 s=$3 d=$4 nm="${arm}_s${s}_e${d}"
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$V2WM" solver=lip "solver.actor_path=/workspace/actors/lip4_${arm}_s${s}.pt" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
}
log "P4: 18 eval cells"
g=0
for arm in lewmbase lewmwin; do for s in $SEEDS; do for d in $DRAWS; do
  ev "$g" "$arm" "$s" "$d" & g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done; done
wait

WIN="$WIN" python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import os, sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
def arm(a):
    out = []
    for s in (0,1,2):
        vs = [rows.get(f"{a}_s{s}_e{d}") for d in (42,43,44)]
        if all(v is not None for v in vs): out.append(mean(vs))
    return out
b, w = arm("lewmbase"), arm("lewmwin")
print(f"=== LeWM TRANSFER of PLDM winner '{os.environ['WIN']}' (held-out, EGL, amax 1.6) ===")
for nm, xs in (("LeWM baseline", b), ("LeWM + winner", w)):
    if xs: print(f"  {nm:16s} {' '.join(f'{x:5.1f}' for x in xs)}   mean {mean(xs):.1f}")
if b:
    off = mean(b) - 87.8
    print(f"  ANCHOR: rebuilt baseline {mean(b):.1f} vs historical 87.8 ({off:+.1f})"
          + ("  OK" if abs(off) <= 3 else "  <<< OFF -- LeWM side not trustworthy"))
if b and w and len(b) == 3 and len(w) == 3:
    d = [x-y for x, y in zip(w, b)]; m = mean(d)
    sd = math.sqrt(sum((x-m)**2 for x in d)/2)
    t = m/(sd/math.sqrt(3)) if sd > 0 else float('inf')
    print(f"  transfer delta {m:+.2f} sd {sd:.2f} t {t:.2f} (df=2)")
    print("  reading: helps BOTH bases => a genuine LIP improvement; helps PLDM")
    print("  only => it repairs a weak-base deficiency the strong base lacks,")
    print("  which is a finding about PLDM rather than about the method.")
print("LEWM_TRANSFER_DONE")
PY
log "done"
