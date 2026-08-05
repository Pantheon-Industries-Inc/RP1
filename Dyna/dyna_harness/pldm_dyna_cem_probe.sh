#!/bin/bash
# DISCRIMINATING TEST: does Dyna's +8 on PLDM come from better GRADIENTS or
# just a generally better world model?
#
# LIP backprops through the WM (needs a trustworthy Jacobian); CEM only ranks
# sampled rollouts (needs correct ordering). Ranking is a far weaker
# requirement, which is the proposed reason LIP gains ~10 over TD+CEM on LeWM
# and ~0 on frozen PLDM. Dyna fine-tuning lifted PLDM-LIP 75.1 -> 83.1.
#
# Prediction, quantitative and falsifiable:
#   gradient story  => LIP gains MUCH more from Dyna than CEM does. CEM's
#                      rankings were already adequate on frozen PLDM (73.3),
#                      so it should move little.
#   general-quality => CEM and TD+CEM gain comparably to LIP, and the
#                      gradient explanation is unnecessary.
#
# 6 cells (~15 min). TD+CEM uses the FINE-TUNED TD teacher, so the comparison
# is matched: every component is trained on the same WM's latents.
# Waits for the POST seed replication to finish (evals must not overlap
# training -- that rule has not been revalidated).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_cemprobe.log
WMD=/workspace/models/dyna_pldm_5050
DTD=/workspace/metrics/dynapldm_TD.pt
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"
mkdir -p "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== DYNA-WM CEM PROBE (pid $$) ==="
ls "$WMD"/*.pt >/dev/null 2>&1 || die "fine-tuned WM missing"
[ -f "$DTD" ] || die "fine-tuned TD teacher missing"

log "P0: waiting for the POST seed replication"
T0=$(date +%s)
until grep -q "PLDM_POSTSEED_DONE" "$L/driver_postseed.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 28800 ] && { log "WARN: not done in 8h, proceeding once idle"; break; }
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 60; done
log "P0: box quiet"

ev(){ # gpu name extra...
  local gpu=$1 nm=$2 d; shift 2
  d=$(echo "$nm" | grep -oE "e4[234]$" | tr -d e)
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$WMD" "$@" output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
}

log "P1: CEM and TD+CEM on the Dyna-fine-tuned WM, 6 cells"
g=0
for d in $DRAWS; do ev "$g" "dynawm_cem_e${d}" solver=cem & g=$((g+1)); done
for d in $DRAWS; do ev "$g" "dynawm_cemtd_e${d}" solver=cem "+metric=$DTD" & g=$((g+1)); done
wait

python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
def m3(p):
    vs = [rows.get(f"{p}_e{d}") for d in (42,43,44)]
    return None if any(v is None for v in vs) else mean(vs)
def seeds(p, ns):
    out = []
    for s in ns:
        vs = [rows.get(f"{p}_s{s}_e{d}") for d in (42,43,44)]
        if all(v is not None for v in vs): out.append(mean(vs))
    return out
print("=== GRADIENT vs GENERAL-QUALITY PROBE (PLDM, held-out, EGL) ===")
frozen = {"latent+CEM": m3("ref_cem"), "TD+CEM": m3("ref_cemtd")}
dyna   = {"latent+CEM": m3("dynawm_cem"), "TD+CEM": m3("dynawm_cemtd")}
lip_f = seeds("grid_mw01_a45_lr3e-4", range(20))
lip_d = seeds("dynapldm", range(20))
if lip_f: frozen["LIP"] = mean(lip_f)
if lip_d: dyna["LIP"] = mean(lip_d)
print(f"  {'planner':12s} {'frozen':>8s} {'Dyna-WM':>9s} {'gain':>7s}")
gains = {}
for k in ("latent+CEM", "TD+CEM", "LIP"):
    a, b = frozen.get(k), dyna.get(k)
    if a is not None and b is not None:
        gains[k] = b - a
        n = f" (n={len(lip_d)})" if k == "LIP" else ""
        print(f"  {k:12s} {a:8.1f} {b:9.1f} {b-a:+7.1f}{n}")
if len(gains) == 3:
    search = max(gains["latent+CEM"], gains["TD+CEM"])
    print(f"  LIP gain {gains['LIP']:+.1f} vs best search-planner gain {search:+.1f}"
          f" -> ratio {gains['LIP']/search:.2f}x" if search > 0 else
          f"  LIP gain {gains['LIP']:+.1f}; search planners did not gain")
    print("  reading: LIP >> CEM => Dyna repaired the WM's GRADIENTS, which is what")
    print("  LIP uniquely needs; comparable gains => generally better WM, and the")
    print("  gradient explanation for the cross-base asymmetry is unnecessary.")
print("PLDM_CEMPROBE_DONE")
PY
log "done"
