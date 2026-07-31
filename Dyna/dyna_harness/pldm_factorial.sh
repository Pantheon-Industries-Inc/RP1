#!/bin/bash
# ADAPTIVE 2^4 FACTORIAL over whatever the OFAT screen found to matter.
#
# WHY THIS SHAPE. A blind grid over ~30 knobs is impossible and, worse, mostly
# measures noise: with 12 cells the campaign already produced a phantom +2.0
# from one lucky seed. OFAT first asks "which factors move anything at all?"
# against a common static teacher; this then takes the FOUR biggest movers and
# runs a FULL FACTORIAL over them -- 2^4 = 16 cells, every combination.
#
# A full factorial is what actually exposes interplay: from 16 cells you get
# all four main effects AND all six pairwise interactions, each estimated from
# 8 cells rather than 1, so the estimates are averages and far steadier than
# any single-cell comparison. That is the specific thing OFAT cannot see -- if
# expand-weight only helps when replay is also on, OFAT scores both as null.
#
# Every arm keeps --arch v4 (user directive) and freezes the critic at 5%, so
# all 16 cells share one static teacher and E_final stays a common yardstick.
#
# Ranked on E_final, then the top cells get seeded success evals -- E_final is
# a filter, not a selector (LeWM's lowest-ever E_final planned worse).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_factorial.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"
AMAX=4.5; NGPU=8; NTOP=4
mkdir -p "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== PLDM 2^${NTOP} FACTORIAL (pid $$) ==="
log "P0: waiting for the OFAT screen"
T0=$(date +%s)
until grep -q "PLDM_OFAT_DONE" "$L/driver_ofat.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 28800 ] && die "OFAT never finished in 8h"
  sleep 180
done
while pgrep -f "train_lip_a[c]|train_metri[c]|eval_w[m]" >/dev/null; do sleep 60; done
log "P0: OFAT done, box quiet"

# ---- pick the NTOP biggest movers, and recover each one's flag from the
#      OFAT arm table so the factorial uses the exact same setting.
mapfile -t FACTORS < <(python3 - "$L" "$NTOP" <<'PY'
import sys, re, glob, os
L, ntop = sys.argv[1], int(sys.argv[2])
# tag -> the flag OFAT used (kept in sync with pldm_ofat_screen.sh)
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
def lastE(p):
    try: t = open(p).read()
    except OSError: return None
    m = re.findall(r"E_final ([0-9.]+)", t)
    return float(m[-1]) if m else None
vals = {}
for f in glob.glob(f"{L}/ofat_*_s0.log"):
    tag = os.path.basename(f)[5:-7]
    e = lastE(f)
    if e is not None: vals[tag] = e
base = vals.get("base")
if base is None: sys.exit(0)
# rank by |delta|, keep only knobs we know the flag for, one per knob family
fam = lambda t: re.sub(r"[0-9]+$", "", t)
seen, out = set(), []
for tag, e in sorted(vals.items(), key=lambda kv: -abs(kv[1]-base)):
    if tag == "base" or tag not in FLAG: continue
    if fam(tag) in seen: continue          # keep the better level per family
    seen.add(fam(tag)); out.append(f"{tag}|{FLAG[tag]}")
    if len(out) >= ntop: break
print("\n".join(out))
PY
)
[ "${#FACTORS[@]}" -ge 2 ] || die "OFAT produced fewer than 2 usable factors -- nothing to cross"
log "P1: factorial over ${#FACTORS[@]} factors:"
for f in "${FACTORS[@]}"; do log "    ${f%%|*}  (${f#*|})"; done

# ---- full factorial: bit i of the cell index = factor i on/off
NF=${#FACTORS[@]}; NCELL=$(( 1 << NF ))
log "P2: training $NCELL cells"
g=0
for ((c=0; c<NCELL; c++)); do
  tag=""; extra=""
  for ((i=0; i<NF; i++)); do
    if (( (c >> i) & 1 )); then
      nm="${FACTORS[$i]%%|*}"; fl="${FACTORS[$i]#*|}"
      tag="${tag}${nm}."; extra="$extra $fl"
    fi
  done
  [ -z "$tag" ] && tag="none."
  tag="fx_${tag%.}"
  out=/workspace/actors/lip4_pldm_${tag}_s0.pt
  [ -f "$out" ] && { log "  $tag present"; continue; }
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$g timeout 43200 python3 "$P/train_lip_ac.py" \
    --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
    --actor-lr 3e-4 --actor-lr-final 3e-5 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --arch v4 --seed 0 --freeze-critic-frac 0.05 $extra \
    --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/fx_${tag}_s0.log" 2>&1 &
  g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; log "  wave done"; }
done
wait
log "P2: factorial trained"

# ---- main effects + pairwise interactions from E_final
FACT_STR=$(printf '%s ' "${FACTORS[@]}")
FACT_STR="$FACT_STR" python3 - "$L" 2>&1 <<'PY' | tee -a "$DRV"
import os, sys, re, itertools
L = sys.argv[1]
facs = [f.split("|")[0] for f in os.environ["FACT_STR"].split()]
NF = len(facs)
def lastE(tag):
    try: t = open(f"{L}/fx_{tag}_s0.log").read()
    except OSError: return None
    m = re.findall(r"E_final ([0-9.]+)", t)
    return float(m[-1]) if m else None
cells = {}
for c in range(1 << NF):
    on = [facs[i] for i in range(NF) if (c >> i) & 1]
    tag = "fx_" + (".".join(on) if on else "none")
    e = lastE(tag)
    if e is not None: cells[frozenset(on)] = e
print(f"=== PLDM 2^{NF} FACTORIAL on E_final (common static teacher, seed 0) ===")
print(f"  cells measured: {len(cells)}/{1<<NF}")
if len(cells) < (1 << NF):
    print("  WARNING: incomplete -- effects below are estimated from what exists")
for k in sorted(cells, key=lambda s: cells[s]):
    print(f"    {'+'.join(sorted(k)) or '(none)':38s} {cells[k]:7.2f}")
print("  --- main effects (mean E_final with factor ON minus OFF) ---")
for f in facs:
    on  = [v for k, v in cells.items() if f in k]
    off = [v for k, v in cells.items() if f not in k]
    if on and off:
        print(f"    {f:12s} {sum(on)/len(on) - sum(off)/len(off):+7.2f}"
              f"   (n={len(on)}/{len(off)})")
print("  --- pairwise interactions (deviation from additive) ---")
for a, b in itertools.combinations(facs, 2):
    q = {}
    for ka, kb in itertools.product([0,1],[0,1]):
        vs = [v for k, v in cells.items()
              if (a in k) == bool(ka) and (b in k) == bool(kb)]
        if vs: q[(ka,kb)] = sum(vs)/len(vs)
    if len(q) == 4:
        inter = q[(1,1)] - q[(1,0)] - q[(0,1)] + q[(0,0)]
        flag = "  <<< non-additive" if abs(inter) > 0.5 else ""
        print(f"    {a:12s} x {b:12s} {inter:+7.2f}{flag}")
print("  reading: a large interaction means the two knobs are NOT independent --")
print("  OFAT would have scored at least one of them null. Main effects here are")
print("  averages over 8 cells, so they are far steadier than OFAT's single-cell")
print("  deltas. E_final remains a FILTER: the best cells still need seeded")
print("  success evals to decide whether lower E means better plans or more")
print("  exploitation of the world model.")
print("PLDM_FACTORIAL_DONE")
PY
log "done"
