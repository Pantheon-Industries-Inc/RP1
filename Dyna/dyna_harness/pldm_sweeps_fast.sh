#!/bin/bash
# ACCELERATED OFAT + FACTORIAL: work queue instead of wave barriers.
#
# THE PROBLEM THIS FIXES. The wave pattern
#     g=$((g+1)); [ $g -ge 8 ] && { wait; g=0; }
# blocks on the SLOWEST arm in each wave. Measured live: 5 of 8 GPUs idle at
# 0% while three stragglers (iters 32, iters16+12k steps, 15k-step frozen
# critic) ran on -- those arms are 2-4x longer than their wave-mates, so most
# of the box sat unused for over an hour. Stage gating made it worse: OFAT was
# blocked on "box quiet" although none of its arms depend on the stragglers.
#
# THE FIX.
#  1. Slot pool, not waves. A job starts the instant ANY slot frees. mkdir is
#     the mutex (atomic on POSIX), so no lost updates.
#  2. 12 concurrent slots over 8 GPUs (~1.5 jobs/GPU, peaking at 2). Measured
#     headroom: each train uses 5-16 GB of 143 GB and leaves ~35% of the SM
#     idle, and CPU load was 3.3 of 192 cores. Memory worst case ~32 GB/GPU.
#  3. No drain-wait. New arms coexist with whatever is still finishing; the
#     slot pool simply gives them the free capacity.
#
# Writes the same markers/logs the downstream stages already wait on
# (driver_ofat.log / driver_factorial.log), so lewm_transfer.sh still chains.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4     # 12 jobs x 4 threads << 192 cores
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
PLDM=/workspace/models/PLDM_OgBench_lewm
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
AMAX=4.5; NSLOT=12; NGPU=8
SLOTDIR=/tmp/lipslots
mkdir -p "$L" "$R" "$SLOTDIR"
OF=$L/driver_ofat.log; FX=$L/driver_factorial.log
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$L/driver_fast.log"; }
cd "$CODE"

# ---------------- slot pool: mkdir is atomic, so this is a real mutex
acquire(){ local s; while :; do
    for s in $(seq 0 $((NSLOT-1))); do
      if mkdir "$SLOTDIR/$s" 2>/dev/null; then echo "$s"; return; fi
    done; sleep 3; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
rmdir "$SLOTDIR"/* 2>/dev/null || true      # clear stale slots from a prior run

# ---------------- launch one training job into a slot (blocks only for a slot)
spawn(){ # tag logfile extra-args...
  local tag=$1 lf=$2; shift 2
  local out=/workspace/actors/lip4_pldm_${tag}_s0.pt
  [ -f "$out" ] && { log "  $tag present"; return 0; }
  local slot; slot=$(acquire); local gpu=$(( slot % NGPU ))
  ( # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$P/train_lip_ac.py" \
      --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
      --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
      --actor-lr 3e-4 --actor-lr-final 3e-5 \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --arch v4 --seed 0 --freeze-critic-frac 0.05 "$@" \
      --out "$out" --out-value "${out%.pt}_value.pt" > "$lf" 2>&1
    release "$slot"
  ) &
  log "  -> $tag on gpu $gpu (slot $slot)"
}

# ============================ OFAT ============================
ARMS=(
  "base|" "expand01|--expand-weight 0.1" "expand10|--expand-weight 1.0"
  "replay25|--replay-prob 0.25" "replay50|--replay-prob 0.5"
  "mw003|--mean-weight 0.03" "mw000|--mean-weight 0.0"
  "itemb|--iter-mode emb" "headprec|--head-mode precond"
  "wide|--width 512" "deep|--layers 4" "sdim512|--s-dim 512"
  "rec1024|--rec-hidden 1024" "zeroinit|--zero-init" "gdinit|--gd-init 0.1"
  "featnorm|--feat-norm" "s0z0|--s0-mode z0" "feedend|--feed end"
  "feedtraj|--feed traj" "pcross0|--p-cross 0.0" "pcross6|--p-cross 0.6"
  "maxd20|--max-delta 20" "batch256|--batch 256"
)
log "=== FAST SWEEPS (pid $$): OFAT ${#ARMS[@]} arms, $NSLOT slots, no drain-wait ==="
echo "[$(date -u +%m%d-%H:%M:%S)] FAST runner took over OFAT" >> "$OF"
for e in "${ARMS[@]}"; do
  t="${e%%|*}"; x="${e#*|}"
  # shellcheck disable=SC2086
  spawn "of_$t" "$L/ofat_${t}_s0.log" $x
done
wait
log "OFAT: all arms done"

python3 - "$L" 2>&1 | tee -a "$OF" <<'PY'
import sys, re, glob, os
L = sys.argv[1]
def lastE(p, k="E_final"):
    try: t = open(p).read()
    except OSError: return None
    m = re.findall(rf"{k} ([0-9.]+)", t); return float(m[-1]) if m else None
rows = [(os.path.basename(f)[5:-7], lastE(f), lastE(f, "E_first"))
        for f in sorted(glob.glob(f"{L}/ofat_*_s0.log"))]
rows = [r for r in rows if r[1] is not None]
base = next((e for t, e, _ in rows if t == "base"), None)
rows.sort(key=lambda r: r[1])
print("=== PLDM OFAT SCREEN (common static teacher, seed 0) ===")
print(f"  {'arm':10s} {'E_final':>8s} {'E_first':>8s} {'vs base':>9s}")
for t, ef, e1 in rows:
    d = f"{100*(ef-base)/base:+.1f}%" if base else "?"
    print(f"  {t:10s} {ef:8.2f} {(e1 or 0):8.2f} {d:>9s}"
          + ("  <<<" if base and abs(ef-base) > 0.05*base else ""))
mv = [t for t, ef, _ in rows if base and abs(ef-base) > 0.05*base]
print(f"  movers (>5%): {' '.join(mv) if mv else 'NONE'}")
print("  E_final is a FILTER not a selector -- lower is not automatically better.")
print("PLDM_OFAT_DONE")
PY

# ========================== FACTORIAL ==========================
mapfile -t FACTORS < <(python3 - "$L" <<'PY'
import sys, re, glob, os
L = sys.argv[1]
FLAG = {"expand01":"--expand-weight 0.1","expand10":"--expand-weight 1.0",
 "replay25":"--replay-prob 0.25","replay50":"--replay-prob 0.5",
 "mw003":"--mean-weight 0.03","mw000":"--mean-weight 0.0",
 "itemb":"--iter-mode emb","headprec":"--head-mode precond","wide":"--width 512",
 "deep":"--layers 4","sdim512":"--s-dim 512","rec1024":"--rec-hidden 1024",
 "zeroinit":"--zero-init","gdinit":"--gd-init 0.1","featnorm":"--feat-norm",
 "s0z0":"--s0-mode z0","feedend":"--feed end","feedtraj":"--feed traj",
 "pcross0":"--p-cross 0.0","pcross6":"--p-cross 0.6","maxd20":"--max-delta 20",
 "batch256":"--batch 256"}
def lastE(p):
    try: t = open(p).read()
    except OSError: return None
    m = re.findall(r"E_final ([0-9.]+)", t); return float(m[-1]) if m else None
v = {os.path.basename(f)[5:-7]: lastE(f) for f in glob.glob(f"{L}/ofat_*_s0.log")}
v = {k: e for k, e in v.items() if e is not None}
base = v.get("base")
if base is None: sys.exit(0)
fam = lambda t: re.sub(r"[0-9]+$", "", t)
seen, out = set(), []
for t, e in sorted(v.items(), key=lambda kv: -abs(kv[1]-base)):
    if t == "base" or t not in FLAG or fam(t) in seen: continue
    seen.add(fam(t)); out.append(f"{t}|{FLAG[t]}")
    if len(out) >= 4: break
print("\n".join(out))
PY
)
if [ "${#FACTORS[@]}" -lt 2 ]; then
  log "FACTORIAL: fewer than 2 usable factors -- skipping"
  echo "PLDM_FACTORIAL_DONE" >> "$FX"
else
  NF=${#FACTORS[@]}; NC=$(( 1 << NF ))
  log "=== FACTORIAL 2^$NF = $NC cells ==="
  for f in "${FACTORS[@]}"; do log "    ${f%%|*} (${f#*|})"; done
  echo "[$(date -u +%m%d-%H:%M:%S)] FAST runner took over the factorial" >> "$FX"
  for ((c=0; c<NC; c++)); do
    tag=""; extra=""
    for ((i=0; i<NF; i++)); do
      if (( (c >> i) & 1 )); then
        tag="${tag}${FACTORS[$i]%%|*}."; extra="$extra ${FACTORS[$i]#*|}"
      fi
    done
    [ -z "$tag" ] && tag="none."
    tag="fx_${tag%.}"
    # shellcheck disable=SC2086
    spawn "$tag" "$L/${tag}_s0.log" $extra
  done
  wait
  log "FACTORIAL: all cells done"
  FS=$(printf '%s ' "${FACTORS[@]}")
  FACT_STR="$FS" python3 - "$L" 2>&1 | tee -a "$FX" <<'PY'
import os, sys, re, itertools
L = sys.argv[1]
facs = [f.split("|")[0] for f in os.environ["FACT_STR"].split()]
NF = len(facs)
def lastE(tag):
    try: t = open(f"{L}/fx_{tag}_s0.log").read()
    except OSError: return None
    m = re.findall(r"E_final ([0-9.]+)", t); return float(m[-1]) if m else None
cells = {}
for c in range(1 << NF):
    on = [facs[i] for i in range(NF) if (c >> i) & 1]
    e = lastE(".".join(on) if on else "none")
    if e is not None: cells[frozenset(on)] = e
print(f"=== PLDM 2^{NF} FACTORIAL on E_final (common static teacher) ===")
print(f"  cells: {len(cells)}/{1<<NF}")
for k in sorted(cells, key=lambda s: cells[s]):
    print(f"    {'+'.join(sorted(k)) or '(none)':38s} {cells[k]:7.2f}")
print("  --- main effects (ON minus OFF) ---")
for f in facs:
    on  = [v for k, v in cells.items() if f in k]
    off = [v for k, v in cells.items() if f not in k]
    if on and off:
        print(f"    {f:12s} {sum(on)/len(on)-sum(off)/len(off):+7.2f}  (n={len(on)}/{len(off)})")
print("  --- pairwise interactions (deviation from additive) ---")
for a, b in itertools.combinations(facs, 2):
    q = {}
    for ka, kb in itertools.product([0,1],[0,1]):
        vs = [v for k, v in cells.items()
              if (a in k)==bool(ka) and (b in k)==bool(kb)]
        if vs: q[(ka,kb)] = sum(vs)/len(vs)
    if len(q) == 4:
        it = q[(1,1)]-q[(1,0)]-q[(0,1)]+q[(0,0)]
        print(f"    {a:12s} x {b:12s} {it:+7.2f}" + ("  <<< non-additive" if abs(it)>0.5 else ""))
print("  main effects average 8 cells each, so they are far steadier than OFAT's")
print("  single-cell deltas. Best cells still need seeded success evals.")
print("PLDM_FACTORIAL_DONE")
PY
fi
log "done"
