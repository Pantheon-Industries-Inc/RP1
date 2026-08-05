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

# ---------------- eval helper: success is the OBJECTIVE, E_final is not.
# The opt-capacity card proved this: iters32 had near-lowest E_final (9.04)
# and the WORST success (70.0 vs base ~73-75). So the factorial is scored on
# SUCCESS, with E_final kept only as a secondary diagnostic.
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
SUM=$R/summary_pldmgrid_egl.csv; EPHI=8000; EVAL_RANGE="${EPHI}:10000"
touch "$SUM" "$SUM.lock"
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
evspawn(){ # tag draw
  local tag=$1 d=$2 nm="fxev_${1}_e${2}"
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && return 0
  local slot; slot=$(acquire); local gpu=$(( slot % NGPU ))
  (
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
      python3 "$P/eval_wm.py" --config-name cube seed=$d \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
      policy="$PLDM" solver=lip \
      "solver.actor_path=/workspace/actors/lip4_pldm_${tag}_s0.pt" \
      output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
    sr=""
    grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
      sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
    flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
    release "$slot"
  ) &
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
# EXCLUDE knobs that change the GOAL DISTRIBUTION E_final is measured on.
# p-cross controls cross-episode goal sampling and max-delta the goal range,
# so changing them changes the TASK, not the optimiser: pcross0 scored 4.11
# (-60%) purely because same-episode goals are easier, and pcross6/maxd20 rose
# for the mirror-image reason. Crossing them into the factorial would poison
# half the cells with an easier problem rather than a better actor.
DIST = {"pcross", "maxd"}
seen, out = set(), []
for t, e in sorted(v.items(), key=lambda kv: -abs(kv[1]-base)):
    if t == "base" or t not in FLAG or fam(t) in seen or fam(t) in DIST: continue
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
  log "FACTORIAL: all cells trained"
  log "FACTORIAL: evaluating ALL $NC cells x 3 draws (success is the objective)"
  for ((c=0; c<NC; c++)); do
    tag=""
    for ((i=0; i<NF; i++)); do
      (( (c >> i) & 1 )) && tag="${tag}${FACTORS[$i]%%|*}."
    done
    [ -z "$tag" ] && tag="none."
    tag="fx_${tag%.}"
    for d in 42 43 44; do evspawn "$tag" "$d"; done
  done
  wait
  log "FACTORIAL: evals done"
  FS=$(printf '%s ' "${FACTORS[@]}")
  FACT_STR="$FS" SUMF="$SUM" python3 - "$L" 2>&1 | tee -a "$FX" <<'PY'
import os, sys, re, itertools
L = sys.argv[1]
rows = {}
for ln in open(os.environ["SUMF"]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
def succ(tag):
    vs = [rows.get(f"fxev_{tag}_e{d}") for d in (42,43,44)]
    return None if any(v is None for v in vs) else sum(vs)/3
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
# success per cell (the objective); E_final kept as a secondary diagnostic
scells = {}
for c in range(1 << NF):
    on = [facs[i] for i in range(NF) if (c >> i) & 1]
    s = succ("fx_" + (".".join(on) if on else "none"))
    if s is not None: scells[frozenset(on)] = s
print(f"=== PLDM 2^{NF} FACTORIAL (frozen PLDM, held-out, EGL, seed 0) ===")
print(f"  cells: E_final {len(cells)}/{1<<NF}, success {len(scells)}/{1<<NF}")
print(f"  reference: 20-seed LIP 72.8 | TD+CEM 73.3 | latent+CEM 66.7")
print(f"  {'cell':38s} {'success':>8s} {'E_final':>8s}")
for k in sorted(scells, key=lambda s: -scells[s]):
    print(f"    {'+'.join(sorted(k)) or '(none)':38s} {scells[k]:8.1f} {cells.get(k, float('nan')):8.2f}")
def effects(d, label, unit):
    print(f"  --- main effects on {label} (ON minus OFF) ---")
    for f in facs:
        on  = [v for k, v in d.items() if f in k]
        off = [v for k, v in d.items() if f not in k]
        if on and off:
            print(f"    {f:12s} {sum(on)/len(on)-sum(off)/len(off):+7.2f} {unit}  (n={len(on)}/{len(off)})")
    print(f"  --- pairwise interactions on {label} ---")
    for a, b in itertools.combinations(facs, 2):
        q = {}
        for ka, kb in itertools.product([0,1],[0,1]):
            vs = [v for k, v in d.items() if (a in k)==bool(ka) and (b in k)==bool(kb)]
            if vs: q[(ka,kb)] = sum(vs)/len(vs)
        if len(q) == 4:
            it = q[(1,1)]-q[(1,0)]-q[(0,1)]+q[(0,0)]
            big = abs(it) > (2.0 if unit == "pts" else 0.5)
            print(f"    {a:12s} x {b:12s} {it:+7.2f} {unit}" + ("  <<< non-additive" if big else ""))
if scells: effects(scells, "SUCCESS", "pts")
if cells:  effects(cells,  "E_final (diagnostic only)", "   ")
print("  Effects average 8 cells each -- far steadier than OFAT single-cell deltas.")
print("  SUCCESS is the objective: the opt-capacity card showed iters32 with the")
print("  lowest E_final and the WORST success, so an E_final ranking cannot pick")
print("  a winner. Single seed still, so the top cell needs seeds before belief.")
print("PLDM_FACTORIAL_DONE")
PY
fi
log "done"
