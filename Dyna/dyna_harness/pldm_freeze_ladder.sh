#!/bin/bash
# FREEZE LADDER on PLDM -- the twin of lewm_freeze_ladder.sh, on the weak base.
#
# WHY BOTH BASES. --freeze-critic-frac controls how long the critic keeps
# adapting under LIP's own rollouts before it is pinned (train_lip_ac.py:456-459
# -- it gates the freeze point AND the expectile anneal AND the critic LR
# cosine, all of which complete over freeze_at steps). Since --init-value
# already supplies a fully trained 6000-step TD teacher, frac 0.05 means "use
# the pretrained teacher nearly as-is" and frac 0.8 means "let it drift 4800
# steps". This campaign has run both ends without ever intending to compare
# them, so the axis is unmeasured on either base.
#
# THE BANKED PLDM PAIR IS ALREADY A CLEAN CONTRAST. --mean-weight defaults to
# 0.1, which is exactly what the grid passed explicitly, so:
#     fx_none                (frac 0.05, 1 seed x 3 draws)   72.7
#     grid_mw01_a45_lr3e-4   (frac 0.80, 20 seeds x 3 draws) 72.83  [68.0-79.3]
# differ in NOTHING but the freeze fraction -- and they are 0.1 apart. That is
# the prior this run tests properly: on PLDM the axis looks like a NULL. The
# 20-seed spread (68.0 to 79.3, 11.3 pts) is also the reminder of why: at 3
# seeds this measurement cannot resolve anything under ~3 pts.
#
# WHAT MAKES IT WORTH RUNNING ANYWAY. On LeWM the same axis is confounded with
# a +1.8 gap (transfer baseline 89.6 at frac 0.05 vs historical dsp_pre 87.8 at
# frac 0.8). If LeWM shows a ladder and PLDM stays flat, the knob interacts with
# base strength -- the teacher's late drift only hurts when the base is good
# enough for the actor to exploit it -- and that is a mechanism, not a knob.
# If BOTH are flat, the LeWM gap is 3-seed noise and the story ends cleanly.
#
# DESIGN. Identical to the PLDM factorial's fx_none cell except the fraction:
# PLDM_OgBench_lewm, amax 4.5, horizon 5, iters 8, steps 6000, n-step 50,
# arch v4, mean-weight default 0.1, same LR/expectile schedules, seeds {0,1,2},
# draws {42,43,44}, held-out 8000:10000, EGL. Four rungs run FRESH at 3 seeds
# each rather than reusing fx_none/grid actors, so every rung has equal seed
# support and one naming scheme:
#
#     freeze_at    300  |  2000  |  3000  |  4800
#     frac         0.05 | 0.3334 |  0.5   |  0.8
#
# SHARED SLOT POOL. This uses the SAME /tmp/frzslots pool as the LeWM ladder so
# the two cooperate over one 12-slot budget instead of oversubscribing the box.
# It therefore must NOT clear stale slots on startup -- that would release slots
# the LeWM run is holding. Stale slots are cleared only when nothing is training.
#
# 12 trains + 36 evals, sharing the pool with the LeWM ladder. ~2h for both.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_pldmfrz.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
AMAX=4.5; NSLOT=12; NGPU=8; SLOTDIR=/tmp/frzslots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do
    for s in $(seq 0 $((NSLOT-1))); do
      mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }
    done; sleep 3; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
# shared pool: only safe to clear when the box is genuinely idle
pgrep -f "train_lip_a[c]|eval_w[m]" >/dev/null || rmdir "$SLOTDIR"/* 2>/dev/null || true

ARMS=("pfrz05|0.05" "pfrz2k|0.3334" "pfrz3k|0.5" "pfrz48|0.8")

log "=== PLDM FREEZE LADDER (pid $$): ${#ARMS[@]} arms x 3 seeds, amax $AMAX ==="
for f in "$QF1" "$QF5" "$QTD" "$AH5" "$PLDM/config.json"; do
  [ -e "$f" ] || die "missing $f"; done
n_ckpt=$(find "$PLDM" -maxdepth 1 -name '*.pt' | wc -l)
[ "$n_ckpt" = 1 ] || die "$PLDM holds $n_ckpt .pt files, loader needs exactly 1"
log "  banked refs: fx_none frac0.05 = 72.7 (1 seed) | grid frac0.8 = 72.83 (20 seeds)"

# ------------------------------------------------------------------- P1 train
log "P1: 12 trains (shared 12-slot pool with the LeWM ladder)"
for e in "${ARMS[@]}"; do
  t="${e%%|*}"; fr="${e#*|}"
  for s in $SEEDS; do
    out=/workspace/actors/lip4_${t}_s${s}.pt
    [ -f "$out" ] && { log "  $t/s$s present"; continue; }
    slot=$(acquire); gpu=$(( slot % NGPU ))
    (
      CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$P/train_lip_ac.py" \
        --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
        --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
        --actor-lr 3e-4 --actor-lr-final 3e-5 \
        --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
        --arch v4 --seed "$s" --freeze-critic-frac "$fr" \
        --out "$out" --out-value "${out%.pt}_value.pt" > "$L/pfrz_${t}_s${s}.log" 2>&1
      release "$slot"
    ) &
    log "  -> $t/s$s frac $fr on gpu $gpu (slot $slot)"
  done
done
wait
log "P1: trains done"

# GATE: the trainer prints where it froze; a mis-specified fraction must not
# masquerade as a result.
for e in "${ARMS[@]}"; do
  t="${e%%|*}"; fr="${e#*|}"
  want=$(awk -v f="$fr" 'BEGIN{printf "%d", int(f*6000)}')
  got=$(grep -oE "^step [0-9]+: critic\+teacher frozen" "$L/pfrz_${t}_s0.log" \
        | grep -oE "[0-9]+" | head -1)
  [ "$got" = "$want" ] || die "$t froze at '$got', expected $want -- fraction wrong"
  log "  gate ok: $t froze at step $got"
done

# -------------------------------------------------------------------- P2 eval
ev(){ # slot arm seed draw
  local slot=$1 arm=$2 s=$3 d=$4 gpu=$(( $1 % NGPU )) nm="${2}_s${3}_e${4}"
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { release "$slot"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$PLDM" solver=lip "solver.actor_path=/workspace/actors/lip4_${arm}_s${s}.pt" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
  release "$slot"
}
log "P2: 36 eval cells"
for e in "${ARMS[@]}"; do
  t="${e%%|*}"
  for s in $SEEDS; do for d in $DRAWS; do
    slot=$(acquire); ev "$slot" "$t" "$s" "$d" &
  done; done
done
wait
log "P2: evals done"

# -------------------------------------------------------------------- P3 card
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
def sd(xs):
    m = mean(xs); return math.sqrt(sum((x-m)**2 for x in xs)/max(len(xs)-1, 1))
def arm(a):
    o = []
    for s in (0, 1, 2):
        vs = [rows.get(f"{a}_s{s}_e{d}") for d in (42, 43, 44)]
        if all(v is not None for v in vs): o.append(mean(vs))
    return o
LAD = [("pfrz05", 300, "0.05"), ("pfrz2k", 2000, "0.3334"),
       ("pfrz3k", 3000, "0.5"), ("pfrz48", 4800, "0.8  <- trainer default")]
print("=== PLDM CRITIC-FREEZE LADDER (amax 4.5, held-out 8000:10000, EGL) ===")
print("    everything but --freeze-critic-frac held identical; 3 seeds x 3 draws")
print("    banked: fx_none frac0.05 = 72.7 (1 seed) | grid frac0.8 = 72.83 (20 seeds)")
print(f"  {'freeze_at':>9s} {'frac':22s} {'s0':>5s} {'s1':>5s} {'s2':>5s} {'mean':>6s} {'spread':>7s}")
res = {}
for a, at, lab in LAD:
    xs = arm(a)
    if not xs: print(f"  {at:9d} {lab:22s}   (no complete seeds)"); continue
    res[a] = xs
    pad = list(xs) + [float('nan')]*(3-len(xs))
    print(f"  {at:9d} {lab:22s} {pad[0]:5.1f} {pad[1]:5.1f} {pad[2]:5.1f} "
          f"{mean(xs):6.2f} {max(xs)-min(xs):7.2f}")
lo, hi = res.get("pfrz05"), res.get("pfrz48")
if lo and hi and len(lo) == 3 and len(hi) == 3:
    d = [x-y for x, y in zip(lo, hi)]
    m, s = mean(d), sd(d); t = m/(s/math.sqrt(3)) if s > 0 else float("inf")
    print(f"\n  frac 0.05 vs 0.8 (paired on seed): {m:+.2f}  sd {s:.2f}  t {t:.2f} (df=2, crit 2.92)"
          f"  {'SIGNIFICANT' if abs(t) > 2.92 else 'not significant'}")
    print(f"  banked contrast said {72.7-72.83:+.2f} -- this run is the seeded version of it")
if res:
    span = max(mean(v) for v in res.values()) - min(mean(v) for v in res.values())
    best = max(res, key=lambda k: mean(res[k]))
    print(f"  best rung {best} at {mean(res[best]):.2f}; full-ladder span {span:.2f} pts")
    print(f"  PLDM refs: latent+CEM 66.7 | TD+CEM 73.3 (75.3 at gamma .98) | LIP 20-seed 72.8")
    print(f"             factorial winner batch256+replay50+expand10 = 83.3 (seed 0)")
    print("  READING vs the LeWM twin: ladder on LeWM but flat here => the knob")
    print("  interacts with base strength (late teacher drift only hurts once the")
    print("  base is good enough for the actor to exploit it). Flat on BOTH => the")
    print("  LeWM 89.6-vs-87.8 gap was 3-seed noise. Ladder on BOTH => a real,")
    print("  universal and so far unswept axis, and 0.8 was never justified either.")
    print("  Caveat either way: the 20-seed grid spread is 68.0-79.3, so 3 seeds")
    print("  cannot resolve under ~3 pts and a null here is 'not detected', not 'zero'.")
print("PLDM_FRZ_DONE")
PY
log "done"
