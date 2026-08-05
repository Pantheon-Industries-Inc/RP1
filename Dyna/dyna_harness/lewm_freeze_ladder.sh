#!/bin/bash
# FREEZE LADDER on LeWM: how long should the critic keep training?
#
# WHY. The LeWM transfer test's "baseline" arm scored 89.6 against a historical
# anchor of 87.8 (3 seeds) / 87.1 (6 seeds). Diffing the two commands, ONE flag
# differs: the transfer arms carry --freeze-critic-frac 0.05 (freeze at step
# 300 of 6000) while the historical dsp_pre omitted it and took the default 0.8
# (step 4800). 0.05 was never a performance choice -- it was adopted so all 16
# factorial cells would share one static teacher and E_final would stay a common
# yardstick. It then rode into the LeWM test unexamined.
#
# WHAT THE FLAG ACTUALLY CONTROLS. freeze_at gates three things at once
# (train_lip_ac.py:456-459), not just the freeze point:
#   * the critic stops updating after freeze_at
#   * the expectile anneal 0.1 -> 0.03 completes over freeze_at steps
#   * the critic LR cosine 1e-3 -> 1e-4 completes over freeze_at steps
# Since --init-value already supplies a fully trained 6000-step TD teacher,
# frac 0.05 means "use the pretrained teacher nearly as-is" and frac 0.8 means
# "let it drift 4800 steps under LIP's own rollouts". Opposite ends of a real
# design axis, and only the two ends have ever been measured -- on different
# assets, so not even cleanly against each other.
#
# THREE CANDIDATE EXPLANATIONS for 89.6 vs 87.8, which this separates:
#   (a) the freeze flag is worth ~+1.8
#   (b) the post-volume-loss rebuild of the cache + TD teacher differs
#   (c) 3-seed noise (+1.8, combined se ~0.93, t~1.9)
# The frz48 arm IS the historical recipe rebuilt on today's assets, so it is
# the isolation run: land near 87.8 and the flag is real (a); land near 89.6
# and the rebuild moved (b); the seed spread tells us about (c).
#
# PRIOR FROM PLDM: there the fraction looks like a NULL -- fx_none at frac 0.05
# scored 72.7 (1 seed x 3 draws) and the grid at the default 0.8 scored 72.8
# (20 seeds). If LeWM shows a real ladder and PLDM does not, that is a
# base-strength interaction, not a universal knob.
#
# DESIGN. Everything else is byte-identical to the lewmbase arm: v2WM, amax 1.6,
# horizon 5, iters 8, steps 6000, n-step 50, arch v4, the same LR/expectile
# schedules, seeds {0,1,2}, draws {42,43,44}, held-out ep_range 8000:10000, EGL.
# Only --freeze-critic-frac varies. The frac 0.05 point is ALREADY BANKED as
# lewmbase (89.56), so this adds three rungs and yields a 4-point curve:
#
#     freeze_at    300  |  2000  |  3000  |  4800
#     frac         0.05 | 0.3334 |  0.5   |  0.8      <- 0.8 = historical
#     banked       89.56|   ?    |   ?    |   ?
#
# 9 trains (~50 min, one wave over 12 slots) + 27 evals (~6 min). ~1h.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_lewmfrz.log
V2WM=/workspace/models/v2WM
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
LF1=/workspace/caches/v2_tr8000_fs1.pt
LF5=/workspace/caches/v2_tr8000_fs5.pt
LTD=/workspace/metrics/v2_tr8000_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
AMAX=1.6; NSLOT=12; NGPU=8; SLOTDIR=/tmp/frzslots
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
rmdir "$SLOTDIR"/* 2>/dev/null || true

# tag|frac   (int(frac*6000) is printed by the trainer as the freeze step and
#             is gated below -- a typo'd frac must not pass silently)
ARMS=("frz2k|0.3334" "frz3k|0.5" "frz48|0.8")

log "=== LeWM FREEZE LADDER (pid $$): ${#ARMS[@]} arms x 3 seeds, amax $AMAX ==="
for f in "$LF1" "$LF5" "$LTD" "$AH5" "$V2WM/config.json"; do
  [ -e "$f" ] || die "missing $f -- the rebuilt LeWM assets are gone"; done
# the loader takes the model DIR and requires exactly one .pt inside; the
# checkpoint here is weights_epoch_22.pt, so match by glob, not by name.
n_ckpt=$(find "$V2WM" -maxdepth 1 -name '*.pt' | wc -l)
[ "$n_ckpt" = 1 ] || die "v2WM holds $n_ckpt .pt files, loader needs exactly 1"
b=$(sc lewmbase_s0_e42); [ -n "$b" ] || die "lewmbase rows absent; the frac-0.05 rung is not banked"
log "  banked rung: frac 0.05 (freeze 300) = 89.56"

# ------------------------------------------------------------------- P1 train
log "P1: 9 trains"
for e in "${ARMS[@]}"; do
  t="${e%%|*}"; fr="${e#*|}"
  for s in $SEEDS; do
    out=/workspace/actors/lip4_${t}_s${s}.pt
    [ -f "$out" ] && { log "  $t/s$s present"; continue; }
    slot=$(acquire); gpu=$(( slot % NGPU ))
    (
      CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$P/train_lip_ac.py" \
        --cache "$LF5" --cache-td "$LF1" --h5 "$AH5" --wm "$V2WM" --init-value "$LTD" \
        --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
        --actor-lr 3e-4 --actor-lr-final 3e-5 \
        --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
        --arch v4 --seed "$s" --freeze-critic-frac "$fr" \
        --out "$out" --out-value "${out%.pt}_value.pt" > "$L/frz_${t}_s${s}.log" 2>&1
      release "$slot"
    ) &
    log "  -> $t/s$s frac $fr on gpu $gpu (slot $slot)"
  done
done
wait
log "P1: trains done"

# GATE: the trainer prints the step it froze at. Verify each arm froze where the
# frac says, so a mis-specified fraction cannot masquerade as a result.
for e in "${ARMS[@]}"; do
  t="${e%%|*}"; fr="${e#*|}"
  want=$(awk -v f="$fr" 'BEGIN{printf "%d", int(f*6000)}')
  got=$(grep -oE "^step [0-9]+: critic\+teacher frozen" "$L/frz_${t}_s0.log" \
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
    policy="$V2WM" solver=lip "solver.actor_path=/workspace/actors/lip4_${arm}_s${s}.pt" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
  release "$slot"
}
log "P2: 27 eval cells"
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
LAD = [("lewmbase", 300, "0.05  (transfer-test arm, banked)"),
       ("frz2k",   2000, "0.3334"),
       ("frz3k",   3000, "0.5"),
       ("frz48",   4800, "0.8   <- HISTORICAL dsp_pre recipe")]
print("=== LeWM CRITIC-FREEZE LADDER (v2WM, amax 1.6, held-out 8000:10000, EGL) ===")
print("    everything but --freeze-critic-frac held identical; 3 seeds x 3 draws")
print(f"  {'freeze_at':>9s} {'frac':34s} {'s0':>5s} {'s1':>5s} {'s2':>5s} {'mean':>6s} {'spread':>7s}")
res = {}
for a, at, lab in LAD:
    xs = arm(a)
    if not xs: print(f"  {at:9d} {lab:34s}   (no complete seeds)"); continue
    res[a] = xs
    pad = list(xs) + [float('nan')]*(3-len(xs))
    print(f"  {at:9d} {lab:34s} {pad[0]:5.1f} {pad[1]:5.1f} {pad[2]:5.1f} "
          f"{mean(xs):6.2f} {max(xs)-min(xs):7.2f}")
print()
hist3, hist6 = 87.8, 87.1
h = res.get("frz48")
if h and len(h) == 3:
    m = mean(h); s = sd(h); se = s/math.sqrt(3)
    print(f"  ISOLATION -- frz48 IS the historical recipe on rebuilt assets:")
    print(f"    rebuilt {m:.2f} +/- {se:.2f}   historical 3-seed {hist3}  6-seed {hist6}")
    t = (m - hist3)/se if se > 0 else float("inf")
    if abs(m - hist3) < 1.5:
        print(f"    -> lands on the anchor ({m-hist3:+.2f}). The REBUILD is faithful, so the")
        print(f"       89.6 gap is the FREEZE FLAG, not the assets.")
    else:
        print(f"    -> misses the anchor by {m-hist3:+.2f} (t~{t:.2f}). The rebuilt cache/TD")
        print(f"       teacher differs from the lost originals; the historical 87.8 is")
        print(f"       stale as a cross-campaign reference and should be re-baselined.")
b = res.get("lewmbase")
if b and h and len(b) == 3 and len(h) == 3:
    d = [x-y for x, y in zip(b, h)]
    m, s = mean(d), sd(d); t = m/(s/math.sqrt(3)) if s > 0 else float("inf")
    print(f"\n  frac 0.05 vs 0.8 (paired on seed): {m:+.2f}  sd {s:.2f}  t {t:.2f} (df=2, crit 2.92)"
          f"  {'SIGNIFICANT' if abs(t) > 2.92 else 'not significant'}")
best = max(res, key=lambda k: mean(res[k])) if res else None
if best:
    print(f"  best rung: {best} at {mean(res[best]):.2f}")
    print(f"  spreads: " + "  ".join(f"{a}={max(res[a])-min(res[a]):.1f}" for a, _, _ in LAD if a in res))
    print("  A monotone curve is a real axis worth a 6-seed confirm. A flat one means")
    print("  the teacher's late drift does not matter on a strong base -- matching PLDM,")
    print("  where 0.05 (72.7) and 0.8 (72.8) were indistinguishable -- and then the")
    print("  89.6-vs-87.8 gap is 3-seed noise and nothing more.")
print("LEWM_FRZ_DONE")
PY
log "done"
