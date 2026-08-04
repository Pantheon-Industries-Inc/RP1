#!/bin/bash
# REPLAY-PROB x EXPAND-WEIGHT LEVELS -- the winner bundle's own dials, never swept.
#
# WHY THIS GOES AHEAD OF THE PER-BASE SWEEP. replay-prob and expand-weight ARE
# the winner bundle: +10.6 on PLDM, and every sweep still queued builds on top
# of them held at 0.5 / 1.0. mean-weight and amax are refinements on that base.
# If the base levels are wrong, every refinement is measured on the wrong point.
#
# WHAT HAS ACTUALLY BEEN MEASURED:
#     --replay-prob    0, 0.25, 0.5     <- 0.5 is the TOP of the range
#     --expand-weight  0, 0.1, 1.0      <- 1.0 is the TOP of the range
# and the intermediate levels (0.25, 0.1) were scored ONLY on E_final in the
# OFAT screen. This campaign has twice proved E_final is a filter and not a
# selector: iters32 had near-lowest E_final and the WORST success; pcross0 was
# the best E_final arm of all and -3.3 on success. So 0.25 and 0.1 have
# effectively never been measured. The only success-scored contrast either knob
# has ever had is bundle-on vs bundle-off.
#
# That is structurally identical to the gamma situation that just collapsed:
# a value at the edge of a short grid, never probed beyond, adopted as a winner.
# Treat a winner here with the same suspicion.
#
# EXPAND HAS A SHARPER, MECHANICAL CASE. The winner cross showed the expand DOSE
# is load-bearing -- PLDM's winner benefit ran +6.00 at a 5% dose to +12.67 at
# 50%. But dose was varied through freeze_at, never through the weight itself.
# Meanwhile the operating point moved from freeze 300 to freeze 3000, a 10x
# increase in how long the expand term is even called (it lives inside
# critic_step, which stops at freeze_at). So expand-weight 1.0 was calibrated
# under a regime we no longer run. That is a mismatch, not a selected-from-noise
# artifact, which makes it the better bet of the two.
#
# replay-prob's case is weaker and should be stated as such: its only argument
# is "0.5 was the largest level we tried". replay-prob 1.0 is included as a
# BOUNDARY PROBE, not a candidate -- it replaces the entire actor batch with the
# previous step's own imagined windows, leaving no expert-cache contexts at all,
# and is expected to degenerate. It is here to bound the curve.
#
# SHARED KNOBS. Both are data/critic-side, so by directive a winner must hold on
# BOTH bases: scored by min(delta_LeWM, delta_PLDM).
#
# BASELINE is lr2base -- the deployed config with the upgraded 12k teacher and
# gamma 0.98, i.e. these arms differ from it in exactly one flag. The older
# lwin3k 88.67 / pwin3k 85.33 are NOT valid references here: they used the old
# 6k teacher at gamma 1.0.
#
# 5 arms x 2 bases x 3 seeds = 30 trains + 90 cells. ~3h.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_repexp.log
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
BASEFLAGS="--batch 256 --replay-prob 0.5 --expand-weight 1.0"
NSLOT=10; NGPU=8; SLOTDIR=/tmp/reslots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
rmdir "$SLOTDIR"/* 2>/dev/null || true

# tag|override appended after BASEFLAGS (argparse: last wins)
ARMS=(
  "rep025|--replay-prob 0.25"
  "rep075|--replay-prob 0.75"
  "rep100|--replay-prob 1.0"      # boundary probe, expected to degenerate
  "exp03|--expand-weight 0.3"
  "exp30|--expand-weight 3.0"
)

log "=== REPLAY x EXPAND LEVELS (pid $$): ${#ARMS[@]} arms x 2 bases x 3 seeds ==="
log "P0: waiting for the LR + freeze-gap sweep"
T0=$(date +%s)
until grep -q "LR_SWEEP_DONE" "$L/driver_lrsweep2.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && { log "WARN: proceeding after 12h"; break; }
  sleep 180
done
while pgrep -f "train_lip_a[c]|train_metri[c]" >/dev/null; do sleep 60; done
[ -f /workspace/td_winner.env ] || die "td_winner.env absent"
# shellcheck disable=SC1091
. /workspace/td_winner.env
log "P0: teacher $TD_ARM, --gamma $TD_GAMMA, baseline = lr2base"

BASES=(
 "lewm|/workspace/models/v2WM|/workspace/caches/v2_tr8000_fs5.pt|/workspace/caches/v2_tr8000_fs1.pt|$TD_LEWMPRE|1.6"
 "pldm|/workspace/models/PLDM_OgBench_lewm|/workspace/caches/pldm_tr8000_fs5.pt|/workspace/caches/pldm_tr8000_fs1.pt|$TD_PLDMPRE|4.5"
)

log "P1: 30 trains"
for e in "${BASES[@]}"; do
  IFS='|' read -r nm wm c5 c1 td am <<< "$e"
  for a in "${ARMS[@]}"; do
    t="${a%%|*}"; x="${a#*|}"
    for s in $SEEDS; do
      out=/workspace/actors/lip4_re_${nm}_${t}_s${s}.pt
      [ -f "$out" ] && { log "  $nm/$t/s$s present"; continue; }
      slot=$(acquire)
      ( # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
          --cache "$c5" --cache-td "$c1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
          --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$am" \
          --actor-lr 3e-4 --actor-lr-final 3e-5 \
          --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
          --arch v4 --seed "$s" --freeze-critic-frac 0.5 --gamma "$TD_GAMMA" $BASEFLAGS $x \
          --out "$out" --out-value "${out%.pt}_value.pt" > "$L/re_${nm}_${t}_s${s}.log" 2>&1
        release "$slot"
      ) &
    done
  done
done
wait
log "P1: $(ls /workspace/actors/lip4_re_*_s[0-9].pt 2>/dev/null | wc -l)/30 actors"

log "P2: 90 eval cells (h25)"
for e in "${BASES[@]}"; do
  IFS='|' read -r nm wm c5 c1 td am <<< "$e"
  for a in "${ARMS[@]}"; do
    t="${a%%|*}"
    for s in $SEEDS; do
      [ -f "/workspace/actors/lip4_re_${nm}_${t}_s${s}.pt" ] || continue
      for d in $DRAWS; do
        k="reev_${nm}_${t}_s${s}_e${d}"
        c=$(sc "$k"); [ -n "$c" ] && [ "$c" != FAIL ] && continue
        slot=$(acquire); g=$(( slot % NGPU ))
        (
          CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 7200 \
            python3 "$P/eval_wm.py" --config-name cube seed=$d \
            eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
            eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
            policy="$wm" solver=lip \
            "solver.actor_path=/workspace/actors/lip4_re_${nm}_${t}_s${s}.pt" \
            output.filename="${k}.txt" > "$L/eval_${k}.log" 2>&1
          sr=""
          grep -q "ep_range ${EPHI}:10000" "$L/eval_${k}.log" && \
            sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${k}.log" | tail -1 | grep -oE "[0-9.]+$")
          flock "$SUM.lock" -c "echo '${k},${sr:-FAIL}' >> '$SUM'"
          release "$slot"
        ) &
      done
    done
  done
done
wait
log "P2: evals done"

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
def arm(fmt):
    o = []
    for s in (0, 1, 2):
        vs = [rows.get(fmt.format(s=s, d=d)) for d in (42, 43, 44)]
        if all(v is not None for v in vs): o.append(mean(vs))
    return o
TAGS = [("rep025", "replay 0.25"), ("rep075", "replay 0.75"),
        ("rep100", "replay 1.0 (probe)"), ("exp03", "expand 0.3"),
        ("exp30", "expand 3.0")]
print("=== REPLAY x EXPAND LEVELS (h25, held-out, 3 seeds x 3 draws) ===")
print("    baseline = lr2base: deployed config (replay 0.5 / expand 1.0),")
print("    upgraded 12k teacher, gamma 0.98 -- arms differ in ONE flag.")
BASE, res = {}, {}
for nm in ("lewm", "pldm"):
    BASE[nm] = arm("lr2baseev_%s_s{s}_e{d}" % nm)
for nm, label in (("lewm", "LeWM (amax 1.6)"), ("pldm", "PLDM (amax 4.5)")):
    b = BASE[nm]
    hdr = f"{mean(b):.2f}" if len(b) == 3 else "MISSING"
    print(f"\n  --- {label} --- baseline {hdr}")
    print(f"  {'arm':20s} {'s0':>6s} {'s1':>6s} {'s2':>6s} {'mean':>7s} {'vs base':>8s} {'t':>6s}")
    for t, lab in TAGS:
        xs = arm(f"reev_{nm}_{t}_s{{s}}_e{{d}}")
        res[(nm, t)] = xs
        if not xs: print(f"  {lab:20s}   (incomplete)"); continue
        pad = list(xs) + [float('nan')]*(3-len(xs))
        ds, ts = "       -", "     -"
        if len(b) == 3 and len(xs) == 3:
            dd = [x-y for x, y in zip(xs, b)]; m, s = mean(dd), sd(dd)
            tt = m/(s/math.sqrt(3)) if s > 0 else float("inf")
            ds, ts = f"{m:+8.2f}", f"{tt:6.2f}"
        star = "  <<<" if len(b) == 3 and len(xs) == 3 and mean(xs) > mean(b) else ""
        print(f"  {lab:20s} {pad[0]:6.1f} {pad[1]:6.1f} {pad[2]:6.1f} "
              f"{mean(xs):7.2f} {ds} {ts}{star}")
print(f"\n  shared-knob score (both are data/critic-side: min of the two deltas)")
print(f"  {'arm':20s}{'d LeWM':>9s}{'d PLDM':>9s}{'min':>9s}")
best, bs = None, 0.0
for t, lab in TAGS:
    a, p = res.get(("lewm", t)), res.get(("pldm", t))
    if not (a and p and len(a) == 3 and len(p) == 3): continue
    if len(BASE["lewm"]) != 3 or len(BASE["pldm"]) != 3: continue
    dl = mean(a) - mean(BASE["lewm"]); dp = mean(p) - mean(BASE["pldm"])
    s = min(dl, dp)
    print(f"  {lab:20s}{dl:+9.2f}{dp:+9.2f}{s:+9.2f}")
    if s > bs: best, bs = lab, s
print(f"  -> {'winner: ' + best + f' (min {bs:+.2f})' if best else 'no arm beats the baseline on both bases; keep 0.5 / 1.0'}")
print()
print("  READING. expand is the one with a mechanical case: its dose scaled 10x")
print("  when the operating point moved from freeze 300 to 3000, and 1.0 was")
print("  calibrated under the old dose. replay's only argument was that 0.5 was")
print("  the biggest level ever tried -- the same shape of argument that made")
print("  gamma 0.98 look like a winner before a proper curve showed it was not.")
print("  replay 1.0 is a boundary probe (no expert contexts left in the batch),")
print("  not a candidate; if it does NOT collapse, that itself is informative.")
print("  CAVEATS. 3 seeds, ~3 pts resolution. <<< is a screen not a test, and")
print("  with 5 arms x 2 bases some clear it by chance. Any winner needs a")
print("  6-seed confirm before adoption -- three findings this week did not survive one.")
print("REPLAY_EXPAND_DONE")
PY
log "done"
