#!/bin/bash
# LEARNING-RATE + FREEZE-GAP SWEEP, both bases, at the selected operating point.
#
# TWO MOTIVATIONS, and they are the same motivation.
#
# (1) THE 90+ QUESTION. The 8 measured cells leave a Pareto frontier of exactly
# two points: win@300 (LeWM 90.00 / PLDM 79.56) and win@3000 (88.67 / 85.33).
# Everything else is dominated. So the only way to buy LeWM's 90 today costs
# 5.77 PLDM points -- 4.3 PLDM points per LeWM point. But ALL of that trade
# happens between freeze_at 300 and 2000, and nothing was measured in between.
# frz600/1000/1400 fill that gap.
#
# (2) freeze_at IS AN LR KNOB. It does not merely stop the critic; it sets the
# HORIZON of the critic LR cosine and the expectile anneal
# (train_lip_ac.py:456-459) -- both complete OVER freeze_at steps. So frz300
# means the critic LR collapses 1e-3 -> 1e-4 in 300 steps, and frz4800 means it
# decays slowly over 4800. The freeze ladder may therefore have been measuring
# an LR-SCHEDULE effect all along, with "when does the critic stop" and "how
# fast does its LR decay" perfectly confounded. The clr* arms below vary the
# critic LR at a FIXED freeze point, which is exactly what separates them.
#
# WHAT HAS AND HAS NOT BEEN SWEPT:
#   actor-lr   only ever DOWNWARD -- lrsw_alr (/3) was -2.2 on LeWM, and the
#              PLDM grid tried 1e-4. Upward is unexplored, and the campaign's
#              own note says "LIP is lr-sensitive".
#   critic-lr  NEVER SWEPT, on either base, in any campaign. 1e-3 -> 1e-4 was
#              inherited and never questioned.
#   decay shape  never questioned either; every run uses final = lr/10.
#
# BASELINE for every arm is the selected operating point:
#   --freeze-critic-frac 0.5 --batch 256 --replay-prob 0.5 --expand-weight 1.0
#   plus per-base amax (LeWM 1.6 / PLDM 4.5)
# already measured as lwin3k = 88.67 (LeWM) and pwin3k = 85.33 (PLDM), so no
# baseline work is repeated. Arm flags are appended AFTER the base flags, so
# argparse's last-wins gives the arm its override.
#
# SHARED vs PER-BASE. By directive, TD/critic and data knobs are shared and
# LIP-side knobs may differ per base. critic-lr is critic-side, so a clr winner
# must hold on BOTH bases to be adopted. actor-lr is LIP-side, so an alr winner
# may be adopted per base -- which is the cheap route to LeWM 90+ that costs
# PLDM nothing. The card scores both ways.
#
# 10 arms x 2 bases x 3 seeds = 60 trains, + 180 h25 eval cells. Selection runs
# at h25 (the tuned operating point); the winner can be taken to h100 after.
# ~7h, queued behind the planner ladder.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_lrsweep2.log
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
WIN="--batch 256 --replay-prob 0.5 --expand-weight 1.0"
NSLOT=10; NGPU=8; SLOTDIR=/tmp/lrslots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do
    for s in $(seq 0 $((NSLOT-1))); do
      mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }
    done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
pgrep -f "train_lip_a[c]|train_pwm_a[c]|eval_w[m]" >/dev/null || rmdir "$SLOTDIR"/* 2>/dev/null || true


# tag|flags appended after the base config (argparse: last wins)
ARMS=(
  # --- freeze gap: the whole LeWM/PLDM trade lives between 300 and 2000 ---
  "frz600|--freeze-critic-frac 0.1"
  "frz1000|--freeze-critic-frac 0.1667"
  "frz1400|--freeze-critic-frac 0.2334"
  # --- actor LR: only ever tested downward before ---
  "alr1e3|--actor-lr 1e-3 --actor-lr-final 1e-4"
  "alr6e4|--actor-lr 6e-4 --actor-lr-final 6e-5"
  "alr15e5|--actor-lr 1.5e-4 --actor-lr-final 1.5e-5"
  "alrflat|--actor-lr 3e-4 --actor-lr-final 3e-4"
  # --- critic LR: never swept in any campaign, on any base ---
  "clr3e3|--critic-lr 3e-3 --critic-lr-final 3e-4"
  "clr3e4|--critic-lr 3e-4 --critic-lr-final 3e-5"
  "clrflat|--critic-lr 1e-3 --critic-lr-final 1e-3"
)

log "=== LR + FREEZE-GAP SWEEP (pid $$): ${#ARMS[@]} arms x 2 bases x 3 seeds ==="
# TAKE 2. The first run of this sweep was killed at 34/60 actors: it was mapping
# the LR landscape against a critic we were about to replace. gamma/steps were
# never adopted from the TD sweep (g98 +3.3, s12k +2.7), and LIP warm-starts its
# critic from that teacher, so the two axes are not independent -- how fast the
# critic should learn depends on which critic it is. Actor names are lip4_lr2_*
# so the 34 stale gamma-1.0 actors are not silently reused.
log "P0: waiting for the TD teacher upgrade"
T0=$(date +%s)
until grep -q "TD_UPGRADE_DONE" "$L/driver_tdupgrade.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && { log "WARN: proceeding after 12h"; break; }
  sleep 180
done
while pgrep -f "train_metri[c]|train_pwm_a[c]" >/dev/null; do sleep 60; done
[ -f /workspace/td_winner.env ] || die "td_winner.env absent -- upgrade did not select"
# shellcheck disable=SC1091
. /workspace/td_winner.env
log "P0: upgraded teacher = $TD_ARM, LIP --gamma $TD_GAMMA"

# base|wm|fs5|fs1|TD|amax|baseline-tag   -- teacher comes from the upgrade
BASES=(
 "lewm|/workspace/models/v2WM|/workspace/caches/v2_tr8000_fs5.pt|/workspace/caches/v2_tr8000_fs1.pt|$TD_LEWMPRE|1.6|lwin3k"
 "pldm|/workspace/models/PLDM_OgBench_lewm|/workspace/caches/pldm_tr8000_fs5.pt|/workspace/caches/pldm_tr8000_fs1.pt|$TD_PLDMPRE|4.5|pwin3k"
)
for e in "${BASES[@]}"; do
  IFS='|' read -r nm wm c5 c1 td am bt <<< "$e"
  for f in "$wm/config.json" "$c5" "$c1" "$td" "$AH5"; do [ -e "$f" ] || die "$nm: missing $f"; done
done

# ------------------------------------------------------------------ P1 train
log "P1: 60 trains"
for e in "${BASES[@]}"; do
  IFS='|' read -r nm wm c5 c1 td am bt <<< "$e"
  for a in "${ARMS[@]}"; do
    t="${a%%|*}"; x="${a#*|}"
    for s in $SEEDS; do
      out=/workspace/actors/lip4_lr2_${nm}_${t}_s${s}.pt
      [ -f "$out" ] && { log "  $nm/$t/s$s present"; continue; }
      slot=$(acquire)
      ( # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
          --cache "$c5" --cache-td "$c1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
          --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$am" \
          --actor-lr 3e-4 --actor-lr-final 3e-5 \
          --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
          --arch v4 --seed "$s" --freeze-critic-frac 0.5 --gamma "$TD_GAMMA" $WIN $x \
          --out "$out" --out-value "${out%.pt}_value.pt" > "$L/lrsw_${nm}_${t}_s${s}.log" 2>&1
        release "$slot"
      ) &
    done
  done
done
wait
nt=$(ls /workspace/actors/lip4_lr2_*_s[0-9].pt 2>/dev/null | wc -l)
log "P1: $nt/60 trains produced actors"

# GATE: the freeze arms must actually freeze where their fraction says.
for a in frz600:600 frz1000:1000 frz1400:1400; do
  t="${a%%:*}"; want="${a##*:}"
  got=$(grep -oE "^step [0-9]+: critic\+teacher frozen" "$L/lrsw_lewm_${t}_s0.log" 2>/dev/null \
        | grep -oE "[0-9]+" | head -1)
  [ "$got" = "$want" ] && log "  gate ok: $t froze at $got" || log "  WARN: $t froze at '$got', expected $want"
done

# ------------------------------------------------------------------- P2 eval
ev(){ # slot base wm arm seed draw
  local slot=$1 nm=$2 wm=$3 t=$4 s=$5 d=$6 g=$(( $1 % NGPU )) k="lr2ev_${2}_${4}_s${5}_e${6}"
  local c; c=$(sc "$k"); [ -n "$c" ] && [ "$c" != FAIL ] && { release "$slot"; return 0; }
  CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$wm" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_lr2_${nm}_${t}_s${s}.pt" \
    output.filename="${k}.txt" > "$L/eval_${k}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${k}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${k}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${k},${sr:-FAIL}' >> '$SUM'"
  log "  $k = ${sr:-FAIL}"
  release "$slot"
}
log "P2: 180 eval cells (h25)"
for e in "${BASES[@]}"; do
  IFS='|' read -r nm wm c5 c1 td am bt <<< "$e"
  for a in "${ARMS[@]}"; do
    t="${a%%|*}"
    for s in $SEEDS; do
      [ -f "/workspace/actors/lip4_lr2_${nm}_${t}_s${s}.pt" ] || continue
      for d in $DRAWS; do slot=$(acquire); ev "$slot" "$nm" "$wm" "$t" "$s" "$d" & done
    done
  done
done
wait
log "P2: evals done"

# ------------------------------------------------------------------- P3 card
ARMS_STR=$(printf '%s ' "${ARMS[@]}")
ARMS_STR="$ARMS_STR" python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import os, sys, math
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
tags = [e.split("|")[0] for e in os.environ["ARMS_STR"].split()]
BASE = {"lewm": arm("lwin3k_s{s}_e{d}"), "pldm": arm("pwin3k_s{s}_e{d}")}
res = {}
for nm in ("lewm", "pldm"):
    for t in tags:
        xs = arm("lr2ev_%s_%s_s{s}_e{d}" % (nm, t))
        if xs: res[(nm, t)] = xs
GROUP = {"frz": "freeze gap (also a critic-LR-horizon change)",
         "alr": "actor LR", "clr": "critic LR"}
print("=== LR + FREEZE-GAP SWEEP (h25, held-out 8000:10000, EGL, 3 seeds x 3 draws) ===")
print("    baseline = the selected op point: frac 0.5 + batch256/replay50/expand10")
for nm, lbl in (("lewm", "LeWM (amax 1.6)"), ("pldm", "PLDM (amax 4.5)")):
    b = BASE[nm]
    print(f"\n  --- {lbl} --- baseline {mean(b):.2f}" if b else f"\n  --- {lbl} ---")
    print(f"  {'arm':10s} {'s0':>6s} {'s1':>6s} {'s2':>6s} {'mean':>7s} {'vs base':>8s} {'t':>6s}")
    ranked = sorted([t for t in tags if (nm, t) in res],
                    key=lambda t: -mean(res[(nm, t)]))
    for t in ranked:
        xs = res[(nm, t)]
        pad = list(xs) + [float('nan')]*(3-len(xs))
        ds, ts = "       -", "     -"
        if b and len(b) == 3 and len(xs) == 3:
            dd = [x-y for x, y in zip(xs, b)]; m, s = mean(dd), sd(dd)
            tt = m/(s/math.sqrt(3)) if s > 0 else float("inf")
            ds, ts = f"{m:+8.2f}", f"{tt:6.2f}"
        star = "  <<<" if b and len(xs) == 3 and mean(xs) > mean(b) else ""
        print(f"  {t:10s} {pad[0]:6.1f} {pad[1]:6.1f} {pad[2]:6.1f} "
              f"{mean(xs):7.2f} {ds} {ts}{star}")
# ---- the 90+ question, answered on the joint frontier
print("\n=== THE 90+ FRONTIER (LeWM >= 90 without giving up PLDM) ===")
print("    known frontier before this sweep: win@300 = 90.00/79.56,")
print("                                      win@3000 = 88.67/85.33 (deployed)")
print(f"  {'arm':10s} {'LeWM':>7s} {'PLDM':>7s} {'vs deployed LeWM':>18s} {'vs deployed PLDM':>18s}")
cands = []
for t in tags:
    a, p = res.get(("lewm", t)), res.get(("pldm", t))
    if not (a and p): continue
    la, pa = mean(a), mean(p)
    cands.append((t, la, pa))
    print(f"  {t:10s} {la:7.2f} {pa:7.2f} {la-mean(BASE['lewm']):+18.2f} {pa-mean(BASE['pldm']):+18.2f}")
hits = [c for c in cands if c[1] >= 90.0]
if hits:
    best = max(hits, key=lambda c: c[2])
    print(f"  -> {len(hits)} arm(s) reach LeWM >= 90. Best PLDM among them: {best[0]}"
          f" ({best[1]:.2f} / {best[2]:.2f})")
    cost = mean(BASE['pldm']) - best[2]
    print(f"     costs {cost:+.2f} PLDM vs deployed; the old frontier cost -5.77 for 90.00")
else:
    print("  -> no arm reaches LeWM 90 at 3 seeds. The 5.77-point trade at win@300")
    print("     remains the only route, and the plateau argument says do not take it.")
print("\n  READING.")
print("  * clr* arms vary the critic LR at a FIXED freeze point, so they separate")
print("    'when the critic stops' from 'how fast its LR decays' -- confounded in")
print("    the freeze ladder, where freeze_at sets the cosine horizon itself.")
print("  * alr is LIP-side, so an actor-LR winner may be adopted PER BASE and")
print("    costs the other base nothing -- the cheap route to LeWM 90+.")
print("    clr is critic-side and must hold on BOTH bases to be adopted.")
print("  * A frz600/1000/1400 winner would mean the gap between the two known")
print("    Pareto points was hiding a better operating point all along.")
print("  CAVEATS. 3 seeds resolves ~3 pts; <<< marks arms above baseline, which")
print("  is a screen and not a significance test -- with 10 arms x 2 bases, some")
print("  will clear it by chance. Any adopted winner needs a 6-seed confirm.")
print("  Selection is at h25 only; the winner should then be checked at h100.")
print("LR_SWEEP_DONE")
PY
log "done"
