#!/bin/bash
# PER-BASE LIP-SIDE SWEEP: the two knobs the LR sweep does not touch, swept
# independently on each base because the directive allows it.
#
# WHY THESE TWO, AND WHY PER BASE. The shared-knob rule binds the TD/critic and
# data side -- replay-prob, expand-weight, expectile, gamma, n-step, p-cross,
# freeze-critic-frac. It explicitly leaves LIP-side knobs free to differ, and
# amax already does (LeWM 1.6 / PLDM 4.5). mean-weight is LIP-side too and has
# NEVER been swept on LeWM at all; PLDM's factorial preferred 0.03 over the 0.1
# default, which was never carried across. So a winner here is adoptable for one
# base at ZERO cost to the other -- the cheapest headroom left.
#
# WHAT MEAN-WEIGHT DOES. The actor loss is
#     loss = e_path[-1] + mean_weight * mean(e_path)          (train_lip_ac.py:431)
# where e_path[k] is the critic at the rollout endpoint after inner iterate k.
# So mean_weight trades ARRIVING (endpoint only) against DESCENDING MONOTONICALLY
# (all iterates shaped). It is the same axis PWM sits at the far end of --
# PWM's objective is a pure trajectory mean, i.e. mean_weight = infinity, and it
# collapses on bottleneck tasks for exactly that reason. PLDM wanting 0.03
# (more endpoint-dominated than the 0.1 default) is the same signal.
#
# AMAX. LeWM's 1.6 and PLDM's 4.5 were both selected by an earlier sweep -- but
# that sweep ran BEFORE the winner bundle (batch256/replay50/expand10) and before
# the freeze fraction moved to 0.5, and amax interacts with both: a larger action
# box means the actor's imagined states wander further off the expert manifold,
# which is precisely what replay and expand correct. So the amax optimum has
# plausibly moved and nobody has rechecked it at the current operating point.
# Also a standing caveat: amax was originally selected ON THE EVAL DRAWS (~3 pts
# of optimism). Re-selecting it here inherits that, so treat an amax winner as
# provisional until confirmed on a fresh draw.
#
# BASELINE is the deployed operating point, already measured, not recomputed:
#     --freeze-critic-frac 0.5 --batch 256 --replay-prob 0.5 --expand-weight 1.0
#     lwin3k = 88.67 (LeWM, amax 1.6)   pwin3k = 85.33 (PLDM, amax 4.5)
# Arm flags append AFTER the base flags so argparse's last-wins applies.
#
# The teacher and --gamma come from the TD upgrade, same as the LR sweep, so the
# two sweeps are measured against the same critic and their winners compose.
#
# REFERENCE POINT WORTH BEATING: LeWM's best base ever measured is 90.44
# (pcross0+batch256+replay50 at freeze 0.05) with 90.00 and 89.56 alongside it --
# all three inside 0.9 of each other and of one seed's noise. The deployed
# shared config gives LeWM 88.67, so there is ~1.8 pts sitting on the table that
# the shared-knob constraint currently forbids recovering. A per-base winner here
# recovers it WITHOUT breaking the constraint, because these knobs are per-base
# by rule.
#
# 6 arms x 2 bases x 3 seeds = 36 trains + 108 h25 cells. ~4h.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_perbase.log
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
WIN="--batch 256 --replay-prob 0.5 --expand-weight 1.0"
NSLOT=10; NGPU=8; SLOTDIR=/tmp/pbslots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
rmdir "$SLOTDIR"/* 2>/dev/null || true

log "=== PER-BASE LIP SWEEP (pid $$): amax + mean-weight, 6 arms x 2 bases x 3 seeds ==="
# Now queued behind the replay/expand level sweep as well: replay-prob and
# expand-weight ARE the winner bundle these arms sit on top of, and both were
# pinned at the top of a 3-level grid whose middle levels were only ever scored
# on E_final. If those base levels move, every refinement here is measured at
# the wrong operating point, so they get resolved first.
log "P0: waiting for the LR sweep and the replay/expand level sweep"
T0=$(date +%s)
until grep -q "REPLAY_EXPAND_DONE" "$L/driver_repexp.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 57600 ] && { log "WARN: proceeding after 16h"; break; }
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]" >/dev/null; do sleep 60; done
[ -f /workspace/td_winner.env ] || die "td_winner.env absent"
# shellcheck disable=SC1091
. /workspace/td_winner.env
# GAMMA 0.98, adopted by decision on a tie. The 6-point curve was a null --
# column ranges 1.55-4.23 against ~3 pts of noise, per-column optima scattered
# (0.98/0.90/1.00/0.95), min-rule winner 1.00 at +0.00 vs 0.98 at -0.67. On the
# 4-WM mean 0.98 leads by 0.22, and it is +1.11 on LeWM specifically. Neither
# value is distinguishable from the other; 0.98 is chosen so the whole chain
# (LR sweep, lr2base, dyna_h100) sits at one gamma.
# CONSEQUENCE: the banked baselines lwin3k 88.67 / pwin3k 85.33 were trained at
# gamma 1.0 with the OLD 6k teacher, so they are NOT the right reference for
# these arms. lr2base (deployed config, upgraded teacher, gamma 0.98) is, and
# it is being measured in parallel.
log "P0: teacher = $TD_ARM (12k steps), LIP --gamma $TD_GAMMA (adopted on a tie)"

# base|wm|fs5|fs1|TD|base-amax|baseline-tag
BASES=(
 "lewm|/workspace/models/v2WM|/workspace/caches/v2_tr8000_fs5.pt|/workspace/caches/v2_tr8000_fs1.pt|$TD_LEWMPRE|1.6|lwin3k"
 "pldm|/workspace/models/PLDM_OgBench_lewm|/workspace/caches/pldm_tr8000_fs5.pt|/workspace/caches/pldm_tr8000_fs1.pt|$TD_PLDMPRE|4.5|pwin3k"
)
# arms are PER BASE: amax levels bracket each base's own tuned value
# BUGFIX (take 1 produced 0/36 actors and still printed DONE): the arms were
# returned space-separated, but every arm CONTAINS a space ("mw000|--mean-weight
# 0.0"), so `for a in $(arms_for ...)` word-split mid-flag, every train got
# garbage and died, and the card rendered against an empty result set. Arms are
# now newline-delimited and read with `while IFS= read -r`, which does not split
# on the spaces inside a flag.
arms_for(){ case "$1" in
  lewm) printf '%s\n' "mw000|--mean-weight 0.0" "mw003|--mean-weight 0.03" \
        "mw030|--mean-weight 0.3" "amaxlo|--amax 1.2" "amaxhi|--amax 2.0" \
        "amaxvhi|--amax 2.4" ;;
  pldm) printf '%s\n' "mw000|--mean-weight 0.0" "mw003|--mean-weight 0.03" \
        "mw030|--mean-weight 0.3" "amaxlo|--amax 3.5" "amaxhi|--amax 5.5" \
        "amaxvhi|--amax 6.5" ;;
esac; }

log "P1: 36 trains"
for e in "${BASES[@]}"; do
  IFS='|' read -r nm wm c5 c1 td am bt <<< "$e"
  while IFS= read -r a; do
    t="${a%%|*}"; x="${a#*|}"
    for s in $SEEDS; do
      out=/workspace/actors/lip4_pb_${nm}_${t}_s${s}.pt
      [ -f "$out" ] && { log "  $nm/$t/s$s present"; continue; }
      slot=$(acquire)
      ( # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
          --cache "$c5" --cache-td "$c1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
          --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$am" \
          --actor-lr 3e-4 --actor-lr-final 3e-5 \
          --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
          --arch v4 --seed "$s" --freeze-critic-frac 0.5 --gamma "$TD_GAMMA" $WIN $x \
          --out "$out" --out-value "${out%.pt}_value.pt" > "$L/pbsw_${nm}_${t}_s${s}.log" 2>&1
        release "$slot"
      ) &
    done
  done < <(arms_for "$nm")
done
wait
log "P1: $(ls /workspace/actors/lip4_pb_*_s[0-9].pt 2>/dev/null | wc -l)/36 actors"

na=$(ls /workspace/actors/lip4_pb_*_s[0-9].pt 2>/dev/null | wc -l)
[ "$na" -ge 18 ] || die "only $na/36 actors -- refusing to score an empty sweep (take 1 printed a card against 0 actors)"
log "P2: 108 eval cells (h25)"
for e in "${BASES[@]}"; do
  IFS='|' read -r nm wm c5 c1 td am bt <<< "$e"
  while IFS= read -r a; do
    t="${a%%|*}"
    for s in $SEEDS; do
      [ -f "/workspace/actors/lip4_pb_${nm}_${t}_s${s}.pt" ] || continue
      for d in $DRAWS; do
        k="pbev_${nm}_${t}_s${s}_e${d}"
        c=$(sc "$k"); [ -n "$c" ] && [ "$c" != FAIL ] && continue
        slot=$(acquire); g=$(( slot % NGPU ))
        (
          CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 7200 \
            python3 "$P/eval_wm.py" --config-name cube seed=$d \
            eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
            eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
            policy="$wm" solver=lip \
            "solver.actor_path=/workspace/actors/lip4_pb_${nm}_${t}_s${s}.pt" \
            output.filename="${k}.txt" > "$L/eval_${k}.log" 2>&1
          sr=""
          grep -q "ep_range ${EPHI}:10000" "$L/eval_${k}.log" && \
            sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${k}.log" | tail -1 | grep -oE "[0-9.]+$")
          flock "$SUM.lock" -c "echo '${k},${sr:-FAIL}' >> '$SUM'"
          release "$slot"
        ) &
      done
    done
  done < <(arms_for "$nm")
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
TAGS = ["mw000", "mw003", "mw030", "amaxlo", "amaxhi", "amaxvhi"]
LVL = {"lewm": {"amaxlo": "1.2", "amaxhi": "2.0", "amaxvhi": "2.4"},
       "pldm": {"amaxlo": "3.5", "amaxhi": "5.5", "amaxvhi": "6.5"}}
BASE = {"lewm": ("lwin3k_s{s}_e{d}", "amax 1.6, mw 0.1"),
        "pldm": ("pwin3k_s{s}_e{d}", "amax 4.5, mw 0.1")}
print("=== PER-BASE LIP SWEEP (h25, held-out 8000:10000, EGL, 3 seeds x 3 draws) ===")
print("    baseline = deployed op point; these knobs are PER-BASE by directive,")
print("    so a winner on one base costs the other base nothing.")
for nm, label in (("lewm", "LeWM"), ("pldm", "PLDM")):
    b = arm(BASE[nm][0])
    hdr = f"{mean(b):.2f}" if len(b) == 3 else "-"
    print(f"\n  --- {label} --- baseline {hdr} ({BASE[nm][1]})")
    print(f"  {'arm':9s} {'level':>7s} {'s0':>6s} {'s1':>6s} {'s2':>6s} {'mean':>7s} {'vs base':>8s} {'t':>6s}")
    got = [(t, arm(f"pbev_{nm}_{t}_s{{s}}_e{{d}}")) for t in TAGS]
    got = [(t, xs) for t, xs in got if xs]
    for t, xs in sorted(got, key=lambda kv: -mean(kv[1])):
        lvl = LVL[nm].get(t, t.replace("mw", "0.")[:4] if t.startswith("mw") else "")
        if t == "mw000": lvl = "0.0"
        elif t == "mw003": lvl = "0.03"
        elif t == "mw030": lvl = "0.3"
        pad = list(xs) + [float('nan')]*(3-len(xs))
        ds, ts = "       -", "     -"
        if len(b) == 3 and len(xs) == 3:
            dd = [x-y for x, y in zip(xs, b)]; m, s = mean(dd), sd(dd)
            tt = m/(s/math.sqrt(3)) if s > 0 else float("inf")
            ds, ts = f"{m:+8.2f}", f"{tt:6.2f}"
        star = "  <<<" if len(b) == 3 and len(xs) == 3 and mean(xs) > mean(b) else ""
        print(f"  {t:9s} {lvl:>7s} {pad[0]:6.1f} {pad[1]:6.1f} {pad[2]:6.1f} "
              f"{mean(xs):7.2f} {ds} {ts}{star}")
print("\n  REFERENCE: LeWM's best base ever measured is 90.44 (pcross0+batch256+")
print("  replay50 @ freeze 0.05), with 90.00 and 89.56 alongside -- all three")
print("  inside 0.9 of each other, i.e. inside one seed's noise. The deployed")
print("  shared config gives LeWM 88.67, so ~1.8 pts sit on the table. Anything")
print("  here that clears ~90.5 on LeWM would be genuinely new ground.")
print("  mean-weight is the same axis PWM sits at the far end of (its objective")
print("  is a pure trajectory mean = mean_weight infinity, and it collapses on")
print("  bottleneck tasks). PLDM already preferred 0.03 over the 0.1 default.")
print("  CAVEATS. 3 seeds, ~3 pts resolution; <<< is a screen not a test, and")
print("  with 6 arms x 2 bases some will clear it by chance. amax was ORIGINALLY")
print("  selected on these same eval draws (~3 pts optimism), so an amax winner")
print("  is provisional until confirmed on a fresh draw.")
print("PERBASE_SWEEP_DONE")
PY
log "done"
