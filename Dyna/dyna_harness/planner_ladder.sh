#!/bin/bash
# PLANNER LADDER x HORIZON: latent+CEM, TD+CEM, PWM and LIP on all four world
# models, at BOTH the in-distribution horizon (h25) and the out-of-distribution
# one (h100).
#
# WHY THE PLANNER AXIS. The freeze/cross/Dyna campaign measured ONE planner
# (LIP) on four WMs, which leaves the obvious question open: how much of PLDM's
# 72.8 -> 91.33 is the world model getting better versus LIP getting better at
# exploiting it? A search planner answers it directly. latent+CEM has NO learned
# value and NO learned actor -- it is pure WM rollout plus latent distance -- so
# if it also rises on the fine-tuned WMs, the Dyna fine-tune genuinely improved
# the MODEL and the gain is not a planner artifact.
#
#   latent+CEM  solver=cem                  no learned component at all. The
#                                           cleanest read on model quality.
#   TD+CEM      solver=cem "+metric=<TD>"   search scored by the learned
#                                           quasimetric; still no actor, so
#                                           TD+CEM minus latent+CEM isolates
#                                           what the critic contributes.
#   PWM         solver=pwm                  the AMORTIZED policy: a closed-loop
#                                           latent actor-critic, one forward
#                                           pass per step, no search and no
#                                           inner optimisation loop. LIP minus
#                                           PWM is what iterative refinement
#                                           buys over a plain actor.
#   LIP         solver=lip                  the campaign's planner, re-run here
#                                           under uniform naming so the whole
#                                           table is self-contained (and the
#                                           banked h25 numbers get re-validated).
#
# WHY THE HORIZON AXIS. Everything in this campaign was trained, tuned and
# measured at goal_offset 25 / budget 50 (h25). h100 -- goal = frame t+100,
# budget 200 -- is OUT OF DISTRIBUTION for every arm here. It is the honest
# stress test of whether "PLDM caught up to LeWM" is a property of the policies
# or an artifact of the operating point they were all tuned at. Prior evidence
# says h100 does not collapse (single cells held near their h25 rates) while
# h200 did, so h100 sits right at the edge where differences should show.
#
# THE FOUR WORLD MODELS, each with its OWN TD teacher trained on its OWN cache
# with the shared settings (expectile 0.03, n-step 50, 6000 steps, seed 0), so
# TD+CEM on POST is a like-for-like upgrade rather than a teacher transplant:
#   LeWM PRE   v2WM               PLDM PRE   PLDM_OgBench_lewm
#   LeWM POST  dyna_pb_jl         PLDM POST  dyna_pb_jp
#
# The correct TD+CEM override is `solver=cem "+metric=<path>"` (a top-level
# hydra override). `solver.cost_path` does NOT exist -- an earlier run FAILed
# every TD+CEM reference cell that way, so this is spelled out deliberately.
#
# PWM SETTINGS. Trainer defaults (steps 8000, gamma 0.99, max-delta 12,
# expectile 0.1->0.03, width 512, layers 3, actor-lr 5e-4) except --amax, which
# takes each base's LIP-tuned operating point (1.6 / 4.5). PWM's own amax
# optimum is unexplored -- a first-pass caveat, stated not hidden. Follows the
# tworoom_pwm.sh precedent (including solver.batch_size=10 at eval) so the PWM
# row stays comparable across campaigns.
#
# 12 PWM trains + 192 eval cells (96 per horizon). h100 cells carry a 4x budget
# so they cost ~4x the wall-clock of an h25 cell. ~3-4h total.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_planladder.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
NSLOT=10; NGPU=8; SLOTDIR=/tmp/plslots
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

# horizon tag | goal_offset_steps | eval_budget
HZ=("h25|25|50" "h100|100|200")

# tag|wm-dir|fs5|fs1|TD|amax|lip-actor-prefix
WMS=(
 "lewmpre|/workspace/models/v2WM|/workspace/caches/v2_tr8000_fs5.pt|/workspace/caches/v2_tr8000_fs1.pt|/workspace/metrics/v2_tr8000_TD.pt|1.6|lwin3k"
 "lewmpost|/workspace/models/dyna_pb_jl|/workspace/caches/pbjl_tr8000_fs5.pt|/workspace/caches/pbjl_tr8000_fs1.pt|/workspace/metrics/pbjl_TD.pt|1.6|pbpost_jl"
 "pldmpre|/workspace/models/PLDM_OgBench_lewm|/workspace/caches/pldm_tr8000_fs5.pt|/workspace/caches/pldm_tr8000_fs1.pt|/workspace/metrics/pldm_TD.pt|4.5|pwin3k"
 "pldmpost|/workspace/models/dyna_pb_jp|/workspace/caches/pbjp_tr8000_fs5.pt|/workspace/caches/pbjp_tr8000_fs1.pt|/workspace/metrics/pbjp_TD.pt|4.5|pbpost_jp"
)

log "=== PLANNER LADDER x HORIZON (pid $$): 4 planners, 4 WMs, h25 + h100 ==="
for e in "${WMS[@]}"; do
  IFS='|' read -r t wm c5 c1 td am lipt <<< "$e"
  for f in "$wm/config.json" "$c5" "$c1" "$td"; do [ -e "$f" ] || die "$t: missing $f"; done
  n=$(find "$wm" -maxdepth 1 -name '*.pt' | wc -l)
  [ "$n" = 1 ] || die "$t: $wm holds $n .pt files, loader needs exactly 1"
  for s in $SEEDS; do
    [ -f "/workspace/actors/lip4_${lipt}_s${s}.pt" ] || die "$t: LIP actor lip4_${lipt}_s${s}.pt missing"
  done
done
[ -f "$P/train_pwm_ac.py" ] || die "train_pwm_ac.py missing"
grep -q "PWMSolver" "$CODE/stable_worldmodel/solver/pwm.py" || die "PWMSolver missing"
log "P0: preflight OK -- 4 WMs, caches, TD teachers, 12 LIP actors, PWM stack"

# ---------------------------------------------------------------- generic eval
ev(){ # slot name wm-dir draw offset budget extra-hydra...
  local slot=$1 nm=$2 wm=$3 d=$4 off=$5 bud=$6; shift 6
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { release "$slot"; return 0; }
  CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) MUJOCO_EGL_DEVICE_ID=$(( slot % NGPU )) \
  timeout 14400 python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=$off eval.eval_budget=$bud "+eval.ep_range=$EVAL_RANGE" \
    policy="$wm" "$@" output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  $nm = ${sr:-FAIL}"
  release "$slot"
}

# --------------------------- P1 CEM + TD+CEM + LIP: no training needed
# These use assets that already exist, so they land first. 48 CEM cells and
# 72 LIP cells across both horizons.
log "P1: latent+CEM, TD+CEM and LIP -- both horizons, no training required"
for h in "${HZ[@]}"; do
  IFS='|' read -r hz off bud <<< "$h"
  for e in "${WMS[@]}"; do
    IFS='|' read -r t wm c5 c1 td am lipt <<< "$e"
    for d in $DRAWS; do
      slot=$(acquire); ev "$slot" "pl_cem_${t}_${hz}_e${d}"   "$wm" "$d" "$off" "$bud" solver=cem &
      slot=$(acquire); ev "$slot" "pl_tdcem_${t}_${hz}_e${d}" "$wm" "$d" "$off" "$bud" solver=cem "+metric=$td" &
      for s in $SEEDS; do
        slot=$(acquire)
        ev "$slot" "pl_lip_${t}_s${s}_${hz}_e${d}" "$wm" "$d" "$off" "$bud" \
           solver=lip "solver.actor_path=/workspace/actors/lip4_${lipt}_s${s}.pt" &
      done
    done
  done
done
wait
log "P1: CEM / TD+CEM / LIP cells done"

# ---------------------------------------------------------- P2 PWM (12 trains)
log "P2: PWM actors, 4 WMs x 3 seeds (trainer defaults, per-base amax)"
for e in "${WMS[@]}"; do
  IFS='|' read -r t wm c5 c1 td am lipt <<< "$e"
  for s in $SEEDS; do
    out=/workspace/actors/pwm_pl_${t}_s${s}.pt
    [ -f "$out" ] && { log "  $t/s$s present"; continue; }
    slot=$(acquire)
    (
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_pwm_ac.py" \
        --cache "$c5" --cache-td "$c1" --wm "$wm" --init-value "$td" \
        --amax "$am" --seed "$s" \
        --out "$out" --out-value "${out%.pt}_value.pt" \
        > "$L/pwmpl_${t}_s${s}.log" 2>&1
      release "$slot"
    ) &
    log "  -> PWM $t/s$s (amax $am)"
  done
done
wait
np=$(ls /workspace/actors/pwm_pl_*_s[0-9].pt 2>/dev/null | wc -l)
log "P2: $np/12 PWM actors trained"

# --------------------------------------------------------- P3 PWM eval (72)
log "P3: PWM evals, 4 WMs x 3 seeds x 3 draws x 2 horizons"
for h in "${HZ[@]}"; do
  IFS='|' read -r hz off bud <<< "$h"
  for e in "${WMS[@]}"; do
    IFS='|' read -r t wm c5 c1 td am lipt <<< "$e"
    for s in $SEEDS; do
      a=/workspace/actors/pwm_pl_${t}_s${s}.pt
      [ -f "$a" ] || { log "  PWM $t/s$s missing, skipping its $hz cells"; continue; }
      for d in $DRAWS; do
        slot=$(acquire)
        ev "$slot" "pl_pwm_${t}_s${s}_${hz}_e${d}" "$wm" "$d" "$off" "$bud" \
           solver=pwm "solver.actor_path=$a" solver.batch_size=10 &
      done
    done
  done
done
wait
log "P3: PWM evals done"

# ------------------------------------------------------------------ P4 card
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
def draws(p):
    vs = [rows.get(f"{p}_e{d}") for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else mean(vs)
def seeded(fmt):
    o = []
    for s in (0, 1, 2):
        m = draws(fmt.format(s=s))
        if m is not None: o.append(m)
    return mean(o) if len(o) == 3 else None
COL = [("lewmpre", "LeWM PRE"), ("lewmpost", "LeWM POST"),
       ("pldmpre", "PLDM PRE"), ("pldmpost", "PLDM POST")]
PLAN = [("latent+CEM", lambda t, h: draws(f"pl_cem_{t}_{h}")),
        ("TD+CEM",     lambda t, h: draws(f"pl_tdcem_{t}_{h}")),
        ("PWM",        lambda t, h: seeded("pl_pwm_%s_s{s}_%s" % (t, h))),
        ("LIP",        lambda t, h: seeded("pl_lip_%s_s{s}_%s" % (t, h)))]
res = {}
for hz, label in (("h25", "h25  (goal t+25, budget 50 -- IN distribution)"),
                  ("h100", "h100 (goal t+100, budget 200 -- OUT of distribution)")):
    print(f"\n=== PLANNER LADDER, {label} ===")
    print("    held-out 8000:10000, EGL, cube. CEM rows: 3 draws, no training")
    print("    seed (no learned actor). PWM/LIP rows: 3 seeds x 3 draws.")
    print(f"  {'planner':12s}" + "".join(f"{n:>12s}" for _, n in COL))
    for pl, fn in PLAN:
        vals = {t: fn(t, hz) for t, _ in COL}
        res[(pl, hz)] = vals
        print(f"  {pl:12s}" + "".join(
            f"{vals[t]:12.2f}" if vals[t] is not None else f"{'-':>12s}" for t, _ in COL))
    print(f"  {'':12s}" + "".join(f"{'':>12s}" for _ in COL))
    print(f"  {'DYNA DELTA':12s}{'LeWM':>12s}{'PLDM':>12s}")
    for pl, _ in PLAN:
        v = res[(pl, hz)]
        def d(a, b):
            return f"{v[b]-v[a]:+12.2f}" if v[a] is not None and v[b] is not None else f"{'-':>12s}"
        print(f"  {pl:12s}{d('lewmpre','lewmpost')}{d('pldmpre','pldmpost')}")
print("\n=== HORIZON TRANSFER (h100 minus h25, same planner and WM) ===")
print(f"  {'planner':12s}" + "".join(f"{n:>12s}" for _, n in COL))
for pl, _ in PLAN:
    a, b = res.get((pl, "h25"), {}), res.get((pl, "h100"), {})
    print(f"  {pl:12s}" + "".join(
        f"{b[t]-a[t]:+12.2f}" if a.get(t) is not None and b.get(t) is not None
        else f"{'-':>12s}" for t, _ in COL))
print()
print("  READING.")
print("  * latent+CEM uses NO learned value and NO learned actor, so its Dyna")
print("    delta is the cleanest read on whether the fine-tune improved the")
print("    WORLD MODEL. If it rises, the gain is real model improvement; if only")
print("    LIP rises, the fine-tune mostly made the model easier to exploit.")
print("  * TD+CEM minus latent+CEM isolates the critic's contribution per WM.")
print("  * LIP minus PWM is what iterative refinement buys over a one-forward-")
print("    pass amortized policy at the same amax.")
print("  * The horizon block is the stress test: every arm was trained and tuned")
print("    at h25, so h100 is out of distribution for all of them. If the")
print("    PLDM-POST / LeWM-POST convergence survives h100 it is a property of")
print("    the policies; if it only holds at h25 it is an operating-point")
print("    artifact. Prior single-cell evidence says h100 does not collapse")
print("    (h200 did), so h100 sits at the edge where differences should show.")
print("  CAVEATS. PWM runs at each base's LIP-tuned amax (its own optimum is")
print("  unexplored) on trainer defaults otherwise. CEM rows have no training")
print("  seed, so their spread is draw-only and not comparable to PWM/LIP seed")
print("  spread. Everything is 3 seeds -- ~3 pts resolution. h100 was never")
print("  tuned for, so low h100 numbers are not evidence of a broken planner.")
print("PLANNER_LADDER_DONE")
PY
log "done"
