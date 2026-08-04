#!/bin/bash
# BUDGET-MATCHED AT 9,000 QUERIES/DECISION -- the Adam side.
#
# WHY. The 3,000-query card (exp30_full's lcem3k/tcem3k arms) showed the two
# costs respond to CEM budget in OPPOSITE directions on LeWM:
#     TD + CEM       85.11 (300x10)  ->  82.44 (300x30)   -2.67
#     latent + CEM   78.67 (300x10)  ->  80.67 (300x30)   +2.00
# i.e. extra CEM iterations exploit the critic's error rather than plan better,
# while pure latent distance (no error to exploit) keeps improving. Reporting the
# whole table at 300x30 = 9,000 therefore needs Adam at 9,000 too, otherwise the
# headline CEM rows sit at 3x Adam's budget and the CEM-vs-Adam gap is confounded
# in the OTHER direction from before.
#
# THE TWO DECOMPOSITIONS. 9,000 queries factors two ways for a gradient solver,
# and they are different experiments:
#     ar  300 x 30   more parallel RESTARTS. Matches CEM's population exactly,
#                    so ar-vs-CEM is purely descent-vs-resampling at identical
#                    sample counts and identical iteration counts.
#     as  100 x 90   more gradient STEPS. Keeps Adam's own restart count and
#                    spends the whole 3x on deeper descent.
# Adam's banked 3,000 point is 100x30, so `as` is the pure iteration-count ladder
# (30 -> 90 steps at fixed restarts) and `ar` is the pure restart ladder
# (100 -> 300 restarts at fixed steps). Together they say whether Adam lost to
# CEM for want of exploration or for want of convergence -- and whether a
# gradient solver on a learned metric shows the same over-exploitation turn that
# CEM does, which is the real question. If `as` DROPS below the 3,000 point on
# TD while holding or rising on latent, critic exploitation is a property of the
# budget and not of the sampler.
#
# CEM at 9,000 is already banked for both bases (f30_lcem_* / f30_tcem_*) and is
# not re-run. This script adds only the four Adam arms.
#
# CONTENTION. exp30_full.sh is running on this box. This script deliberately
# shares its slot directory /tmp/f30slots, so the two drivers share ONE 8-slot
# budget instead of over-subscribing 4 GPUs -- sampling-solver evals core-dump
# under GPU contention and the driver records the crash as a real 0.0, which
# would silently poison exactly the rows we are adding. It also never rmdir's
# the slot dir at startup (exp30_full does, which is how the orphaned fine-tunes
# came to share the box) and it holds no slot while idle.
#
# 48 cells: 2 bases x 2 arms x (3 latent draws + 3 seeds x 3 draws). Adam at 300
# samples is heavier than CEM's forward-only pass, ~4-6 min/cell => ~2-3 h behind
# exp30_full's queue.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_b9k.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"
DRAWS="42 43 44"; SEEDS="0 1 2"
NSLOT=8; NGPU=4; SLOTDIR=/tmp/f30slots      # SHARED with exp30_full on purpose
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
# NO rmdir of $SLOTDIR/* here -- that is what freed exp30_full's held slots.

wm_pre(){  case "$1" in lewm) echo /workspace/models/v2WM ;; *) echo /workspace/models/PLDM_OgBench_lewm ;; esac; }
# critics: the EMA teacher co-trained inside each seed's LIPv4 run, same as exp30
crit(){ if [ "$2" -le 2 ]; then echo /workspace/actors/lip4_re_${1}_exp30_s${2}_value.pt
        else echo /workspace/actors/lip4_bcd_${1}_exp30_s${2}_value.pt; fi; }

log "=== BUDGET 9k, Adam arms (pid $$) transformers $(python3 -c 'import transformers;print(transformers.__version__)')"
[ "$(python3 -c 'import transformers;print(transformers.__version__)')" = "4.49.0" ] \
  || die "wrong transformers -- must match exp30's single provenance"
for nm in lewm pldm; do
  [ -f "$(wm_pre $nm)/config.json" ] || die "$nm: no WM config"
  for s in $SEEDS; do [ -f "$(crit $nm $s)" ] || die "missing critic $(crit $nm $s)"; done
done
grep -q "num_samples" "$CODE/stable_worldmodel/solver/gd.py" || die "GradientSolver lacks num_samples"
log "P0: preflight OK -- 2 WMs, 6 critics, gd.py takes num_samples"

ev(){ # name wm draw solver-overrides...
  local nm=$1 wm=$2 d=$3; shift 3
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm banked ($c)"; return 0; }
  local slot; slot=$(acquire); local g=$(( slot % NGPU ))
  ( CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 14400 \
      python3 "$P/eval_wm.py" --config-name cube seed=$d \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
      policy="$wm" "$@" output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
    sr=""
    grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
      sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
    flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
    log "  $nm = ${sr:-FAIL}"; release "$slot" ) &
}

# arm tag | num_samples | n_steps
ARMS=("ar|300|30" "as|100|90")

log "P1: 48 Adam cells at 9,000 queries -- 2 bases x {300x30, 100x90} x {latent, TD}"
for nm in lewm pldm; do
  wm=$(wm_pre "$nm")
  for a in "${ARMS[@]}"; do
    IFS='|' read -r tag ns st <<< "$a"
    for d in $DRAWS; do
      ev "b9k_ladam${tag}_pre_${nm}_e${d}" "$wm" "$d" \
         solver=adam solver.num_samples=$ns solver.n_steps=$st
    done
    for s in $SEEDS; do
      m=$(crit "$nm" "$s")
      for d in $DRAWS; do
        ev "b9k_tadam${tag}_pre_${nm}_s${s}_e${d}" "$wm" "$d" \
           solver=adam solver.num_samples=$ns solver.n_steps=$st "+metric=$m"
      done
    done
  done
done
wait
log "P1: Adam 9k cells done"

# ------------------------------------------------------------------- the card
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"):
        try: rows[k] = float(v)
        except ValueError: pass
mean = lambda xs: sum(xs)/len(xs)
def sd(xs):
    if len(xs) < 2: return None
    m = mean(xs); return (sum((x-m)**2 for x in xs)/(len(xs)-1))**0.5
def draws(p):
    vs = [rows.get(f"{p}_e{d}") for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else mean(vs)
def seeded(fmt):
    o = [draws(fmt.format(s=s)) for s in (0, 1, 2)]
    o = [v for v in o if v is not None]
    return (mean(o), sd(o)) if len(o) == 3 else (None, None)
def f(v):  return f"{v:.2f}" if v is not None else "   -  "
def fs(v): return f"{v:.2f}" if v is not None else "  -  "

print("\n=== BUDGET LADDER, PRE-Dyna, h25, held-out 8000:10000, 3 seeds x 3 draws ===")
for nm, lab in (("lewm", "LeWM"), ("pldm", "PLDM")):
    print(f"\n  {lab}")
    print(f"    {'planner':22s}{'samples x steps':>17s}{'queries':>9s}{'score':>8s}{'sd':>7s}")
    LAT = [("latent + Adam",      "100 x 30",  3000, draws(f"f30_ladam_pre_{nm}")),
           ("latent + Adam  [ar]", "300 x 30",  9000, draws(f"b9k_ladamar_pre_{nm}")),
           ("latent + Adam  [as]", "100 x 90",  9000, draws(f"b9k_ladamas_pre_{nm}")),
           ("latent + CEM",       "300 x 10",  3000, draws(f"f30_lcem3k_pre_{nm}")),
           ("latent + CEM",       "300 x 30",  9000, draws(f"f30_lcem_pre_{nm}"))]
    for n, sx, q, v in LAT:
        print(f"    {n:22s}{sx:>17s}{q:>9,d}{f(v):>8s}{'-':>7s}")
    TD = [("TD + Adam",       "100 x 30", 3000, seeded("f30_tadam_pre_%s_s{s}" % nm)),
          ("TD + Adam  [ar]", "300 x 30", 9000, seeded("b9k_tadamar_pre_%s_s{s}" % nm)),
          ("TD + Adam  [as]", "100 x 90", 9000, seeded("b9k_tadamas_pre_%s_s{s}" % nm)),
          ("TD + CEM",        "300 x 10", 3000, seeded("f30_tcem3k_pre_%s_s{s}" % nm)),
          ("TD + CEM",        "300 x 30", 9000, seeded("f30_tcem_pre_%s_s{s}" % nm)),
          ("LIPv4 (RLP)",     "~16 tot",    16, seeded("f30_lip_pre_%s_s{s}" % nm))]
    for n, sx, q, (v, s) in TD:
        print(f"    {n:22s}{sx:>17s}{q:>9,d}{f(v):>8s}{fs(s):>7s}")

print("\n=== THE TWO LADDERS (9,000 minus the 3,000 point, same solver+cost) ===")
print(f"  {'':26s}{'restarts 100->300':>19s}{'steps 30->90':>15s}")
for nm, lab in (("lewm", "LeWM"), ("pldm", "PLDM")):
    for cost, b3, bar, bas in (
        ("latent", draws(f"f30_ladam_pre_{nm}"), draws(f"b9k_ladamar_pre_{nm}"), draws(f"b9k_ladamas_pre_{nm}")),
        ("TD",     seeded("f30_tadam_pre_%s_s{s}" % nm)[0],
                   seeded("b9k_tadamar_pre_%s_s{s}" % nm)[0],
                   seeded("b9k_tadamas_pre_%s_s{s}" % nm)[0])):
        d1 = f"{bar-b3:+.2f}" if (bar is not None and b3 is not None) else "  -  "
        d2 = f"{bas-b3:+.2f}" if (bas is not None and b3 is not None) else "  -  "
        print(f"  {lab+' '+cost+' + Adam':26s}{d1:>19s}{d2:>15s}")
    cc = seeded("f30_tcem3k_pre_%s_s{s}" % nm)[0], seeded("f30_tcem_pre_%s_s{s}" % nm)[0]
    lc = draws(f"f30_lcem3k_pre_{nm}"), draws(f"f30_lcem_pre_{nm}")
    for cost, (a, b) in (("latent", lc), ("TD", cc)):
        d = f"{b-a:+.2f}" if (a is not None and b is not None) else "  -  "
        print(f"  {lab+' '+cost+' + CEM  (10->30 it)':26s}{d:>19s}")

print("\n  READING.")
print("  * The CEM 3,000->9,000 rows are the known result: on LeWM the LEARNED")
print("    cost got worse (-2.67) while the LATENT cost got better (+2.00).")
print("    Extra iterations exploit critic error rather than plan better.")
print("  * [as] (30->90 steps at fixed restarts) is the same test for a gradient")
print("    solver. If TD [as] drops while latent [as] holds or rises, the")
print("    over-exploitation turn is a property of BUDGET, not of the sampler,")
print("    and it is the strongest available evidence that the ceiling on")
print("    search-with-a-learned-metric is the metric's error surface.")
print("  * [ar] (100->300 restarts at fixed steps) is the exploration control.")
print("    If [ar] gains where [as] loses, Adam's 3,000 deficit was too few")
print("    restarts, and CEM's advantage was resampling breadth all along.")
print("  * [ar] at 300x30 is CEM's EXACT population and iteration count, so")
print("    CEM minus [ar] is descent-vs-resampling with nothing else different.")
print("  CAVEAT. lr stays at the config default 0.1 for every Adam arm. A 90-step")
print("  run at a step size tuned for 30 may simply be over-descending, which is")
print("  a confound between 'more steps' and 'wrong schedule' that this card")
print("  cannot separate -- a decaying-lr arm would be the follow-up.")
print("BUDGET9K_DONE")
PY
log "done"
