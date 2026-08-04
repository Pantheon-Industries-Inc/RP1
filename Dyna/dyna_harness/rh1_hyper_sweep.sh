#!/bin/bash
# rh=1 HYPER SWEEP at 6000 steps, evaluated with I3 (argmin readout, cap 50).
#
# TRAINING MATCHES THE READOUT. Every arm trains with --term-index min, i.e.
# loss = min_j V(traj_j, zg), which is the training analogue of the argmin
# readout AND the cost the task actually rewards: cube fires terminate_at_goal
# the moment the cube enters the 0.04 m threshold at ANY step, so reaching the
# goal somewhere along the plan is what scores -- not reaching it at chunk H-1.
# The production actors (--term-index last, same recipe, 6000 steps, seeds 0-2)
# are therefore the NOT-trained-for-alignment contrast, already evaluated at
# rh1-argmin, so 'minbase' minus that contrast isolates the objective change.
#
# ALIGNMENT READOUT IS FIXED, NOT SWEPT. Every cell is read with align_mode=argmin and
# cap 50: while t < 25 the remaining budget is >= 5 chunks so lim = 4 and the
# argmin sees the FULL 5-chunk lookahead; from t = 30 the cap bites (lim
# 3,2,1,0) and the reachable set shortens as the deadline closes. No tuned
# target -- the readout picks the lowest-value chunk it can still reach.
#
# DESIGN: one-factor-at-a-time around the production recipe, 3 seeds, 6000 steps.
# The BASELINE NEEDS NO RETRAIN -- lip4_re_lewm_exp30_s{0,1,2} already are the
# recipe at 6000 steps, so the control is the real thing rather than a
# budget-matched stand-in. That is why this sweep can afford 6000 steps.
#
# Held fixed: arch v4, horizon 5, n-step 50, gamma 0.98, freeze-critic-frac 0.5,
# batch 256, LR/expectile schedules, and --term-index last / --replay-stride 0
# (both bit-identical to the pre-patch trainer, verified by unit test).
#
# 27 actors (~50 min each, 1 per GPU, 4 at a time => ~5.6 h) + ~162 cells (~30 min).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_hypsw.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
WM=/workspace/models/v2WM
TD=/workspace/metrics/up_lewmpre_g98s12k_s0.pt
C5=/workspace/caches/v2_tr8000_fs5.pt; C1=/workspace/caches/v2_tr8000_fs1.pt
STEPS=6000; SEEDS="0 1 2"; DRAWS="42 43 44"
NGPU=4; SLOTDIR=/tmp/hypswslots
mkdir -p "$SLOTDIR" "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
NSLOT=4
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 10; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
cd "$CODE"

python3 - <<'PY' || exit 1
import subprocess, sys, inspect
sys.path.insert(0, "/workspace/code/stable-worldmodel")
from stable_worldmodel.solver import LIPSolver
h = subprocess.run([sys.executable, "/workspace/code/stable-worldmodel/scripts/plan/train_lip_ac.py",
                    "--help"], capture_output=True, text=True).stdout
ok = ("align_mode" in inspect.signature(LIPSolver.__init__).parameters
      and "--replay-stride" in h and "--term-index" in h)
print("[preflight] argmin + trainer levers present:", ok)
sys.exit(0 if ok else 1)
PY
for f in "$C5" "$C1" "$TD" "$AH5" "$WM/config.json"; do [ -e "$f" ] || die "missing $f"; done
# let any in-flight evals clear so the trains get whole GPUs
for i in $(seq 1 60); do pgrep -f "eval_w[m].py" >/dev/null || break; sleep 20; done
log "=== rh=1 HYPER SWEEP @ ${STEPS} steps, I3 argmin readout, OFAT x 3 seeds"

# tag | extra training flags (one change from the recipe)
ARMS=(
  "minbase|"
  "amax12|--amax 1.2"
  "amax22|--amax 2.2"
  "exp10|--expand-weight 1.0"
  "exp60|--expand-weight 6.0"
  "rp025|--replay-prob 0.25"
  "rp075|--replay-prob 0.75"
  "it16|--iters 16"
  "stride1|--replay-stride 1"
  "mw03|--mean-weight 0.3"
)
BASE="--arch v4 --horizon 5 --iters 8 --n-step 50 --gamma 0.98 --batch 256 \
--amax 1.6 --replay-prob 0.5 --expand-weight 3.0 --freeze-critic-frac 0.5 \
--actor-lr 3e-4 --actor-lr-final 3e-5 --critic-lr 1e-3 --critic-lr-final 1e-4 \
--expectile 0.1 --expectile-final 0.03 --mean-weight 0.1 --term-index min"

log "P1: 27 trains (later flags win, so each arm overrides exactly one BASE value)"
for e in "${ARMS[@]}"; do
  IFS='|' read -r tag extra <<< "$e"
  for s in $SEEDS; do
    out=/workspace/actors/lip4_hs_${tag}_s${s}.pt
    [ -f "$out" ] && { log "  $tag/s$s present"; continue; }
    slot=$(acquire)
    ( # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
        --cache "$C5" --cache-td "$C1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
        $BASE --steps "$STEPS" $extra --seed "$s" \
        --out "$out" --out-value "${out%.pt}_value.pt" \
        > "$L/hs_${tag}_s${s}.log" 2>&1
      release "$slot" ) &
    log "  -> $tag/s$s ($extra)"
  done
done
wait
na=$(ls /workspace/actors/lip4_hs_*_s[0-9].pt 2>/dev/null | wc -l)
log "P1: $na/27 actors"
[ "$na" -ge 3 ] || die "too few actors ($na)"

NSLOT=8; rmdir "$SLOTDIR"/* 2>/dev/null || true
log "P2: evals -- rh1 @ I3 argmin, and rh5 regression"
for e in "${ARMS[@]}"; do
  IFS='|' read -r tag extra <<< "$e"
  for s in $SEEDS; do
    A=/workspace/actors/lip4_hs_${tag}_s${s}.pt
    [ -f "$A" ] || { log "  WARN missing $A"; continue; }
    for d in $DRAWS; do
      for mode in rh1argmin rh5; do
        nm="f30_lip_hs${tag}_${mode}_pre_lewm_s${s}_e${d}"
        cc=$(sc "$nm"); [ -n "$cc" ] && [ "$cc" != FAIL ] && continue
        if [ "$mode" = rh1argmin ]; then
          EX=(plan_config.receding_horizon=1 "+plan_config.eval_budget=50"
              "+solver.align_deadline=true" "+solver.align_mode=argmin")
        else
          EX=(plan_config.receding_horizon=5)
        fi
        slot=$(acquire)
        (
          CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) MUJOCO_EGL_DEVICE_ID=$(( slot % NGPU )) \
          timeout 14400 python3 "$P/eval_wm.py" --config-name cube seed=$d \
            eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
            eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=8000:10000" \
            "${EX[@]}" policy="$WM" solver=lip "solver.actor_path=$A" \
            output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
          sr=""
          grep -q "ep_range 8000:10000" "$L/eval_${nm}.log" && \
            sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
          flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
          log "  $nm = ${sr:-FAIL}"; release "$slot"
        ) &
      done
    done
  done
done
wait
log "P2: cells done"

python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, statistics as st
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"):
        try: rows[k] = float(v)
        except ValueError: pass
D = (42, 43, 44); S = (0, 1, 2)
FL = st.mean([rows["f30_nomove_h25_e%d" % d] for d in D])
def agg(pat):
    per = []
    for s in S:
        v = [rows.get(pat % (s, d)) for d in D]
        if any(x is None for x in v): return None
        per.append(st.mean(v))
    return st.mean(per), st.stdev(per), per
def hard(v): return None if v is None else 100.0*(v-FL)/(100.0-FL)
ARMS = [("amax12", "amax 1.2"), ("amax22", "amax 2.2"), ("exp10", "expand 1.0"),
        ("exp60", "expand 6.0"), ("rp025", "replay-prob 0.25"), ("rp075", "replay-prob 0.75"),
        ("it16", "iters 16"), ("stride1", "replay-stride 1"), ("mw03", "mean-weight 0.3")]
ARMS = [("minbase", "min objective only")] + ARMS
ctl1 = agg("f30_lip_rh1argmin_pre_lewm_s%d_e%d")
ctl5 = agg("f30_lip_rh5ws0_pre_lewm_s%d_e%d")
print("\n=== rh=1 HYPER SWEEP, 6000 steps, I3 argmin readout (cap 50) ===")
print("  %-20s%11s%7s%10s%11s%7s" % ("arm", "rh1 argmin", "sd", "hard", "rh5", "sd"))
f = lambda x: "%11.2f" % x if x is not None else "%11s" % "-"
g = lambda x: "%7.2f" % x if x is not None else "%7s" % "-"
h = lambda x: "%10.1f" % x if x is not None else "%10s" % "-"
print("  %-20s%s%s%s%s%s   <- term-index LAST, not trained for the readout" % (
    "production recipe", f(ctl1[0] if ctl1 else None), g(ctl1[1] if ctl1 else None),
    h(hard(ctl1[0]) if ctl1 else None), f(ctl5[0] if ctl5 else None),
    g(ctl5[1] if ctl5 else None)))
best = (None, -1)
for tag, lab in ARMS:
    a = agg("f30_lip_hs%s_rh1argmin_pre_lewm_s%%d_e%%d" % tag)
    b = agg("f30_lip_hs%s_rh5_pre_lewm_s%%d_e%%d" % tag)
    if a and a[0] > best[1]: best = (lab, a[0])
    print("  %-20s%s%s%s%s%s" % (lab, f(a[0] if a else None), g(a[1] if a else None),
                                 h(hard(a[0]) if a else None), f(b[0] if b else None),
                                 g(b[1] if b else None)))
if ctl1:
    print("\n  paired vs control at rh1-argmin (same seeds):")
    for tag, lab in ARMS:
        a = agg("f30_lip_hs%s_rh1argmin_pre_lewm_s%%d_e%%d" % tag)
        if not a: continue
        ds = [a[2][i] - ctl1[2][i] for i in range(3)]
        m, sd = st.mean(ds), st.stdev(ds)
        t = m/(sd/3**0.5) if sd > 0 else float("inf")
        flag = "  *" if abs(t) > 4.30 else ""      # t crit, df=2, p<0.05
        print("    %-20s %+6.2f  sd %5.2f  t %6.2f  %d/3 pos%s" % (
            lab, m, sd, t, sum(1 for x in ds if x > 0), flag))
    print("    (* = |t| > 4.30, the df=2 critical value; anything else is NOT resolved")
    print("     at 3 seeds, and with 9 arms expect ~0.5 false positives at p<0.05)")
print("\n  REFERENCE: rh5 committed 90.00 | rh1 unaligned 73.33 | rh1 deadline@40 84.89")
print("HYPSW_DONE")
PY
log "done"
