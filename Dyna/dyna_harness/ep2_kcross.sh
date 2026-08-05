#!/bin/bash
# EPOCH 2 x REPLAN CADENCE: fill in the missing k so every epoch-2 arm exists at
# BOTH k=1 and k=5.
#
# WHY. `receding_horizon` is the number of PLANNED CHUNKS executed before
# replanning. One chunk = action_block = 5 primitive steps = exactly one world-
# model step (frames are captured at step_ct % action_block == 0), and one chunk
# is 5 DISTINCT 5-dim actions -- real action chunking, not frameskip (policy.py
# popleft() applies each action once; the yaml's "# frameskip" comment is wrong).
# So:
#     receding_horizon=1  -> replan every chunk  = every WM step  = k=1
#     receding_horizon=5  -> execute the whole 5-chunk plan       = k=5
# k=1 is the protocol both source papers use (DINO-WM replans "at the next time
# step"; PLDM: "re-plans at every kth interaction", "we use k=1"). k=5 is what
# this repo ships for every env config, and what all our search arms have used.
# PWM alone was overridden to k=1, which made it the ONLY arm in our own tables
# on a different protocol -- the defect this script closes for epoch 2.
#
# NOT A PURE CADENCE ABLATION. PlanConfig.warm_start defaults to True but never
# engages at receding_horizon=5, because keep_horizon == horizon leaves `rest`
# empty. At k=1 the 4 unused chunks DO seed the next solve. So k=1 turns on plan
# reuse for CEM as well as replanning more often. That is the literature-standard
# behaviour, but the two effects are not separated here -- state it, do not claim
# a clean cadence ablation.
#
# WHAT ALREADY EXISTS (from post3_ep2.sh) and is NOT re-run:
#     f30_lip_post2_*     RLP           k=5
#     f30_pwm_post2_*     Reactive PWM  k=1
#     f30_lcem3k_post2_*  latent+CEM    k=5
# WHAT THIS ADDS (42 cells):
#     f30_lipk1_post2_*   RLP           k=1   18
#     f30_pwmk5_post2_*   Reactive PWM  k=5   18
#     f30_lcemk1_post2_*  latent+CEM    k=1    6
#
# Uses the epoch-2 world model and the epoch-2 actors, so it WAITS for
# post3_ep2.sh to print EP2_DONE rather than racing it. Shares /tmp/ep2slots.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_ep2k.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"
DRAWS="42 43 44"; SEEDS="0 1 2"
NSLOT=8; NGPU=4; SLOTDIR=/tmp/ep2slots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }

wm2(){ echo /workspace/models/f30dyna2_$1; }
lip2(){ echo /workspace/actors/lip4_f30post2_${1}_s${2}.pt; }
pwm2(){ echo /workspace/actors/pwm_f30_post2_${1}_s${2}.pt; }

log "=== EPOCH2 x k-CROSS (pid $$): waiting for post3_ep2.sh to finish"
ok=0
for i in $(seq 1 480); do            # up to 8 h at 60 s
  if grep -q "FATAL" "$L/driver_ep2.log" 2>/dev/null; then
    die "post3_ep2.sh reported FATAL -- not proceeding"
  fi
  if grep -q EP2_DONE "$L/driver_ep2.log" 2>/dev/null; then ok=1; break; fi
  sleep 60
done
[ "$ok" = 1 ] || die "timed out waiting for EP2_DONE"
log "post3_ep2.sh complete; verifying epoch-2 assets"
for nm in lewm pldm; do
  [ -f "$(wm2 $nm)/weights_epoch_2.pt" ] || die "$nm: epoch-2 WM missing"
  for s in $SEEDS; do
    [ -f "$(lip2 $nm $s)" ] || die "$nm: RLP actor s$s missing"
  done
done
log "P0: assets OK"

ev(){ # name wm asset solver rh draw
  local nm=$1 wm=$2 asset=$3 slv=$4 rh=$5 d=$6
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm banked ($c)"; return 0; }
  local slot; slot=$(acquire); local g=$(( slot % NGPU )) extra
  case "$slv" in
    lcem) extra="solver=cem solver.n_steps=10" ;;
    lip)  extra="solver=lip solver.actor_path=$asset" ;;
    pwm)  extra="solver=pwm solver.actor_path=$asset solver.batch_size=10" ;;
  esac
  ( # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 14400 \
      python3 "$P/eval_wm.py" --config-name cube seed=$d \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
      plan_config.receding_horizon=$rh \
      policy="$wm" $extra output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
    sr=""
    grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
      sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
    flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
    log "  $nm = ${sr:-FAIL}"; release "$slot" ) &
}

log "P1: 42 cells -- RLP@k1 (18), PWM@k5 (18), latent+CEM@k1 (6)"
for nm in lewm pldm; do
  W=$(wm2 "$nm")
  for d in $DRAWS; do
    ev "f30_lcemk1_post2_${nm}_e${d}" "$W" "" lcem 1 "$d"
  done
  for s in $SEEDS; do
    for d in $DRAWS; do
      ev "f30_lipk1_post2_${nm}_s${s}_e${d}" "$W" "$(lip2 $nm $s)" lip 1 "$d"
      [ -f "$(pwm2 $nm $s)" ] && ev "f30_pwmk5_post2_${nm}_s${s}_e${d}" "$W" "$(pwm2 $nm $s)" pwm 5 "$d"
    done
  done
done
wait
log "P1: done"

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
FLv = [rows.get("f30_nomove_h25_e%d" % d) for d in D]
FL = st.mean([x for x in FLv if x is not None]) if any(FLv) else None
def dr(p):
    v = [rows.get("%s_e%d" % (p, d)) for d in D]
    return None if any(x is None for x in v) else st.mean(v)
def sd(p):
    o = [dr(p % s) for s in S]; o = [x for x in o if x is not None]
    if len(o) != 3: return None, None
    return st.mean(o), st.stdev(o)
def hard(v):
    return None if (v is None or FL is None) else 100.0 * (v - FL) / (100.0 - FL)
# arm -> (k5 pattern, k1 pattern, seeded)
ARMS = [("RLP",           "f30_lip_post2_%s_s%%d",   "f30_lipk1_post2_%s_s%%d",  True),
        ("Reactive PWM",  "f30_pwmk5_post2_%s_s%%d", "f30_pwm_post2_%s_s%%d",    True),
        ("latent+CEM 3k", "f30_lcem3k_post2_%s",     "f30_lcemk1_post2_%s",      False)]
print("\n=== EPOCH 2: replan cadence k=5 (whole plan) vs k=1 (every chunk) ===")
print("    k=1 is the protocol of DINO-WM and PLDM (PLDM: 'we use k=1').")
print("    One chunk = 5 primitive steps = one world-model step.")
for base in ("lewm", "pldm"):
    print("\n  %s" % base.upper())
    print("    %-15s%10s%10s%10s%12s%12s" % ("arm", "k=5", "k=1", "k1-k5", "k=5 hard", "k=1 hard"))
    for lab, p5, p1, seeded in ARMS:
        if seeded:
            a, asd = sd(p5 % base); b, bsd = sd(p1 % base)
        else:
            a = dr(p5 % base); b = dr(p1 % base); asd = bsd = None
        f = lambda v: "%10.2f" % v if v is not None else "%10s" % "-"
        d = "%+10.2f" % (b - a) if (a is not None and b is not None) else "%10s" % "-"
        h5 = "%12.1f" % hard(a) if hard(a) is not None else "%12s" % "-"
        h1 = "%12.1f" % hard(b) if hard(b) is not None else "%12s" % "-"
        print("    %-15s%s%s%s%s%s" % (lab, f(a), f(b), d, h5, h1))
        if seeded and a is not None and b is not None:
            ds = []
            for s in S:
                x, y = dr((p5 % base) % s), dr((p1 % base) % s)
                if x is not None and y is not None: ds.append(y - x)
            if len(ds) > 1:
                m, s_ = st.mean(ds), st.stdev(ds)
                t = m / (s_ / len(ds) ** 0.5) if s_ > 0 else float("inf")
                print("        paired k1-k5  %+.2f  sd %.2f  t %.2f  %d/%d pos"
                      % (m, s_, t, sum(1 for x in ds if x > 0), len(ds)))
print("\n  READING / CAVEATS.")
print("  * If the search arms gain more from k=1 than RLP does, RLP's margin was")
print("    partly an artifact of every arm being denied feedback. That is the")
print("    direction to expect: CEM gets 5x more chances to correct model error.")
print("  * PWM at k=5 is the row that makes our own tables self-consistent --")
print("    it was the ONLY arm previously run at k=1.")
print("  * k=1 also switches ON warm_start (PlanConfig default True, inert at")
print("    k=5 because keep_horizon == horizon). Cadence and plan-reuse are")
print("    therefore NOT separated here.")
print("  * Epoch 1 remains the pre-registered checkpoint; all of this is epoch 2,")
print("    i.e. a post-hoc arm.")
print("EP2K_DONE")
PY
log "done"
