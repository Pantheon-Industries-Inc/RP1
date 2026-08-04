#!/bin/bash
# rh=1 SCREEN THEN CONFIRM. 6000 steps. LeWM. I3 argmin readout.
#
# THE READOUT (I3, eval-only). align_mode=argmin, cap 50: per-chunk V over the
# rollout, chunks_remaining from the step count, j* = argmin over the chunks still
# reachable, recomputed every refinement iteration. execute min(rh, j*+1) -- a
# no-op at rh=1, where it is always 1. rh=1 IS THE POINT OF THIS SWEEP; the rh=5
# column is only a regression check that a changed objective has not broken the
# committed protocol.
#
# TWO OBJECTIVES, CROSSED WITH EVERY HYPER ("the original settings for all these"):
#   min   loss = min_j V(traj_j, zg)   -- the training analogue of an argmin
#         readout, and the cost the task actually rewards: cube fires
#         terminate_at_goal the moment the cube is inside 0.04 m at ANY step.
#   last  loss = V(traj_{H-1}, zg)     -- the production recipe, bit-identical to
#         the pre-patch trainer (verified by unit test).
# So every hyper is measured under both, and last_base reproduces the shipped
# recipe exactly -- it is the anchor the whole sweep hangs off.
#
# STAGE 1 SCREEN: 20 arms x 1 SEED. 20 actors ~= 4.2 h.
# STAGE 2 CONFIRM: the top 3 arms by rh1-argmin get seeds 1 and 2. +6 actors ~= 1.3 h.
#
# HONEST LIMIT OF STAGE 1: one seed is 3 draws, se ~= 3.5 pts. With 20 arms the
# top-3 pick is substantially noise-driven -- that is what a screen is for, and it
# is why stage 2 exists. Only the 3-seed stage-2 numbers are interpretable, and
# even those resolve nothing under ~3 pts. Stage 1 rankings must not be quoted.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_screen.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
WM=/workspace/models/v2WM
TD=/workspace/metrics/up_lewmpre_g98s12k_s0.pt
C5=/workspace/caches/v2_tr8000_fs5.pt; C1=/workspace/caches/v2_tr8000_fs1.pt
STEPS=6000; DRAWS="42 43 44"
NGPU=4; SLOTDIR=/tmp/scrslots
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
      and "--term-index" in h and "--replay-stride" in h)
print("[preflight] argmin readout + trainer levers:", ok)
sys.exit(0 if ok else 1)
PY
for f in "$C5" "$C1" "$TD" "$AH5" "$WM/config.json"; do [ -e "$f" ] || die "missing $f"; done
for i in $(seq 1 90); do pgrep -f "eval_w[m].py" >/dev/null || break; sleep 20; done

BASE="--arch v4 --horizon 5 --iters 8 --n-step 50 --gamma 0.98 --batch 256 \
--amax 1.6 --replay-prob 0.5 --expand-weight 3.0 --freeze-critic-frac 0.5 \
--actor-lr 3e-4 --actor-lr-final 3e-5 --critic-lr 1e-3 --critic-lr-final 1e-4 \
--expectile 0.1 --expectile-final 0.03 --mean-weight 0.1"
HYPERS=(
  "base|"
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
OBJS=("min|--term-index min" "last|--term-index last")

train_one(){ # tag extra seed
  local tag=$1 extra=$2 s=$3
  local out=/workspace/actors/lip4_sc_${tag}_s${s}.pt
  [ -f "$out" ] && { log "  $tag/s$s present"; return 0; }
  local slot; slot=$(acquire)
  ( # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
      --cache "$C5" --cache-td "$C1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
      $BASE --steps "$STEPS" $extra --seed "$s" \
      --out "$out" --out-value "${out%.pt}_value.pt" \
      > "$L/sc_${tag}_s${s}.log" 2>&1
    release "$slot" ) &
  log "  -> $tag/s$s ($extra)"
}

eval_one(){ # tag seed mode
  local tag=$1 s=$2 mode=$3
  local A=/workspace/actors/lip4_sc_${tag}_s${s}.pt
  [ -f "$A" ] || { log "  WARN no actor $A"; return 0; }
  local d
  for d in $DRAWS; do
    local nm="f30_lip_sc${tag}_${mode}_pre_lewm_s${s}_e${d}"
    local cc; cc=$(sc "$nm"); [ -n "$cc" ] && [ "$cc" != FAIL ] && continue
    local EX
    if [ "$mode" = rh1 ]; then
      EX=(plan_config.receding_horizon=1 "+plan_config.eval_budget=50"
          "+solver.align_deadline=true" "+solver.align_mode=argmin")
    else
      EX=(plan_config.receding_horizon=5)
    fi
    local slot; slot=$(acquire)
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
      log "  $nm = ${sr:-FAIL}"; release "$slot" ) &
  done
}

# ============================ STAGE 1: SCREEN, 1 SEED ============================
log "=== STAGE 1: 20 arms x 1 seed @ ${STEPS} steps (2 objectives x 10 hypers)"
for o in "${OBJS[@]}"; do
  IFS='|' read -r otag oflag <<< "$o"
  for hy in "${HYPERS[@]}"; do
    IFS='|' read -r htag hflag <<< "$hy"
    train_one "${otag}_${htag}" "$oflag $hflag" 0
  done
done
wait
na=$(ls /workspace/actors/lip4_sc_*_s0.pt 2>/dev/null | wc -l)
log "STAGE 1: $na/20 actors"
[ "$na" -ge 4 ] || die "too few actors ($na)"

NSLOT=8; rmdir "$SLOTDIR"/* 2>/dev/null || true
log "STAGE 1: evals (rh1 argmin + rh5 regression)"
for o in "${OBJS[@]}"; do
  IFS='|' read -r otag oflag <<< "$o"
  for hy in "${HYPERS[@]}"; do
    IFS='|' read -r htag hflag <<< "$hy"
    eval_one "${otag}_${htag}" 0 rh1
    eval_one "${otag}_${htag}" 0 rh5
  done
done
wait
log "STAGE 1: cells done"

# --------------------- pick the top 3 arms on rh1 (1 seed) ---------------------
TOP=$(python3 - "$SUM" <<'PY'
import sys, statistics as st
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"):
        try: rows[k] = float(v)
        except ValueError: pass
H = ["base","amax12","amax22","exp10","exp60","rp025","rp075","it16","stride1","mw03"]
sc = []
for o in ("min", "last"):
    for h in H:
        t = f"{o}_{h}"
        v = [rows.get("f30_lip_sc%s_rh1_pre_lewm_s0_e%d" % (t, d)) for d in (42, 43, 44)]
        if all(x is not None for x in v):
            sc.append((st.mean(v), t))
sc.sort(reverse=True)
print(" ".join(t for _, t in sc[:3]))
PY
)
log "STAGE 2: top-3 by rh1 (1 seed, screen only): $TOP"
[ -n "$TOP" ] || die "no arms ranked"

# ========================= STAGE 2: CONFIRM, 3 SEEDS =========================
NSLOT=4
for tag in $TOP; do
  case "$tag" in
    min_*)  oflag="--term-index min" ;;
    last_*) oflag="--term-index last" ;;
  esac
  h=${tag#*_}; hflag=""
  for hy in "${HYPERS[@]}"; do
    IFS='|' read -r ht hf <<< "$hy"
    [ "$ht" = "$h" ] && hflag="$hf"
  done
  for s in 1 2; do train_one "$tag" "$oflag $hflag" "$s"; done
done
wait
NSLOT=8; rmdir "$SLOTDIR"/* 2>/dev/null || true
for tag in $TOP; do for s in 1 2; do eval_one "$tag" "$s" rh1; eval_one "$tag" "$s" rh5; done; done
wait
log "STAGE 2: done"

python3 - "$SUM" "$TOP" 2>&1 <<'PY' | tee -a "$DRV"
import sys, statistics as st
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"):
        try: rows[k] = float(v)
        except ValueError: pass
top = sys.argv[2].split() if len(sys.argv) > 2 else []
D = (42, 43, 44)
FL = st.mean([rows["f30_nomove_h25_e%d" % d] for d in D])
def dr(tag, mode, s):
    v = [rows.get("f30_lip_sc%s_%s_pre_lewm_s%d_e%d" % (tag, mode, s, d)) for d in D]
    return None if any(x is None for x in v) else st.mean(v)
def hard(v): return None if v is None else 100.0*(v-FL)/(100.0-FL)
H = [("base","recipe values"),("amax12","amax 1.2"),("amax22","amax 2.2"),
     ("exp10","expand 1.0"),("exp60","expand 6.0"),("rp025","replay-prob 0.25"),
     ("rp075","replay-prob 0.75"),("it16","iters 16"),("stride1","replay-stride 1"),
     ("mw03","mean-weight 0.3")]
print("\n=== STAGE 1 SCREEN: 1 seed, 3 draws -- RANKING IS NOISE-DOMINATED (se ~3.5) ===")
print("  %-20s%11s%11s%11s%11s" % ("hyper", "min rh1", "min rh5", "last rh1", "last rh5"))
f = lambda v: "%11.2f" % v if v is not None else "%11s" % "-"
for h, lab in H:
    print("  %-20s%s%s%s%s" % (lab, f(dr("min_"+h,"rh1",0)), f(dr("min_"+h,"rh5",0)),
                               f(dr("last_"+h,"rh1",0)), f(dr("last_"+h,"rh5",0))))
print("\n  ANCHORS (production actors, 6000 steps, 3 seeds):")
print("    rh5 committed 90.00 | rh1 unaligned 73.33 | rh1 deadline@40 84.89 | rh1 argmin %s"
      % ("%.2f" % st.mean([st.mean([rows["f30_lip_rh1argmin_pre_lewm_s%d_e%d" % (s, d)]
                                   for d in D]) for s in (0, 1, 2)])
         if all("f30_lip_rh1argmin_pre_lewm_s%d_e%d" % (s, d) in rows
                for s in (0, 1, 2) for d in D) else "pending"))
if top:
    print("\n=== STAGE 2 CONFIRM: %s at 3 seeds ===" % ", ".join(top))
    print("  %-20s%11s%7s%11s%11s" % ("arm", "rh1", "sd", "rh1 hard", "rh5"))
    for tag in top:
        per = [dr(tag, "rh1", s) for s in (0, 1, 2)]
        per5 = [dr(tag, "rh5", s) for s in (0, 1, 2)]
        ok = [x for x in per if x is not None]
        if len(ok) < 2: print("  %-20s incomplete" % tag); continue
        m, sd = st.mean(ok), (st.stdev(ok) if len(ok) > 1 else 0.0)
        o5 = [x for x in per5 if x is not None]
        print("  %-20s%11.2f%7.2f%11.1f%11s   [%s]" % (
            tag, m, sd, hard(m), ("%.2f" % st.mean(o5)) if o5 else "-",
            " ".join("%.2f" % x for x in ok)))
    print("\n  Stage-2 numbers are the only quotable ones, and even 3 seeds resolves")
    print("  nothing under ~3 pts. The stage-1 top-3 pick was a screen at 1 seed, so")
    print("  these carry a selection bias: an arm can lead stage 1 on draw noise.")
print("SCREEN_DONE")
PY
log "all done"
