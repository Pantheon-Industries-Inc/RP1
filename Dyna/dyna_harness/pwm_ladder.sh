#!/bin/bash
# PWM row of the planner ladder -- the re-run after the a_dim fix.
#
# WHY THIS EXISTS SEPARATELY. planner_ladder.sh landed 120/192 cells: latent+CEM,
# TD+CEM and LIP all completed at both horizons. Its PWM phase died on all 12
# trains in 40 seconds with
#     Conv1d(25, 10, kernel_size=(1,))
#     RuntimeError: expected input[128, 10, 3] to have 25 channels, got 10
# because train_pwm_ac.py derives the action dim as
#     a_dim = int(getattr(c_act, 'action', torch.zeros(1, 10)).shape[-1])
# -- off the latent cache, with a HARDCODED fallback of 10. Our fs5 caches carry
# no action field (LIP takes actions from a separate --h5), so every cube PWM
# run built a 10-dim actor and died in the WM's action encoder. The right value
# is 25: the cube h5 holds raw 5-dim actions and the WM consumes an action
# BLOCK of fs=5, exactly as LIP computes it (train_lip_ac.py:265-266). The bug
# survived because the PWM stack was validated on tworoom; this is its first
# cube run.
#
# patch_pwm_adim.py adds an optional --h5 and derives a_dim the LIP way, into a
# COPY (train_pwm_ac_cube.py) -- the original is a parallel session's committed
# file on a shared branch and stays untouched. Smoke-tested: a_dim=25, trains at
# 7.5 it/s, so 8000 steps is ~18 min rather than the 70 originally budgeted.
#
# SCHEDULING. Runs NOW, concurrently with the LR sweep, on its own 4-slot pool
# rather than queueing behind it -- the LR sweep has ~8h to go and this is ~1.5h
# of work. 4 extra jobs over 8 GPUs is well inside the measured headroom (each
# train is 5-10 GB of 143 GB). Per-GPU MUJOCO_EGL_DEVICE_ID pinning keeps the
# concurrent renders safe.
#
# Cell names match planner_ladder.sh exactly (pl_pwm_<wm>_s<seed>_<hz>_e<draw>)
# so the full 4-planner table can be reprinted from one CSV.
#
# 12 trains + 72 eval cells.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_pwmladder.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
NSLOT=4; NGPU=8; SLOTDIR=/tmp/pwmslots
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
rmdir "$SLOTDIR"/* 2>/dev/null || true
# offset onto the high GPUs so we interleave with the LR sweep instead of
# stacking on the same devices it starts from
gpu_of(){ echo $(( (4 + $1) % NGPU )); }

HZ=("h25|25|50" "h100|100|200")
WMS=(
 "lewmpre|/workspace/models/v2WM|/workspace/caches/v2_tr8000_fs5.pt|/workspace/caches/v2_tr8000_fs1.pt|/workspace/metrics/v2_tr8000_TD.pt|1.6"
 "lewmpost|/workspace/models/dyna_pb_jl|/workspace/caches/pbjl_tr8000_fs5.pt|/workspace/caches/pbjl_tr8000_fs1.pt|/workspace/metrics/pbjl_TD.pt|1.6"
 "pldmpre|/workspace/models/PLDM_OgBench_lewm|/workspace/caches/pldm_tr8000_fs5.pt|/workspace/caches/pldm_tr8000_fs1.pt|/workspace/metrics/pldm_TD.pt|4.5"
 "pldmpost|/workspace/models/dyna_pb_jp|/workspace/caches/pbjp_tr8000_fs5.pt|/workspace/caches/pbjp_tr8000_fs1.pt|/workspace/metrics/pbjp_TD.pt|4.5"
)

log "=== PWM LADDER (pid $$): 12 trains + 72 cells, patched a_dim ==="
[ -f "$P/train_pwm_ac_cube.py" ] || die "patched trainer missing -- run patch_pwm_adim.py"
grep -q "a_dim. from" "$P/train_pwm_ac_cube.py" || die "patch marker absent in trainer"

log "P1: 12 PWM trains (~18 min each, 4-slot pool)"
for e in "${WMS[@]}"; do
  IFS='|' read -r t wm c5 c1 td am <<< "$e"
  for s in $SEEDS; do
    out=/workspace/actors/pwm_pl_${t}_s${s}.pt
    [ -f "$out" ] && { log "  $t/s$s present"; continue; }
    slot=$(acquire); g=$(gpu_of "$slot")
    (
      CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_pwm_ac_cube.py" \
        --cache "$c5" --cache-td "$c1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
        --amax "$am" --seed "$s" \
        --out "$out" --out-value "${out%.pt}_value.pt" \
        > "$L/pwmpl_${t}_s${s}.log" 2>&1
      release "$slot"
    ) &
    log "  -> PWM $t/s$s (amax $am) gpu $g"
  done
done
wait
np=$(ls /workspace/actors/pwm_pl_*_s[0-9].pt 2>/dev/null | wc -l)
log "P1: $np/12 PWM actors"
[ "$np" -gt 0 ] || { tail -20 "$L/pwmpl_lewmpre_s0.log"; die "no PWM actors -- the fix did not hold"; }
grep -h "a_dim. from" "$L/pwmpl_lewmpre_s0.log" | head -1 | sed 's/^/  /' | tee -a "$DRV"

log "P2: 72 eval cells (h25 + h100)"
for h in "${HZ[@]}"; do
  IFS='|' read -r hz off bud <<< "$h"
  for e in "${WMS[@]}"; do
    IFS='|' read -r t wm c5 c1 td am <<< "$e"
    for s in $SEEDS; do
      a=/workspace/actors/pwm_pl_${t}_s${s}.pt
      [ -f "$a" ] || continue
      for d in $DRAWS; do
        slot=$(acquire); g=$(gpu_of "$slot"); nm="pl_pwm_${t}_s${s}_${hz}_e${d}"
        c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { release "$slot"; continue; }
        (
          CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 14400 \
            python3 "$P/eval_wm.py" --config-name cube seed=$d \
            eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
            eval.goal_offset_steps=$off eval.eval_budget=$bud "+eval.ep_range=$EVAL_RANGE" \
            policy="$wm" solver=pwm "solver.actor_path=$a" solver.batch_size=10 \
            output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
          sr=""
          grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
            sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
          flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
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
def draws(p):
    vs = [rows.get(f"{p}_e{d}") for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else mean(vs)
def seeded(fmt):
    o = [draws(fmt.format(s=s)) for s in (0, 1, 2)]
    o = [x for x in o if x is not None]
    return mean(o) if len(o) == 3 else None
COL = [("lewmpre", "LeWM PRE"), ("lewmpost", "LeWM POST"),
       ("pldmpre", "PLDM PRE"), ("pldmpost", "PLDM POST")]
PLAN = [("latent+CEM", lambda t, h: draws(f"pl_cem_{t}_{h}")),
        ("TD+CEM",     lambda t, h: draws(f"pl_tdcem_{t}_{h}")),
        ("PWM",        lambda t, h: seeded("pl_pwm_%s_s{s}_%s" % (t, h))),
        ("LIP",        lambda t, h: seeded("pl_lip_%s_s{s}_%s" % (t, h)))]
for hz, lbl in (("h25", "h25  (goal t+25, budget 50 -- IN distribution)"),
                ("h100", "h100 (goal t+100, budget 200 -- OUT of distribution)")):
    print(f"\n=== FULL PLANNER LADDER, {lbl} ===")
    print(f"  {'planner':12s}" + "".join(f"{n:>12s}" for _, n in COL))
    vv = {}
    for pl, fn in PLAN:
        v = {t: fn(t, hz) for t, _ in COL}; vv[pl] = v
        print(f"  {pl:12s}" + "".join(
            f"{v[t]:12.2f}" if v[t] is not None else f"{'-':>12s}" for t, _ in COL))
    print(f"  {'DYNA DELTA':12s}{'LeWM':>12s}{'PLDM':>12s}")
    for pl, _ in PLAN:
        v = vv[pl]
        f = lambda a, b: (f"{v[b]-v[a]:+12.2f}" if v[a] is not None and v[b] is not None
                          else f"{'-':>12s}")
        print(f"  {pl:12s}{f('lewmpre','lewmpost')}{f('pldmpre','pldmpost')}")
    if vv["LIP"]["lewmpost"] is not None and vv["PWM"]["lewmpost"] is not None:
        print(f"  {'LIP - PWM':12s}" + "".join(
            f"{vv['LIP'][t]-vv['PWM'][t]:12.2f}"
            if vv["LIP"][t] is not None and vv["PWM"][t] is not None else f"{'-':>12s}"
            for t, _ in COL) + "   <- what iterative refinement buys")
print("\n  PWM is a one-forward-pass closed-loop policy: no search, no inner loop.")
print("  LIP minus PWM is the price of LIP's 8 refinement iterations, and the")
print("  honest deployment question -- PWM is orders of magnitude cheaper.")
print("  CAVEAT: PWM runs at each base's LIP-tuned amax; its own optimum is")
print("  unexplored, so a low PWM row understates it. Trainer defaults otherwise.")
print("PWM_LADDER_DONE")
PY
log "done"
