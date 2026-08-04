#!/bin/bash
# GAMMA CURVE for the TD teacher -- filling in a grid that was never a grid.
#
# WHAT WAS ACTUALLY TESTED. Across the whole campaign the teacher's discount has
# taken exactly three values:
#     gamma 1.0     72.0    effective horizon 1/(1-g) = inf
#     gamma 0.995   72.0                                200
#     gamma 0.98    75.3  <- best                        50
# and LIP's OWN --gamma was never swept at all; it defaulted to 1.0 everywhere
# and is 0.98 today only because the upgrade matched it to the teacher.
#
# So 0.98 is the EDGE of the tested range, not an interior optimum. The three
# points are monotone in one direction and nobody ever went below. There is no
# evidence 0.98 is right -- only that it beat the two values nearer 1.0.
#
# WHY IT MATTERS MORE NOW. The teacher upgrade came back a near-null on average
# (+0.33) but with a consistent structure underneath: gamma 0.98 GAINED on both
# un-fine-tuned bases (LeWM +1.33, PLDM +2.67) and LOST on both Dyna-improved
# ones (-0.67, -2.00). Four cells, two independent bases, one sign pattern. The
# reading that fits: discounting compresses far states so capacity concentrates
# near the goal -- a crutch worth having when the latent geometry is poor, and a
# loss of resolution once Dyna has improved it. If that holds, the optimum is
# base-dependent and 0.98 is a compromise between two different ones. This curve
# tests it directly by measuring the optimum separately per world model.
#
# THE GRID. steps and n-step are held FIXED at the adopted 12000 / 50 so this is
# a clean one-factor curve -- the previous comparison confounded gamma with
# steps (base was 1.0@6k, the winner 0.98@12k).
#     gamma   1.0    0.99   0.98*   0.96    0.95    0.90
#     horizon inf    100     50      25      20      10
#   *0.98@12k already exists from the upgrade and is NOT retrained.
# 0.96 is called out because 1/(1-g) = 25 is exactly the plan length
# (horizon 5 x fs 5). 0.98's horizon of 50 lands exactly on --n-step 50, which
# is either why it won or a coincidence; the only way to tell is to move gamma
# off that coincidence, which this does.
#
# NOT TESTED HERE, stated so it is not mistaken for covered:
#   * gamma x n-step coupling (the horizon/backup-span alignment above)
#   * gamma x expand-weight -- gamma rescales plan_cost/plan_disc in the
#     expansion loss (25.0/1.0 at g=1.0 vs 19.83/0.604 at 0.98), and expand is
#     half the winner bundle
#   * teacher gamma vs LIP gamma independently -- they are matched here on a
#     scale-consistency argument, which is an argument and not a measurement
#
# SCHEDULING. Pinned to GPUs 3 and 7 ONLY, which the LR sweep leaves idle, with
# 4 slots (2 per GPU). It therefore does not contend with the running chain and
# does not need to queue behind it.
#
# 5 arms x 4 WMs x 3 seeds = 60 TD trains (~5 min each) + 180 TD+CEM cells. ~3h.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_gamma.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
NSLOT=4; SLOTDIR=/tmp/gslots
GPUS=(3 7 3 7)                       # the two the LR sweep leaves idle
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
rmdir "$SLOTDIR"/* 2>/dev/null || true

# wmtag|wm-dir|fs1 cache
WMS=(
 "lewmpre|/workspace/models/v2WM|/workspace/caches/v2_tr8000_fs1.pt"
 "lewmpost|/workspace/models/dyna_pb_jl|/workspace/caches/pbjl_tr8000_fs1.pt"
 "pldmpre|/workspace/models/PLDM_OgBench_lewm|/workspace/caches/pldm_tr8000_fs1.pt"
 "pldmpost|/workspace/models/dyna_pb_jp|/workspace/caches/pbjp_tr8000_fs1.pt"
)
# tag|gamma   (g098 is reused from the upgrade, not retrained)
ARMS=("g100|1.0" "g099|0.99" "g096|0.96" "g095|0.95" "g090|0.90")

log "=== GAMMA CURVE (pid $$): ${#ARMS[@]} new arms x 4 WMs x 3 seeds, GPUs 3+7 ==="
for e in "${WMS[@]}"; do IFS='|' read -r t wm c1 <<< "$e"
  for f in "$wm/config.json" "$c1"; do [ -e "$f" ] || die "$t: missing $f"; done; done
n98=$(ls /workspace/metrics/up_*_g98s12k_s[0-9].pt 2>/dev/null | wc -l)
log "P0: preflight OK; reusing $n98/12 existing gamma-0.98 teachers"

log "P1: 60 TD trains (steps 12000, n-step 50 fixed; only gamma varies)"
for e in "${WMS[@]}"; do
  IFS='|' read -r t wm c1 <<< "$e"
  for a in "${ARMS[@]}"; do
    arm="${a%%|*}"; g="${a#*|}"
    for s in $SEEDS; do
      out=/workspace/metrics/gc_${t}_${arm}_s${s}.pt
      [ -f "$out" ] && continue
      slot=$(acquire)
      (
        CUDA_VISIBLE_DEVICES=${GPUS[$slot]} timeout 28800 python3 "$P/train_metric.py" \
          --cache "$c1" --learner td --head quasimetric --expectile 0.03 \
          --gamma "$g" --n-step 50 --steps 12000 --seed "$s" \
          --out "$out" > "$L/gc_${t}_${arm}_s${s}.log" 2>&1
        release "$slot" ) &
    done
  done
done
wait
log "P1: $(ls /workspace/metrics/gc_*_s[0-9].pt 2>/dev/null | wc -l)/60 teachers"

log "P2: 180 TD+CEM cells (h25)"
for e in "${WMS[@]}"; do
  IFS='|' read -r t wm c1 <<< "$e"
  for a in "${ARMS[@]}"; do
    arm="${a%%|*}"
    for s in $SEEDS; do
      m=/workspace/metrics/gc_${t}_${arm}_s${s}.pt; [ -f "$m" ] || continue
      for d in $DRAWS; do
        nm="gc_tdcem_${t}_${arm}_s${s}_e${d}"
        c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && continue
        slot=$(acquire); g=${GPUS[$slot]}
        (
          CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 7200 \
            python3 "$P/eval_wm.py" --config-name cube seed=$d \
            eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
            eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
            policy="$wm" solver=cem "+metric=$m" \
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
log "P2: scoring done"

python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
def cell(fmt):
    o = []
    for s in (0, 1, 2):
        vs = [rows.get(fmt.format(s=s, d=d)) for d in (42, 43, 44)]
        if all(v is not None for v in vs): o.append(mean(vs))
    return mean(o) if len(o) == 3 else None
WMS = ["lewmpre", "lewmpost", "pldmpre", "pldmpost"]
# gamma, label, key-builder  (0.98 comes from the upgrade's own naming)
G = [(1.00, "g100", lambda t: cell("gc_tdcem_%s_g100_s{s}_e{d}" % t)),
     (0.99, "g099", lambda t: cell("gc_tdcem_%s_g099_s{s}_e{d}" % t)),
     (0.98, "g98*", lambda t: cell("up_tdcem_%s_g98s12k_s{s}_e{d}" % t)),
     (0.96, "g096", lambda t: cell("gc_tdcem_%s_g096_s{s}_e{d}" % t)),
     (0.95, "g095", lambda t: cell("gc_tdcem_%s_g095_s{s}_e{d}" % t)),
     (0.90, "g090", lambda t: cell("gc_tdcem_%s_g090_s{s}_e{d}" % t))]
print("=== TD-TEACHER GAMMA CURVE (TD+CEM, h25, held-out, 3 seeds x 3 draws) ===")
print("    steps 12000 and n-step 50 fixed -- gamma is the only factor.")
print("    * 0.98 reused from the teacher upgrade (same recipe).")
print(f"  {'gamma':>6s} {'horizon':>8s}" + "".join(f"{t:>11s}" for t in WMS))
tab = {}
for g, lbl, fn in G:
    v = {t: fn(t) for t in WMS}
    tab[g] = v
    h = "inf" if g >= 1.0 else f"{1/(1-g):.0f}"
    print(f"  {g:6.2f} {h:>8s}" + "".join(
        f"{v[t]:11.2f}" if v[t] is not None else f"{'-':>11s}" for t in WMS))
print()
print("  per-world-model optimum:")
for t in WMS:
    got = [(g, tab[g][t]) for g, _, _ in G if tab[g][t] is not None]
    if not got: continue
    bg, bv = max(got, key=lambda kv: kv[1])
    base = tab.get(1.00, {}).get(t)
    d = f"  ({bv-base:+.2f} vs gamma 1.0)" if base is not None else ""
    print(f"    {t:10s} best gamma {bg:.2f} at {bv:.2f}{d}")
print()
print("  PRE vs POST -- does the optimum move after Dyna?")
for pre, post, nm in (("lewmpre", "lewmpost", "LeWM"), ("pldmpre", "pldmpost", "PLDM")):
    a = [(g, tab[g][pre]) for g, _, _ in G if tab[g][pre] is not None]
    b = [(g, tab[g][post]) for g, _, _ in G if tab[g][post] is not None]
    if a and b:
        print(f"    {nm}: PRE prefers {max(a,key=lambda kv:kv[1])[0]:.2f}, "
              f"POST prefers {max(b,key=lambda kv:kv[1])[0]:.2f}")
print()
print("  shared-knob choice (gamma is critic-side, must hold on BOTH bases):")
best, bs = None, -1e9
for g, _, _ in G:
    v = tab[g]
    if any(v[t] is None for t in WMS): continue
    dl = mean([v["lewmpre"], v["lewmpost"]]) - mean([tab[1.00]["lewmpre"], tab[1.00]["lewmpost"]]) \
         if tab.get(1.00) and all(tab[1.00][t] is not None for t in WMS) else None
    dp = mean([v["pldmpre"], v["pldmpost"]]) - mean([tab[1.00]["pldmpre"], tab[1.00]["pldmpost"]]) \
         if tab.get(1.00) and all(tab[1.00][t] is not None for t in WMS) else None
    if dl is None or dp is None: continue
    s = min(dl, dp)
    print(f"    gamma {g:.2f}: d LeWM {dl:+6.2f}  d PLDM {dp:+6.2f}  min {s:+6.2f}")
    if s > bs: best, bs = g, s
if best is not None:
    print(f"    -> shared winner gamma {best:.2f} (min {bs:+.2f})")
print()
print("  READING. If the curve peaks INSIDE the grid, 0.98 was an edge artifact")
print("  and there is real headroom. If PRE and POST peak at different gammas,")
print("  the near-null +0.33 from the upgrade was two opposite real effects")
print("  cancelling, and the discount should be re-selected after every Dyna")
print("  round rather than fixed once.")
print("  NOT COVERED: gamma x n-step (0.98's horizon of 50 sits exactly on")
print("  --n-step 50, possibly not a coincidence), gamma x expand-weight (gamma")
print("  rescales plan_cost/plan_disc in the expansion loss), and teacher-gamma")
print("  vs LIP-gamma independently. 3 seeds, ~3 pts resolution.")
print("GAMMA_CURVE_DONE")
PY
log "done"
