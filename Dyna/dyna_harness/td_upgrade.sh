#!/bin/bash
# TD-TEACHER UPGRADE: apply the two sweep winners that were never adopted.
#
# THE MISS. The 18-arm TD sweep ranked, scored by TD+CEM on PLDM:
#     g98    75.3  (+3.3)  <<<   gamma 0.98
#     s12k   74.7  (+2.7)  <<<   12000 steps, sd 0.00 across 3 seeds
#     base   72.0   (0.0)        gamma 1.0, 6000 steps
# Both were reported and neither was applied. Every teacher in the freeze
# ladders, the winner cross, both Dyna runs and the (now killed) LR sweep was
# trained at gamma 1.0 / 6000 steps. No script in the campaign passes --gamma at
# all, and BOTH trainers default it to 1.0.
#
# THE TWO GAMMAS ARE COUPLED. LIP warm-starts its critic from this teacher via
# --init-value, and train_lip_ac.py runs its own critic refinement under its own
# --gamma. If the teacher predicts raw step counts (gamma 1.0, unbounded) and
# the refinement predicts discounted returns (gamma 0.98, saturating at
# 1/(1-g) = 50), the warm start is inconsistent in scale. Whatever wins here
# must therefore be passed to train_lip_ac.py as well -- winner.env carries it.
#
# Gamma also moves the expand term: plan_cost/plan_disc over n_plan = 25
# primitive steps go 25.0/1.0 at gamma 1.0 to 19.83/0.604 at 0.98, so the
# bootstrap in the value-expansion loss weakens. Never tested.
#
# WHY 0.98 SHOULD HELP. At gamma 1.0 the critic must represent distance
# linearly out to the horizon and spends capacity getting far states
# numerically right. At 0.98 targets saturate, far states compress together and
# capacity concentrates near the goal -- where the planner needs resolution.
# Note 1/(1-0.98) = 50 is exactly the --n-step 50 backup span.
#
# ARMS. base is the incumbent; the other two combine winners that were only
# ever measured ONE AT A TIME:
#     base         gamma 1.0   steps 6000    n-step 50
#     g98s12k      gamma 0.98  steps 12000   n-step 50
#     g98s12kn25   gamma 0.98  steps 12000   n-step 25   (n25 was +2.4, 3rd)
#
# SELECTION. gamma/steps/n-step are critic-side, so by directive a winner must
# hold on BOTH bases. Score = min(delta_LeWM, delta_PLDM) against base, where
# each base's value averages its PRE and POST world models. Ties and
# non-improvements fall back to base.
#
# 36 TD trains (~2-4 min each) + 108 TD+CEM cells at h25. ~1h on a 6-slot pool,
# leaving room for the PWM rh=1 evals still running on GPUs 4-7.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_tdupgrade.log; ENVF=/workspace/td_winner.env
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
NSLOT=6; NGPU=8; SLOTDIR=/tmp/tdupslots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
rmdir "$SLOTDIR"/* 2>/dev/null || true
gpu_of(){ echo $(( $1 % 4 )); }   # keep off 4-7 while the PWM rh=1 evals run

# wmtag|wm-dir|fs1 cache
WMS=(
 "lewmpre|/workspace/models/v2WM|/workspace/caches/v2_tr8000_fs1.pt"
 "lewmpost|/workspace/models/dyna_pb_jl|/workspace/caches/pbjl_tr8000_fs1.pt"
 "pldmpre|/workspace/models/PLDM_OgBench_lewm|/workspace/caches/pldm_tr8000_fs1.pt"
 "pldmpost|/workspace/models/dyna_pb_jp|/workspace/caches/pbjp_tr8000_fs1.pt"
)
# arm|flags|gamma-to-pass-to-LIP
ARMS=(
 "base|--gamma 1.0 --n-step 50 --steps 6000|1.0"
 "g98s12k|--gamma 0.98 --n-step 50 --steps 12000|0.98"
 "g98s12kn25|--gamma 0.98 --n-step 25 --steps 12000|0.98"
)

log "=== TD TEACHER UPGRADE (pid $$): 3 arms x 4 WMs x 3 seeds ==="
for e in "${WMS[@]}"; do IFS='|' read -r t wm c1 <<< "$e"
  for f in "$wm/config.json" "$c1"; do [ -e "$f" ] || die "$t: missing $f"; done; done

log "P1: 36 TD trains"
for e in "${WMS[@]}"; do
  IFS='|' read -r t wm c1 <<< "$e"
  for a in "${ARMS[@]}"; do
    arm="${a%%|*}"; rest="${a#*|}"; flags="${rest%%|*}"
    for s in $SEEDS; do
      out=/workspace/metrics/up_${t}_${arm}_s${s}.pt
      [ -f "$out" ] && continue
      slot=$(acquire)
      ( # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$(gpu_of "$slot") timeout 28800 python3 "$P/train_metric.py" \
          --cache "$c1" --learner td --head quasimetric --expectile 0.03 \
          --seed "$s" $flags --out "$out" > "$L/tdup_${t}_${arm}_s${s}.log" 2>&1
        release "$slot" ) &
    done
  done
done
wait
n=$(ls /workspace/metrics/up_*_s[0-9].pt 2>/dev/null | wc -l)
log "P1: $n/36 teachers"
[ "$n" -ge 24 ] || die "too few teachers trained"

log "P2: 108 TD+CEM cells (h25)"
for e in "${WMS[@]}"; do
  IFS='|' read -r t wm c1 <<< "$e"
  for a in "${ARMS[@]}"; do
    arm="${a%%|*}"
    for s in $SEEDS; do
      m=/workspace/metrics/up_${t}_${arm}_s${s}.pt; [ -f "$m" ] || continue
      for d in $DRAWS; do
        nm="up_tdcem_${t}_${arm}_s${s}_e${d}"
        c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && continue
        slot=$(acquire); g=$(gpu_of "$slot")
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

ENVF="$ENVF" python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import os, sys
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
ARMS = ["base", "g98s12k", "g98s12kn25"]
GAM = {"base": "1.0", "g98s12k": "0.98", "g98s12kn25": "0.98"}
WMS = ["lewmpre", "lewmpost", "pldmpre", "pldmpost"]
def cell(t, arm):
    o = []
    for s in (0, 1, 2):
        vs = [rows.get(f"up_tdcem_{t}_{arm}_s{s}_e{d}") for d in (42, 43, 44)]
        if all(v is not None for v in vs): o.append(mean(vs))
    return mean(o) if o else None
print("=== TD TEACHER UPGRADE, scored by TD+CEM (h25, held-out 8000:10000) ===")
print(f"  {'arm':13s}" + "".join(f"{t:>12s}" for t in WMS) + f"{'LeWM avg':>10s}{'PLDM avg':>10s}")
tab = {}
for arm in ARMS:
    v = {t: cell(t, arm) for t in WMS}
    tab[arm] = v
    lw = [v[t] for t in ("lewmpre", "lewmpost") if v[t] is not None]
    pl = [v[t] for t in ("pldmpre", "pldmpost") if v[t] is not None]
    print(f"  {arm:13s}" + "".join(f"{v[t]:12.2f}" if v[t] is not None else f"{'-':>12s}" for t in WMS)
          + (f"{mean(lw):10.2f}" if lw else f"{'-':>10s}")
          + (f"{mean(pl):10.2f}" if pl else f"{'-':>10s}"))
def avg(arm, keys):
    v = [tab[arm][t] for t in keys if tab[arm][t] is not None]
    return mean(v) if v else None
bl, bp = avg("base", ["lewmpre", "lewmpost"]), avg("base", ["pldmpre", "pldmpost"])
best, bestscore = "base", 0.0
print(f"\n  {'arm':13s}{'d LeWM':>9s}{'d PLDM':>9s}{'min':>9s}")
for arm in ARMS[1:]:
    al, ap = avg(arm, ["lewmpre", "lewmpost"]), avg(arm, ["pldmpre", "pldmpost"])
    if None in (al, ap, bl, bp): continue
    dl, dp = al-bl, ap-bp; sc = min(dl, dp)
    print(f"  {arm:13s}{dl:+9.2f}{dp:+9.2f}{sc:+9.2f}")
    if sc > bestscore: best, bestscore = arm, sc
print(f"\n  SELECTED: {best}" + ("" if best != "base" else "  (no arm improved BOTH bases)"))
print(f"    gamma for train_lip_ac.py --gamma : {GAM[best]}")
print(f"    teachers: /workspace/metrics/up_<wm>_{best}_s0.pt")
print("  Selection rule: min(delta_LeWM, delta_PLDM) -- gamma/steps/n-step are")
print("  critic-side, so by directive a winner must hold on BOTH bases.")
print("  Caveat: 3 seeds x 3 draws per cell, ~3 pts resolution.")
with open(os.environ["ENVF"], "w") as f:
    f.write(f"TD_ARM={best}\nTD_GAMMA={GAM[best]}\n")
    for t in WMS:
        f.write(f"TD_{t.upper()}=/workspace/metrics/up_{t}_{best}_s0.pt\n")
print("TD_UPGRADE_DONE")
PY
log "wrote $ENVF"
log "done"
