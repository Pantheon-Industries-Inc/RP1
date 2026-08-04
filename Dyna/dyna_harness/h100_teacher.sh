#!/bin/bash
# h100-APPROPRIATE TD TEACHER: gamma x n-step, scored at h100.
#
# THE DEFECT. Every h100 number in this campaign was produced by a critic tuned
# for h25. The teacher recipe is gamma 0.98 / n-step 50, and the discount's
# effective horizon is 1/(1-gamma) = 50 primitive steps. h25 is goal-at-t+25
# with a 50-step budget, so 50 covers the task exactly -- which is very likely
# WHY 0.98 won there, and is the coincidence already flagged (1/(1-0.98) lands
# exactly on --n-step 50).
#
# h100 is goal-at-t+100 with a 200-step budget. At gamma 0.98 the TD target
# saturates at 50, so the critic cannot distinguish 80 steps from 150 -- both
# compress to ~50. That is not a tuning inefficiency, it is a representational
# ceiling BELOW the task's own scale. n-step 50 compounds it: the backup span
# covers a quarter of the budget, so bootstrapping does the rest.
#
# WHAT THIS INVALIDATES (stated so it is not quietly inherited):
#   * the 6-point gamma curve was scored by TD+CEM at h25 ONLY, so its "null"
#     is an h25 null and says nothing about h100
#   * the planner ladder's entire h100 column -- TD+CEM, LIP, PWM -- ran on
#     gamma-0.98 teachers
#   * PLDM POST LIP h100 = 80.22, the one negative Dyna cell in the whole
#     campaign, now has a SECOND candidate explanation (critic ceiling) beside
#     the collection-horizon one, and they are not separable from what we have
#   * dyna_h100.sh built its post-Dyna teacher from TDFLAGS = the h25 winner
#
# THE GRID. steps fixed at the adopted 12000; only gamma and the backup span
# move.
#     gamma    1.00      0.995     0.99      0.98
#     horizon  unbounded 200       100       50 (current)
#     n-step   50 | 200
# 0.995 matches the h100 BUDGET, 0.99 matches the goal OFFSET, 1.0 removes the
# ceiling entirely. Scored by TD+CEM at h100 (goal_offset 100, budget 200) --
# the whole point is that h25 scoring cannot see this.
#
# PREDICTION, recorded before running: gamma 1.0 or 0.995 should beat 0.98 at
# h100 by a visible margin. This differs from the four nulls this week in kind:
# those refined an already-reasonable setting, whereas 0.98 is provably out of
# range for the task. If it comes back flat anyway, that is a strong result --
# it would mean the quasimetric's ranking survives saturation, and the whole
# discount question is closed for good.
#
# SCOPE. Phase 1-2 sweep the two PRE bases, which is what an h100 Dyna inits
# from and measures PRE on. Phase 3 then re-scores the two POST world models
# with the winning recipe, which repairs the planner ladder's h100 TD+CEM row.
#
# 42 new teachers (gamma0.98/n50 is reused from the upgrade) + 144 h100 cells
# + 18 POST cells. h100 cells carry a 4x budget so they cost ~7 min each. ~3h.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_h100td.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
OFF=100; BUD=200                      # h100: the horizon we are tuning FOR
# POD RECYCLED 2026-08-03: the container came back with 4 GPUs, not 8.
# CUDA_VISIBLE_DEVICES=$((slot % NGPU)) would otherwise target devices 4-7
# that no longer exist. 8 slots over 4 H200s is 2 jobs/GPU; each train uses
# 5-10 GB of 143 GB, so memory is not the constraint.
NSLOT=8; NGPU=4; SLOTDIR=/tmp/h1tdslots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
rmdir "$SLOTDIR"/* 2>/dev/null || true

PRE=(
 "lewmpre|/workspace/models/v2WM|/workspace/caches/v2_tr8000_fs1.pt"
 "pldmpre|/workspace/models/PLDM_OgBench_lewm|/workspace/caches/pldm_tr8000_fs1.pt"
)
POST=(
 "lewmpost|/workspace/models/dyna_pb_jl|/workspace/caches/pbjl_tr8000_fs1.pt"
 "pldmpost|/workspace/models/dyna_pb_jp|/workspace/caches/pbjp_tr8000_fs1.pt"
)
# tag|gamma|n-step
ARMS=(
 "g098n50|0.98|50" "g098n200|0.98|200"
 "g099n50|0.99|50" "g099n200|0.99|200"
 "g0995n50|0.995|50" "g0995n200|0.995|200"
 "g100n50|1.0|50" "g100n200|1.0|200"
)

log "=== h100 TEACHER SWEEP (pid $$): 8 arms x 2 PRE bases x 3 seeds, scored at h100 ==="
log "P0: waiting for pipeline B/C/D"
T0=$(date +%s)
until grep -q "BCD_DONE" "$L/driver_bcd.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 57600 ] && { log "WARN: proceeding after 16h"; break; }
  sleep 300
done
while pgrep -f "train_lip_a[c]" >/dev/null; do sleep 60; done
log "P0: box available"

# ---------------------------------------------------------------- P1 teachers
log "P1: TD teachers (steps 12000 fixed; gamma and n-step vary)"
for e in "${PRE[@]}"; do
  IFS='|' read -r t wm c1 <<< "$e"
  for a in "${ARMS[@]}"; do
    IFS='|' read -r arm g n <<< "$a"
    for s in $SEEDS; do
      out=/workspace/metrics/h1td_${t}_${arm}_s${s}.pt
      [ -f "$out" ] && continue
      slot=$(acquire)
      ( CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_metric.py" \
          --cache "$c1" --learner td --head quasimetric --expectile 0.03 \
          --gamma "$g" --n-step "$n" --steps 12000 --seed "$s" \
          --out "$out" > "$L/h1td_${t}_${arm}_s${s}.log" 2>&1
        release "$slot" ) &
    done
  done
done
wait
nt=$(ls /workspace/metrics/h1td_*_s[0-9].pt 2>/dev/null | wc -l)
log "P1: $nt/48 teachers"
[ "$nt" -ge 30 ] || die "too few teachers"

# ------------------------------------------------------- P2 score at h100
evc(){ # slot name wm metric
  local slot=$1 nm=$2 wm=$3 m=$4 d=$5 g=$(( $1 % NGPU ))
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { release "$slot"; return 0; }
  CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 14400 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=$OFF eval.eval_budget=$BUD "+eval.ep_range=$EVAL_RANGE" \
    policy="$wm" solver=cem "+metric=$m" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  $nm = ${sr:-FAIL}"
  release "$slot"
}
log "P2: 144 TD+CEM cells at h100 (goal_offset $OFF, budget $BUD, ~7 min each)"
for e in "${PRE[@]}"; do
  IFS='|' read -r t wm c1 <<< "$e"
  for a in "${ARMS[@]}"; do
    arm="${a%%|*}"
    for s in $SEEDS; do
      m=/workspace/metrics/h1td_${t}_${arm}_s${s}.pt; [ -f "$m" ] || continue
      for d in $DRAWS; do
        slot=$(acquire); evc "$slot" "h1tdcem_${t}_${arm}_s${s}_e${d}" "$wm" "$m" "$d" &
      done
    done
  done
done
wait
log "P2: scoring done"

# ------------------- P3 winner -> re-score the POST world models
WIN=$(python3 - "$SUM" <<'PY'
import sys
rows={}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k,v=ln.strip().split(",")[:2]
    if v not in ("","FAIL"): rows[k]=float(v)
m=lambda x: sum(x)/len(x)
ARMS=["g098n50","g098n200","g099n50","g099n200","g0995n50","g0995n200","g100n50","g100n200"]
def cell(t,a):
    o=[]
    for s in (0,1,2):
        vs=[rows.get(f"h1tdcem_{t}_{a}_s{s}_e{d}") for d in (42,43,44)]
        if all(v is not None for v in vs): o.append(m(vs))
    return m(o) if o else None
best,bs=None,-1e9
base={t:cell(t,"g098n50") for t in ("lewmpre","pldmpre")}
for a in ARMS:
    v={t:cell(t,a) for t in ("lewmpre","pldmpre")}
    if any(v[t] is None or base[t] is None for t in v): continue
    s=min(v["lewmpre"]-base["lewmpre"], v["pldmpre"]-base["pldmpre"])
    if s>bs: best,bs=a,s
print(best or "g098n50")
PY
)
log "P3: h100 winner = $WIN -- re-scoring the POST world models with it"
declare -A GN=( [g098n50]="0.98 50" [g098n200]="0.98 200" [g099n50]="0.99 50" \
  [g099n200]="0.99 200" [g0995n50]="0.995 50" [g0995n200]="0.995 200" \
  [g100n50]="1.0 50" [g100n200]="1.0 200" )
read -r WG WN <<< "${GN[$WIN]}"
for e in "${POST[@]}"; do
  IFS='|' read -r t wm c1 <<< "$e"
  for s in $SEEDS; do
    out=/workspace/metrics/h1td_${t}_${WIN}_s${s}.pt
    [ -f "$out" ] && continue
    slot=$(acquire)
    ( CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_metric.py" \
        --cache "$c1" --learner td --head quasimetric --expectile 0.03 \
        --gamma "$WG" --n-step "$WN" --steps 12000 --seed "$s" \
        --out "$out" > "$L/h1td_${t}_${WIN}_s${s}.log" 2>&1
      release "$slot" ) &
  done
done
wait
for e in "${POST[@]}"; do
  IFS='|' read -r t wm c1 <<< "$e"
  for s in $SEEDS; do
    m=/workspace/metrics/h1td_${t}_${WIN}_s${s}.pt; [ -f "$m" ] || continue
    for d in $DRAWS; do
      slot=$(acquire); evc "$slot" "h1tdcem_${t}_${WIN}_s${s}_e${d}" "$wm" "$m" "$d" &
    done
  done
done
wait
log "P3: POST re-scored"

WIN="$WIN" python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import os, sys
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
m = lambda x: sum(x)/len(x)
def cell(t, a):
    o = []
    for s in (0, 1, 2):
        vs = [rows.get(f"h1tdcem_{t}_{a}_s{s}_e{d}") for d in (42, 43, 44)]
        if all(v is not None for v in vs): o.append(m(vs))
    return m(o) if o else None
G = [("1.0", "unbdd", "g100"), ("0.995", "200", "g0995"), ("0.99", "100", "g099"),
     ("0.98", "50", "g098")]
print("=== h100 TD TEACHER: gamma x n-step, scored by TD+CEM AT h100 ===")
print("    goal_offset 100 / budget 200. steps 12000 fixed.")
print("    horizon = 1/(1-gamma) in primitive steps; 0.98/n50 is the incumbent,")
print("    tuned for h25 and saturating at 50 -- below this task's own scale.")
for t, lab in (("lewmpre", "LeWM PRE"), ("pldmpre", "PLDM PRE")):
    b = cell(t, "g098n50")
    print(f"\n  --- {lab} --- incumbent (0.98 / n50) {b if b is None else round(b,2)}")
    print(f"  {'gamma':>7s} {'horizon':>8s} {'n-step 50':>11s} {'n-step 200':>12s}")
    for g, h, pre in G:
        v50, v200 = cell(t, pre+"n50"), cell(t, pre+"n200")
        f = lambda v: f"{v:11.2f}" if v is not None else f"{'-':>11s}"
        f2 = lambda v: f"{v:12.2f}" if v is not None else f"{'-':>12s}"
        print(f"  {g:>7s} {h:>8s} {f(v50)} {f2(v200)}")
print(f"\n  selected for h100 (min rule over both bases): {os.environ['WIN']}")
print("\n  POST world models re-scored with it, vs the h25-tuned value that")
print("  produced the planner ladder's h100 row:")
for t, old, lab in (("lewmpost", 82.00, "LeWM POST"), ("pldmpost", 79.33, "PLDM POST")):
    v = cell(t, os.environ["WIN"])
    if v is None: print(f"    {lab}: incomplete"); continue
    print(f"    {lab}: {v:.2f}   (h25-tuned teacher gave {old:.2f}, delta {v-old:+.2f})")
print("\n  READING. A large gain confirms the h100 column was measured through a")
print("  critic ceiling, and PLDM POST LIP's -2.89 has a second cause beside the")
print("  collection horizon. A flat result closes the discount question for good:")
print("  it would mean the quasimetric's RANKING survives saturation even when its")
print("  magnitudes do not, which is all CEM needs.")
print("  CAVEAT: 3 seeds, and CEM rows have no training seed so their spread is")
print("  draw-only. LIP at h100 is NOT re-measured here -- that needs new actors.")
print("H100_TD_DONE")
PY
log "done"
