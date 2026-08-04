#!/bin/bash
# STAGES B, C, D of the settled pipeline, chained. Queued behind perbase (A).
# Operating point stays SHARED: replay 0.5 / expand 1.0 / freeze 0.5 / batch 256,
# upgraded 12k teacher, gamma 0.98. Baseline for everything is lr2base.
#
# B  6-SEED CONFIRM of the only 90+ results we have. expand 3.0 hit 90.22
#    (t=3.50) and replay 0.25 hit 90.00 (t=3.46) on LeWM at 3 seeds -- but that
#    was 2 hits out of 10 tests, which is roughly what chance yields at this
#    threshold, and three findings this week (g98, s12k, the PRE/POST split)
#    already died on re-measurement. Adds seeds 3,4,5 to the existing 0,1,2 for
#    the two arms, the deployed baseline, and expand 3.0 on PLDM to confirm the
#    negative there. B is NOT a gate on C or D -- it just tells us whether the
#    one interesting result is real.
#
# C  actor-lr x batch FACTORIAL. batch has never been swept: its only appearance
#    anywhere is an on/off bit in the old 2^4 factorial (freeze 300, seed 0,
#    PLDM only, 256 vs the default 128). actor-lr was swept, but at expand 1.0 /
#    batch 256, and batch x LR is the one interaction with a strong classical
#    prior (bigger batch tolerates a bigger step). Run as a 3x3 factorial rather
#    than sequential OFAT so the interaction is estimable at all -- OFAT scores
#    both null if they only work together, which is exactly how the original
#    replay x expand interaction (+5.50) was nearly missed.
#    NOTE actor-lr is LIP-side (per-base adoptable) but batch is data-side
#    (shared), so the card scores actor-lr per base and batch on the min rule.
#
# D  expand-weight x freeze_at. The effective expand dose is weight x live
#    window -- expand only runs inside critic_step(), which stops at freeze_at.
#    We have moved both separately and never crossed them, so we do not know
#    whether expand 1.0 @ freeze 3000 sits at the peak, and the LeWM 3.0 result
#    might be a dose effect reachable more cheaply by moving freeze instead.
#    This is the only interaction in the whole campaign with a mechanical rather
#    than a selected-from-noise motivation.
#
# 12 + 48 + 30 = 90 trains, 36 + 144 + 90 = 270 cells. ~11h.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_bcd.log
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"
OP="--batch 256 --replay-prob 0.5 --expand-weight 1.0"   # shared operating point
# POD RECYCLED 2026-08-03: the container came back with 4 GPUs, not 8.
# CUDA_VISIBLE_DEVICES=$((slot % NGPU)) would otherwise target devices 4-7
# that no longer exist. 8 slots over 4 H200s is 2 jobs/GPU; each train uses
# 5-10 GB of 143 GB, so memory is not the constraint.
NSLOT=8; NGPU=4; SLOTDIR=/tmp/bcdslots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
rmdir "$SLOTDIR"/* 2>/dev/null || true

log "=== PIPELINE B/C/D (pid $$) ==="
log "P0: waiting for the per-base sweep"
T0=$(date +%s)
until grep -q "PERBASE_SWEEP_DONE" "$L/driver_perbase.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && { log "WARN: proceeding after 12h"; break; }
  sleep 180
done
while pgrep -f "train_lip_a[c]" >/dev/null; do sleep 60; done
[ -f /workspace/td_winner.env ] || die "td_winner.env absent"
# shellcheck disable=SC1091
. /workspace/td_winner.env
log "P0: teacher $TD_ARM, --gamma $TD_GAMMA, operating point: $OP"

wm_of(){   case "$1" in lewm) echo /workspace/models/v2WM ;; *) echo /workspace/models/PLDM_OgBench_lewm ;; esac; }
c5_of(){   case "$1" in lewm) echo /workspace/caches/v2_tr8000_fs5.pt ;; *) echo /workspace/caches/pldm_tr8000_fs5.pt ;; esac; }
c1_of(){   case "$1" in lewm) echo /workspace/caches/v2_tr8000_fs1.pt ;; *) echo /workspace/caches/pldm_tr8000_fs1.pt ;; esac; }
td_of(){   case "$1" in lewm) echo "$TD_LEWMPRE" ;; *) echo "$TD_PLDMPRE" ;; esac; }
amax_of(){ case "$1" in lewm) echo 1.6 ;; *) echo 4.5 ;; esac; }

# train <base> <tag> <seed> <extra-flags...>
train(){
  local nm=$1 tag=$2 s=$3; shift 3
  local out=/workspace/actors/lip4_bcd_${nm}_${tag}_s${s}.pt
  [ -f "$out" ] && return 0
  local slot; slot=$(acquire)
  ( # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
      --cache "$(c5_of "$nm")" --cache-td "$(c1_of "$nm")" --h5 "$AH5" \
      --wm "$(wm_of "$nm")" --init-value "$(td_of "$nm")" \
      --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$(amax_of "$nm")" \
      --actor-lr 3e-4 --actor-lr-final 3e-5 \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --arch v4 --seed "$s" --freeze-critic-frac 0.5 --gamma "$TD_GAMMA" $OP "$@" \
      --out "$out" --out-value "${out%.pt}_value.pt" \
      > "$L/bcd_${nm}_${tag}_s${s}.log" 2>&1
    release "$slot" ) &
}
# ev <base> <tag> <seed> <draw>
ev(){
  local nm=$1 tag=$2 s=$3 d=$4 k="bcdev_${1}_${2}_s${3}_e${4}"
  local c; c=$(sc "$k"); [ -n "$c" ] && [ "$c" != FAIL ] && return 0
  local a=/workspace/actors/lip4_bcd_${nm}_${tag}_s${s}.pt; [ -f "$a" ] || return 0
  local slot; slot=$(acquire); local g=$(( slot % NGPU ))
  (
    CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 7200 \
      python3 "$P/eval_wm.py" --config-name cube seed=$d \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
      policy="$(wm_of "$nm")" solver=lip "solver.actor_path=$a" \
      output.filename="${k}.txt" > "$L/eval_${k}.log" 2>&1
    sr=""
    grep -q "ep_range ${EPHI}:10000" "$L/eval_${k}.log" && \
      sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${k}.log" | tail -1 | grep -oE "[0-9.]+$")
    flock "$SUM.lock" -c "echo '${k},${sr:-FAIL}' >> '$SUM'"
    release "$slot" ) &
}

# ============================== B: 6-seed confirm ==========================
# seeds 3,4,5 only -- 0,1,2 already exist under their original tags and the card
# merges both sets.
log "B: 6-seed confirm, seeds 3-5 (12 trains)"
for s in 3 4 5; do
  train lewm base   "$s"
  train lewm exp30  "$s" --expand-weight 3.0
  train lewm rep025 "$s" --replay-prob 0.25
  train pldm exp30  "$s" --expand-weight 3.0
done
wait
for s in 3 4 5; do for d in $DRAWS; do
  ev lewm base "$s" "$d"; ev lewm exp30 "$s" "$d"
  ev lewm rep025 "$s" "$d"; ev pldm exp30 "$s" "$d"
done; done
wait
log "B: done"

# ====================== C: actor-lr x batch factorial ======================
# 3x3; the (base,256) cell IS lr2base and is not retrained.
log "C: actor-lr x batch factorial, 8 new cells x 2 bases x 3 seeds (48 trains)"
ALRS=("albase|" "alflat|--actor-lr 3e-4 --actor-lr-final 3e-4" "alhi|--actor-lr 6e-4 --actor-lr-final 6e-5")
BATS=("b128|--batch 128" "b256|" "b512|--batch 512")
for nm in lewm pldm; do
  for A in "${ALRS[@]}"; do for B in "${BATS[@]}"; do
    at="${A%%|*}"; af="${A#*|}"; bt="${B%%|*}"; bf="${B#*|}"
    [ "$at" = albase ] && [ "$bt" = b256 ] && continue      # == lr2base
    for s in 0 1 2; do
      # shellcheck disable=SC2086
      train "$nm" "${at}_${bt}" "$s" $af $bf
    done
  done; done
done
wait
for nm in lewm pldm; do
  for A in "${ALRS[@]}"; do for B in "${BATS[@]}"; do
    at="${A%%|*}"; bt="${B%%|*}"
    [ "$at" = albase ] && [ "$bt" = b256 ] && continue
    for s in 0 1 2; do for d in $DRAWS; do ev "$nm" "${at}_${bt}" "$s" "$d"; done; done
  done; done
done
wait
log "C: done"

# ====================== D: expand-weight x freeze_at =======================
# 2x3; the (1.0, 0.5) cell IS lr2base and is not retrained.
log "D: expand x freeze, 5 new cells x 2 bases x 3 seeds (30 trains)"
EXPS=("e10|1.0" "e30|3.0")
FRZS=("f2000|0.3334" "f3000|0.5" "f4800|0.8")
for nm in lewm pldm; do
  for E in "${EXPS[@]}"; do for F in "${FRZS[@]}"; do
    et="${E%%|*}"; ev_="${E#*|}"; ft="${F%%|*}"; fv="${F#*|}"
    [ "$et" = e10 ] && [ "$ft" = f3000 ] && continue        # == lr2base
    for s in 0 1 2; do
      train "$nm" "${et}_${ft}" "$s" --expand-weight "$ev_" --freeze-critic-frac "$fv"
    done
  done; done
done
wait
for nm in lewm pldm; do
  for E in "${EXPS[@]}"; do for F in "${FRZS[@]}"; do
    et="${E%%|*}"; ft="${F%%|*}"
    [ "$et" = e10 ] && [ "$ft" = f3000 ] && continue
    for s in 0 1 2; do for d in $DRAWS; do ev "$nm" "${et}_${ft}" "$s" "$d"; done; done
  done; done
done
wait
log "D: done"

# ================================== cards ==================================
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
def seeds(fmt, ss):
    o = []
    for s in ss:
        vs = [rows.get(fmt.format(s=s, d=d)) for d in (42, 43, 44)]
        if all(v is not None for v in vs): o.append(mean(vs))
    return o
def ttest(a, b):
    if len(a) != len(b) or len(a) < 2: return None
    d = [x-y for x, y in zip(a, b)]; m, s = mean(d), sd(d)
    return m, s, (m/(s/math.sqrt(len(d))) if s > 0 else float("inf")), len(d)-1

print("=== B: 6-SEED CONFIRM of the only 90+ results ===")
print("    3-seed values were: LeWM expand3.0 90.22 (t=3.50), replay0.25 90.00 (t=3.46)")
CRIT = {2: 2.92, 5: 2.57}
for nm, old, new, lab in (
        ("lewm", "reev_lewm_exp30_s{s}_e{d}",  "bcdev_lewm_exp30_s{s}_e{d}",  "LeWM expand 3.0"),
        ("lewm", "reev_lewm_rep025_s{s}_e{d}", "bcdev_lewm_rep025_s{s}_e{d}", "LeWM replay 0.25"),
        ("pldm", "reev_pldm_exp30_s{s}_e{d}",  "bcdev_pldm_exp30_s{s}_e{d}",  "PLDM expand 3.0")):
    arm6 = seeds(old, (0,1,2)) + seeds(new, (3,4,5))
    b6 = seeds("lr2baseev_%s_s{s}_e{d}" % nm, (0,1,2)) + seeds("bcdev_%s_base_s{s}_e{d}" % nm, (3,4,5))
    if len(arm6) < 4 or len(b6) < 4:
        print(f"  {lab:20s} incomplete ({len(arm6)} arm / {len(b6)} base seeds)"); continue
    n = min(len(arm6), len(b6))
    r = ttest(arm6[:n], b6[:n])
    crit = CRIT.get(r[3], 2.57)
    print(f"  {lab:20s} {mean(arm6[:n]):6.2f} vs base {mean(b6[:n]):6.2f}  "
          f"delta {r[0]:+.2f}  sd {r[1]:.2f}  t {r[2]:.2f} (df={r[3]}, crit {crit})  "
          f"{'HOLDS' if abs(r[2]) > crit else 'does NOT hold'}   n={n} seeds")

print("\n=== C: actor-lr x batch (3 seeds; baseline cell = lr2base) ===")
ALR = [("albase", "3e-4->3e-5"), ("alflat", "3e-4 flat"), ("alhi", "6e-4->6e-5")]
BAT = [("b128", "128"), ("b256", "256"), ("b512", "512")]
for nm, lab in (("lewm", "LeWM"), ("pldm", "PLDM")):
    b = seeds("lr2baseev_%s_s{s}_e{d}" % nm, (0,1,2))
    print(f"\n  --- {lab} --- baseline {mean(b):.2f}" if len(b) == 3 else f"\n  --- {lab} ---")
    print(f"  {'actor-lr':14s}" + "".join(f"{d:>11s}" for _, d in BAT))
    grid = {}
    for at, al in ALR:
        cells = []
        for bt, bl in BAT:
            xs = b if (at == "albase" and bt == "b256") else seeds(f"bcdev_{nm}_{at}_{bt}_s{{s}}_e{{d}}", (0,1,2))
            grid[(at, bt)] = xs
            cells.append(f"{mean(xs):11.2f}" if len(xs) == 3 else f"{'-':>11s}")
        print(f"  {al:14s}" + "".join(cells))
    best = max((k for k in grid if len(grid[k]) == 3), key=lambda k: mean(grid[k]), default=None)
    if best: print(f"    best cell: actor-lr {best[0]}, batch {best[1]} = {mean(grid[best]):.2f}")

print("\n=== D: expand-weight x freeze_at (3 seeds; (1.0,3000) = lr2base) ===")
FR = [("f2000", "2000"), ("f3000", "3000"), ("f4800", "4800")]
for nm, lab in (("lewm", "LeWM"), ("pldm", "PLDM")):
    b = seeds("lr2baseev_%s_s{s}_e{d}" % nm, (0,1,2))
    print(f"\n  --- {lab} --- baseline {mean(b):.2f}" if len(b) == 3 else f"\n  --- {lab} ---")
    print(f"  {'expand':10s}" + "".join(f"{'freeze '+d:>13s}" for _, d in FR))
    for et, el in (("e10", "1.0"), ("e30", "3.0")):
        cells = []
        for ft, fl in FR:
            xs = b if (et == "e10" and ft == "f3000") else seeds(f"bcdev_{nm}_{et}_{ft}_s{{s}}_e{{d}}", (0,1,2))
            cells.append(f"{mean(xs):13.2f}" if len(xs) == 3 else f"{'-':>13s}")
        print(f"  {el:10s}" + "".join(cells))
    print("    effective expand dose = weight x (freeze_at/6000)")
print("\n  CAVEATS. C and D are 3 seeds (~3 pts resolution); only B is 6-seed.")
print("  actor-lr is LIP-side (per-base adoptable); batch, expand and freeze are")
print("  shared by directive, so a winner in those must hold on BOTH bases.")
print("BCD_DONE")
PY
log "done"
