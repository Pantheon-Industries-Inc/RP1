#!/bin/bash
# TIER A: retrain RLP for rh=1 on LeWM. 2x2 factorial, 3 seeds, 2000 steps.
#
#   axis 1  --term-index    last | random
#           which chunk the grad-V feature, the E feature and the loss are read
#           at. 'last' is the pre-2026-08-04 behaviour and is BIT-IDENTICAL to it
#           (ti == H-1 for every row, verified). 'random' samples the graded
#           chunk per example so every PREFIX of the plan must be good -- which is
#           what rh=1 needs, since only chunk 0 executes, and what deploy-time
#           deadline alignment needs, since it reads chunks < H-1.
#
#   axis 2  --replay-stride 5 | 1
#           which mid-task state the --replay-prob 0.5 curriculum banks. Stride 5
#           reduces EXACTLY to the old tr[:, -3:] (verified), i.e. the query an
#           rh=5 replan makes. Stride 1 banks post-one-chunk, the query an rh=1
#           replan makes. Today's recipe has only ever trained on stride 5.
#
# (last, 5) IS THE MATCHED CONTROL: same recipe, same 2000-step budget. Without
# it, "new objective wins" is confounded with "2000 steps != 6000 steps" -- the
# mistake that forced walk-backs earlier in this campaign.
#
# EVAL per cell: rh=1 at alignment target 40 (the swept optimum) AND rh=5 as a
# regression check -- a new objective must not break the committed protocol,
# which is the benchmark standard (LeWorldModel paper, Sec. D).
#
# 12 actors (~17 min each, 1 per GPU, 3 waves ~= 51 min) + 72 cells (~20 min).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_rh1sw.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
WM=/workspace/models/v2WM
TD=/workspace/metrics/up_lewmpre_g98s12k_s0.pt
C5=/workspace/caches/v2_tr8000_fs5.pt; C1=/workspace/caches/v2_tr8000_fs1.pt
EPHI=8000; STEPS=2000; SEEDS="0 1 2"; DRAWS="42 43 44"
NGPU=4; SLOTDIR=/tmp/rh1swslots
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
import subprocess, sys
h = subprocess.run([sys.executable, "/workspace/code/stable-worldmodel/scripts/plan/train_lip_ac.py",
                    "--help"], capture_output=True, text=True).stdout
ok = "--term-index" in h and "--replay-stride" in h
print("[preflight] trainer levers present:", ok)
sys.exit(0 if ok else 1)
PY
for f in "$C5" "$C1" "$TD" "$AH5" "$WM/config.json"; do [ -e "$f" ] || die "missing $f"; done
log "=== TIER A rh=1 retrain sweep (pid $$): 2x2 x 3 seeds @ ${STEPS} steps"

# tag | --term-index | --replay-stride
CFGS=("last5|last|5" "last1|last|1" "rand5|random|5" "rand1|random|1")

log "P1: 12 actor trains (1 per GPU)"
for c in "${CFGS[@]}"; do
  IFS='|' read -r tag ti st <<< "$c"
  for s in $SEEDS; do
    out=/workspace/actors/lip4_sw_${tag}_s${s}.pt
    [ -f "$out" ] && { log "  $tag/s$s present"; continue; }
    slot=$(acquire)
    (
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
        --cache "$C5" --cache-td "$C1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
        --arch v4 --horizon 5 --iters 8 --steps "$STEPS" --n-step 50 --amax 1.6 \
        --batch 256 --gamma 0.98 --replay-prob 0.5 --expand-weight 3.0 \
        --freeze-critic-frac 0.5 --actor-lr 3e-4 --actor-lr-final 3e-5 \
        --critic-lr 1e-3 --critic-lr-final 1e-4 --expectile 0.1 --expectile-final 0.03 \
        --term-index "$ti" --replay-stride "$st" --seed "$s" \
        --out "$out" --out-value "${out%.pt}_value.pt" \
        > "$L/rh1sw_${tag}_s${s}.log" 2>&1
      release "$slot"
    ) &
    log "  -> $tag/s$s (term-index $ti, replay-stride $st)"
  done
done
wait
na=$(ls /workspace/actors/lip4_sw_*_s[0-9].pt 2>/dev/null | wc -l)
log "P1: $na/12 actors"
[ "$na" -ge 4 ] || die "too few actors trained ($na)"

NSLOT=8
rmdir "$SLOTDIR"/* 2>/dev/null || true
log "P2: 72 eval cells -- rh1 @ align target 40, and rh5 regression"
for c in "${CFGS[@]}"; do
  IFS='|' read -r tag ti st <<< "$c"
  for s in $SEEDS; do
    A=/workspace/actors/lip4_sw_${tag}_s${s}.pt
    [ -f "$A" ] || { log "  WARN missing $A, skipping its cells"; continue; }
    for d in $DRAWS; do
      for mode in rh1a40 rh5; do
        nm="f30_lip_sw${tag}_${mode}_pre_lewm_s${s}_e${d}"
        cc=$(sc "$nm"); [ -n "$cc" ] && [ "$cc" != FAIL ] && continue
        if [ "$mode" = rh1a40 ]; then
          EXTRA=(plan_config.receding_horizon=1 "+plan_config.eval_budget=40" "+solver.align_deadline=true")
        else
          EXTRA=(plan_config.receding_horizon=5)
        fi
        slot=$(acquire)
        (
          CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) MUJOCO_EGL_DEVICE_ID=$(( slot % NGPU )) \
          timeout 14400 python3 "$P/eval_wm.py" --config-name cube seed=$d \
            eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
            eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=${EPHI}:10000" \
            "${EXTRA[@]}" policy="$WM" solver=lip "solver.actor_path=$A" \
            output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
          sr=""
          grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
            sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
          flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
          log "  $nm = ${sr:-FAIL}"
          release "$slot"
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
def agg(tag, mode):
    per = []
    for s in S:
        v = [rows.get("f30_lip_sw%s_%s_pre_lewm_s%d_e%d" % (tag, mode, s, d)) for d in D]
        if any(x is None for x in v): return None
        per.append(st.mean(v))
    return st.mean(per), st.stdev(per), per
def hard(v): return None if v is None else 100.0*(v-FL)/(100.0-FL)
LAB = {"last5": "last / stride5  (CONTROL)", "last1": "last / stride1",
       "rand5": "random / stride5", "rand1": "random / stride1"}
print("\n=== TIER A: RLP retrained for rh=1 (2000 steps, 3 seeds, LeWM) ===")
print("  %-28s%11s%7s%11s%11s%7s" % ("config", "rh1@t40", "sd", "rh1 hard", "rh5", "sd"))
res = {}
for tag in ("last5", "last1", "rand5", "rand1"):
    a, b = agg(tag, "rh1a40"), agg(tag, "rh5")
    res[tag] = (a, b)
    f = lambda x: "%11.2f" % x if x is not None else "%11s" % "-"
    g = lambda x: "%7.2f" % x if x is not None else "%7s" % "-"
    print("  %-28s%s%s%s%s%s" % (LAB[tag], f(a[0] if a else None), g(a[1] if a else None),
                                 f(hard(a[0]) if a else None), f(b[0] if b else None),
                                 g(b[1] if b else None)))
ctl = res["last5"][0]
if ctl:
    print("\n  vs the matched control (last/stride5 @ 2000 steps), rh1@t40:")
    for tag in ("last1", "rand5", "rand1"):
        a = res[tag][0]
        if not a: continue
        ds = [a[2][i] - ctl[2][i] for i in range(3)]
        m, sd = st.mean(ds), st.stdev(ds)
        t = m/(sd/3**0.5) if sd > 0 else float("inf")
        print("    %-22s %+6.2f  sd %5.2f  t %6.2f  %d/3 pos" % (LAB[tag], m, sd, t,
                                                                 sum(1 for x in ds if x > 0)))
print("\n  REFERENCE (6000-step production actors, same eval):")
print("    rh5 90.00 | rh1 unaligned 73.33 | rh1 @t40 84.89  <- the 5.11 residual to beat")
print("  CAUTION: 2000 steps, so these are NOT comparable to the 6000-step 90.00.")
print("  Only the within-tier deltas against the control are interpretable.")
print("RH1SW_DONE")
PY
log "done"
