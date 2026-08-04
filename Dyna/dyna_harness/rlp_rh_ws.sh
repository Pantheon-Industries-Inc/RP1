#!/bin/bash
# LeWM RLP: replan cadence x plan initialisation. 2 x 2, PRE-Dyna, 3 seeds x 3 draws.
#
#   rh=5  execute the whole 5-chunk plan, replan once  (what every search arm has used)
#   rh=1  replan every chunk = every world-model step  (k=1, the DINO-WM / PLDM protocol)
#   ws=1  warm_start: leftover chunks seed the next solve  (PlanConfig default)
#   ws=0  no warm start: each solve initialises from scratch
#
# PREDICTION (from the code, to be falsified here): the warm-start axis is an
# EXACT no-op for RLP, so ws=0 and ws=1 must agree cell-for-cell.
#   * LIPSolver.solve() accepts init_action and never uses it -- CEM and Adam both
#     call prepare_init_action(); LIP does not. _proposal_lip builds the plan
#     itself: A = restart_noise*randn then A[::R] = 0.0, and with restarts=1 that
#     zeros the whole tensor (init_mode defaults to "zero").
#   * At rh=5 warm start is inert for a second, independent reason:
#     keep_horizon == horizon leaves `rest` empty, so nothing is ever stored.
# So the actor always receives A_0 = 0, exactly the distribution it was trained on
# ("applied for K iterations from A_0 = 0"). If ws=0 and ws=1 DIFFER, this reading
# is wrong and the discrepancy is the finding.
#
# The override must be verified, not assumed: warm_start is absent from cube.yaml
# (it comes from the PlanConfig dataclass default True), so it is added with
# `+plan_config.warm_start=`. If that silently failed, all four arms would look
# identical and would be misread as "warm start is inert". Stage V greps the
# resolved config out of each output file to prove the flag landed.
#
# PRE-Dyna actors (lip4_re_lewm_exp30_s{0,1,2}) -- the 90.00 reference point.
# 36 cells, ~6-8 min on an idle box.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_rhws.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
WM=/workspace/models/v2WM
EPHI=8000; EVAL_RANGE="${EPHI}:10000"
DRAWS="42 43 44"; SEEDS="0 1 2"
NSLOT=8; NGPU=4; SLOTDIR=/tmp/rhwsslots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
actor(){ echo /workspace/actors/lip4_re_lewm_exp30_s${1}.pt; }

log "=== LeWM RLP: rh x warm_start (pid $$)"
[ "$(python3 -c 'import transformers;print(transformers.__version__)')" = "4.49.0" ] \
  || die "wrong transformers"
for s in $SEEDS; do [ -f "$(actor $s)" ] || die "missing actor $(actor $s)"; done
grep -q "init_mode: str = \"zero\"" "$CODE/stable_worldmodel/solver/lip.py" \
  || log "  NOTE: lip.py init_mode default is no longer literally \"zero\" -- re-check the prediction"
log "P0: preflight OK, 3 PRE actors present"

ev(){ # name rh ws seed draw
  local nm=$1 rh=$2 ws=$3 s=$4 d=$5
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm banked ($c)"; return 0; }
  local slot; slot=$(acquire); local g=$(( slot % NGPU ))
  (
    CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 14400 \
      python3 "$P/eval_wm.py" --config-name cube seed=$d \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
      plan_config.receding_horizon=$rh "+plan_config.warm_start=$ws" \
      policy="$WM" solver=lip "solver.actor_path=$(actor $s)" \
      output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
    sr=""
    grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
      sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
    flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
    log "  $nm = ${sr:-FAIL}"; release "$slot" ) &
}

log "P1: 36 cells -- rh in {1,5} x warm_start in {false,true}"
for rh in 1 5; do
  for ws in false true; do
    tag="rh${rh}ws$([ "$ws" = true ] && echo 1 || echo 0)"
    for s in $SEEDS; do for d in $DRAWS; do
      ev "f30_lip_${tag}_pre_lewm_s${s}_e${d}" "$rh" "$ws" "$s" "$d"
    done; done
  done
done
wait
log "P1: cells done"

# ---- V: prove the overrides actually reached PlanConfig -------------------
log "V: verifying the resolved config in each arm's output file"
for rh in 1 5; do for wsn in 0 1; do
  f=$(ls /workspace/models/f30_lip_rh${rh}ws${wsn}_pre_lewm_s0_e42.txt 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    got=$(grep -E "receding_horizon|warm_start" "$f" | tr -d ' ' | tr '\n' ' ')
    log "  rh${rh}ws${wsn}: $got"
  else
    log "  rh${rh}ws${wsn}: OUTPUT FILE MISSING -- cannot verify override"
  fi
done; done

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
FL = st.mean([x for x in FLv if x is not None]) if all(FLv) else None
def cell(tag, s, d): return rows.get("f30_lip_%s_pre_lewm_s%d_e%d" % (tag, s, d))
def agg(tag):
    per = []
    for s in S:
        v = [cell(tag, s, d) for d in D]
        if any(x is None for x in v): return None
        per.append(st.mean(v))
    return st.mean(per), st.stdev(per), per
def hard(v): return None if (v is None or FL is None) else 100.0*(v-FL)/(100.0-FL)
print("\n=== LeWM RLP (PRE-Dyna): replan cadence x plan initialisation ===")
print("    3 planner seeds x 3 task seeds, h25, held-out 8000:10000")
print("  %-10s%10s%8s%12s   %s" % ("arm", "easy", "sd", "hard", "per-seed"))
res = {}
for rh in (1, 5):
    for wsn, wsl in ((0, "ws=0"), (1, "ws=1")):
        tag = "rh%dws%d" % (rh, wsn)
        a = agg(tag); res[tag] = a
        if a is None: print("  %-10s%10s" % (tag, "incomplete")); continue
        h = hard(a[0])
        print("  %-10s%10.2f%8.2f%12s   [%s]" % (
            tag, a[0], a[1], ("%.1f" % h) if h is not None else "-",
            " ".join("%.2f" % x for x in a[2])))
print("\n=== THE PREDICTION: warm start must be an exact no-op for RLP ===")
for rh in (1, 5):
    a, b = res.get("rh%dws0" % rh), res.get("rh%dws1" % rh)
    if a is None or b is None: print("  rh=%d: incomplete" % rh); continue
    diffs = [cell("rh%dws0" % rh, s, d) - cell("rh%dws1" % rh, s, d) for s in S for d in D]
    mx = max(abs(x) for x in diffs)
    verdict = "CONFIRMED no-op" if mx == 0 else "PREDICTION FALSIFIED (max |diff| %.2f)" % mx
    print("  rh=%d  ws0 %.2f  vs  ws1 %.2f   max per-cell |diff| %.2f  -> %s"
          % (rh, a[0], b[0], mx, verdict))
print("\n=== CADENCE EFFECT (the axis that can actually move) ===")
for wsn in (0, 1):
    a, b = res.get("rh5ws%d" % wsn), res.get("rh1ws%d" % wsn)
    if a is None or b is None: continue
    ds = [st.mean([cell("rh1ws%d" % wsn, s, d) for d in D])
          - st.mean([cell("rh5ws%d" % wsn, s, d) for d in D]) for s in S]
    m, sd = st.mean(ds), st.stdev(ds)
    t = m/(sd/len(ds)**0.5) if sd > 0 else float("inf")
    print("  ws=%d: rh5 %.2f -> rh1 %.2f   paired %+.2f  sd %.2f  t %.2f  %d/3 pos"
          % (wsn, a[0], b[0], m, sd, t, sum(1 for x in ds if x > 0)))
print("\n  Reference: this actor set scores 90.00 +/- 0.67 at rh=5 in the main table,")
print("  so rh5 here is also a reproducibility check on that number.")
print("RHWS_DONE")
PY
log "done"
