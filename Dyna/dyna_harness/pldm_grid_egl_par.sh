#!/bin/bash
# PLDM grid, EGL, 8-WAY PARALLEL EVALS -- with a validation gate first.
#
# The campaign rule is "evals MUST run sequentially" (3-way -> SIGABRT, 2-way
# silently corrupted a baseline by 8 pts). That rule was written BEFORE the
# parallelization audit found the mechanism: with MUJOCO_EGL_DEVICE_ID unset,
# every EGL render context lands on physical GPU 0 no matter what
# CUDA_VISIBLE_DEVICES says. So N parallel evals were really N processes
# fighting over one GPU's EGL resources -- which is exactly what crashes and
# corrupts. Pinning BOTH vars to the same physical index per worker gives each
# eval its own GPU for compute and rendering, and the contention disappears.
#
# That is a hypothesis about a rule that once cost 8 points, so P0 TESTS IT:
# three latent+CEM reference cells whose serial values are already known
# (42/43/44 = 64/66/68, and 66/68 were just reproduced exactly on this pod)
# are re-run in parallel. Evals are deterministic given the seed, so parallel
# values must match EXACTLY. Any mismatch aborts before a single grid cell is
# measured -- silent corruption is the failure mode being guarded against, and
# it does not announce itself.
#
# Also fixes the TD+CEM reference: the bar is `solver=cem "+metric=<TD>"`
# (a top-level override), not solver.cost_path -- CEM's config has no such key.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_grid_par.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; NGPU=8
mkdir -p "$R" "$L"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

# ---- one eval, pinned to a physical GPU for BOTH compute and EGL rendering
ev(){ # gpu name extra-hydra-args...
  local gpu=$1 nm=$2 d; shift 2
  d=$(echo "$nm" | grep -oE "e4[234]$" | tr -d e)
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$PLDM" "$@" output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
}

log "=== PLDM GRID, PARALLEL EGL (pid $$) ==="
for f in "$QF1" "$QF5" "$QTD" "$AH5" "$PLDM/weights.pt"; do [ -e "$f" ] || die "missing $f"; done
log "  actors: $(ls /workspace/actors/ | grep -cE 'lip4_pldm_mw.*_s0\.pt$')/12"

# ================= P0 PARALLEL-SAFETY GATE (do not skip) =================
log "P0: parallel-safety validation -- 3 known cells re-run concurrently"
# Reference values MUST come from THIS pod's own serial run, not from the old
# pod's card. First attempt hard-coded the old pod's 64/66/68 and tripped on
# draw 42, which reads 66.0 here (serially AND in parallel) -- a genuine
# one-task pod-to-pod difference on that draw, not a parallelism artifact.
declare -A KNOWN=()
for d in 42 43 44; do
  v=$(sc "ref_cem_e${d}")
  [ -n "$v" ] && [ "$v" != FAIL ] || die "no serial ref_cem_e${d} to validate against -- run the serial grid's P3 first"
  KNOWN[$d]=$v
done
log "  validating against this pod's serial values: ${KNOWN[42]}/${KNOWN[43]}/${KNOWN[44]}"
g=0
for d in 42 43 44; do
  flock "$SUM.lock" -c "sed -i \"/^val_cem_e${d},/d\" '$SUM'"
  ev "$g" "val_cem_e${d}" solver=cem & g=$((g+1))
done; wait
bad=0
for d in 42 43 44; do
  got=$(sc "val_cem_e${d}"); want=${KNOWN[$d]}
  if [ "$got" = "$want" ]; then log "  draw $d: $got == $want OK"
  else log "  draw $d: got '$got', serial value $want  <-- MISMATCH"; bad=1; fi
done
[ "$bad" = 0 ] || die "parallel evals do not reproduce serial values. The historical
  'evals must be sequential' rule still binds on this box -- rerun the serial
  grid (pldm_grid_egl.sh). Do NOT use any parallel numbers."
log "P0: PASSED -- per-GPU EGL pinning makes parallel evals reproduce serial values"

# ============================ P1 references ============================
log "P1: references (TD+CEM via +metric, the correct override)"
g=0
for d in 42 43 44; do ev "$g" "ref_cemtd_e${d}" solver=cem "+metric=$QTD" & g=$((g+1)); done
for d in 42 43 44; do ev "$g" "ref_cem_e${d}" solver=cem & g=$((g+1)); done
wait
log "P1: latent+CEM $(sc ref_cem_e42)/$(sc ref_cem_e43)/$(sc ref_cem_e44) | TD+CEM $(sc ref_cemtd_e42)/$(sc ref_cemtd_e43)/$(sc ref_cemtd_e44)"

# ==================== P2 stage A screen (draw 42) ====================
MW="0.1 0.3 1.0"; AMAX="3.5 4.5"; ALR="3e-4 1e-4"
tag_of(){ echo "mw$(echo "$1" | tr -d .)_a$(echo "$2" | tr -d .)_lr$3"; }
log "P2: stage A screen, 12 cells, ${NGPU}-wide"
g=0
for mw in $MW; do for am in $AMAX; do for alr in $ALR; do
  t=$(tag_of "$mw" "$am" "$alr")
  ev "$g" "grid_${t}_s0_e42" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_pldm_${t}_s0.pt" &
  g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done; done
wait
log "P2: screen done"

# ============ P3 stage B: top-3 x seeds 1,2 trains, then 43/44 ============
TOP=$(python3 - "$SUM" <<'PY'
import sys, re
best = []
for ln in open(sys.argv[1]):
    m = re.match(r"grid_(\S+)_s0_e42,([0-9.]+)$", ln.strip())
    if m: best.append((float(m.group(2)), m.group(1)))
best.sort(reverse=True)
seen, out = set(), []
for v, t in best:
    if t not in seen: seen.add(t); out.append(t)
print(" ".join(out[:3]))
PY
)
[ -n "$TOP" ] || die "could not rank stage A"
log "P3: top-3 = $TOP"
g=0
for t in $TOP; do for s in 1 2; do
  out=/workspace/actors/lip4_pldm_${t}_s${s}.pt
  [ -f "$out" ] && continue
  mw=$(echo "$t" | sed -E 's/mw([0-9]+)_.*/\1/'); mw=$(awk -v x="$mw" 'BEGIN{print x/10}')
  am=$(echo "$t" | sed -E 's/.*_a([0-9]+)_.*/\1/'); am=$(awk -v x="$am" 'BEGIN{print x/10}')
  alr=$(echo "$t" | sed -E 's/.*_lr(.*)$/\1/')
  case "$alr" in 3e-4) alrf=3e-5;; *) alrf=1e-5;; esac
  CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$am" \
    --mean-weight "$mw" --actor-lr "$alr" --actor-lr-final "$alrf" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --arch v4 --seed "$s" --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/grid_train_${t}_s${s}.log" 2>&1 &
  g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done
wait
log "P3: stage B trains done"
g=0
for t in $TOP; do for s in 0 1 2; do for d in 43 44; do
  ev "$g" "grid_${t}_s${s}_e${d}" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_pldm_${t}_s${s}.pt" &
  g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done; done
wait
log "P3: stage B evals done"

# ================================ P4 card ================================
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, re, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
print("=== PLDM GRID CARD (frozen PLDM, held-out 8000:10000, EGL, parallel-validated) ===")
cem = [rows.get(f"ref_cem_e{d}") for d in (42,43,44)]
td  = [rows.get(f"ref_cemtd_e{d}") for d in (42,43,44)]
if all(v is not None for v in cem):
    print(f"  latent+CEM {cem} -> 3-draw {mean(cem):.1f} | 43/44 {mean(cem[1:]):.1f}   (old 66.0)")
if all(v is not None for v in td):
    print(f"  TD+CEM     {td} -> 3-draw {mean(td):.1f} | 43/44 {mean(td[1:]):.1f}   (old 73.3)")
print("  --- stage A screen (seed 0, draw 42 = SELECTOR, not reportable) ---")
for v, k in sorted(((v, k) for k, v in rows.items()
                    if k.startswith("grid_") and k.endswith("_s0_e42")), reverse=True):
    print(f"    {k.replace('grid_','').replace('_s0_e42',''):22s} {v:5.1f}")
print("  --- stage B honest report (draws 43+44, never used for selection) ---")
tags = sorted({re.sub(r"grid_(.*)_s\d_e\d\d", r"\1", k) for k in rows if re.match(r"grid_.*_s\d_e4[34]$", k)})
best = (None, -1, None)
for t in tags:
    per = []
    for s in (0,1,2):
        vs = [rows.get(f"grid_{t}_s{s}_e{d}") for d in (43,44)]
        if all(v is not None for v in vs): per.append(mean(vs))
    if len(per) == 3:
        m = mean(per); sd = math.sqrt(sum((x-m)**2 for x in per)/2)
        print(f"    {t:22s} {' '.join(f'{x:5.1f}' for x in per)}  mean {m:.1f} sd {sd:.2f}")
        if m > best[1]: best = (t, m, (per, sd))
if best[0] and all(v is not None for v in td):
    bar = mean(td[1:]); per, sd = best[2]
    tstat = (best[1]-bar)/(sd/math.sqrt(3)) if sd > 0 else float("inf")
    print(f"  WINNER {best[0]}: {best[1]:.1f} vs TD+CEM(43/44) {bar:.1f} -> {best[1]-bar:+.1f}, "
          f"t={tstat:.2f} (df=2) {'SIGNIFICANT' if tstat > 2.92 else 'not significant'} (one-sided 5%)")
print("PLDM_GRID_PAR_DONE")
PY
log "done"
