#!/bin/bash
# PLDM LIP GRID SEARCH: mean-weight x amax x actor-lr, on the rebuilt pod.
#
# NEW LEVER -- --mean-weight (default 0.1). train_lip_ac.py:431 computes
#     loss = e_path[-1] + mean_weight * mean(e_path)
# where e_path[k] is the plan cost after inner-optimization iterate k. So it is
# DEEP SUPERVISION OVER THE OPTIMIZER'S OWN TRAJECTORY: at 0.1 only the final
# iterate matters, so the actor may take a wild path through action space as
# long as the endpoint scores well IN THE WM'S IMAGINATION -- exactly the
# exploitation a weak base rewards. Raising it forces every iterate to be good.
# Orthogonal to amax, which bounds action MAGNITUDE, not the optimization PATH.
# (Note: this weights intermediate ITERATES, not intermediate rollout timesteps
# -- LIP has no per-timestep term; PWM's --dense is the analogue there.)
#
# RENDERER: egl -- the in-domain renderer (it produced the authors' h5 frames),
# available here because this pod was created with NVIDIA_DRIVER_CAPABILITIES set.
# Writes to its OWN summary CSV: the osmesa pod cached cells under the same
# names, and a mixed-renderer card would be silently wrong. Cells that exist in
# both CSVs become a direct egl-vs-osmesa comparison on PLDM -- previously only
# measured on LeWM. Reference point already in hand: latent+CEM draw 43 read
# 66.0 under egl (old pod) and 64.0 under osmesa (rebuild), i.e. one task in 50.
#
# GRID  mean-weight {0.1, 0.3, 1.0} x amax {3.5, 4.5} x actor-lr {3e-4, 1e-4}
#       = 12 configs. The (0.1, 3.5, 3e-4) and (0.1, 4.5, 3e-4) cells are the
#       OLD a35/a45 configs, so the grid re-measures them: they must land near
#       73.6 / 75.3 or this pod's stack does not reproduce the old numbers.
#
# HONEST SELECTION (mandatory at 12 cells -- the winner's curse is already
# demonstrated in this campaign: a45 read 75.3 at 3 seeds, 74.6 at 6):
#   stage A  screen 12 configs, seed 0, DRAW 42 ONLY
#   stage B  top 3 configs, seeds 0-2, DRAWS 43+44 (never used for selection)
#   headline = winner's 43/44 mean; the 3-draw mean is also printed for
#   comparability with the old cards, flagged as selection-contaminated.
#
# 8 GPUs: 12 trains = 2 waves, 6 trains = 1 wave. ~4h total after setup.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_grid_egl.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1F=/workspace/caches/pldm_full_fs1.pt
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; NGPU=8
mkdir -p "$R" "$L"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== PLDM GRID (pid $$) ==="
until grep -q "SETUP_DONE" "$L/driver_setup.log" 2>/dev/null; do
  grep -q "FATAL" "$L/driver_setup.log" 2>/dev/null && die "setup failed"
  sleep 60
done
[ -d "$EXPERT" ] || die "expert lance missing"
[ -f "$PLDM/weights.pt" ] || die "PLDM weights missing"
log "P0: setup complete"

# ------------------------------------------------- P1 key layout + action h5
if [ ! -f /workspace/.pldm_keys_ok ]; then
  log "P1: PLDM key layout for transformers 4.49 (local copy is 5.x layout)"
  python3 /workspace/invert_pldm_keys.py > "$L/grid_invert.log" 2>&1
  grep -qE "INVERT_OK|nothing to invert" "$L/grid_invert.log" || { tail -5 "$L/grid_invert.log"; die "key inversion failed"; }
  touch /workspace/.pldm_keys_ok
fi
[ -f "$AH5" ] || { log "P1: building expert_actions.h5";
  python3 /workspace/build_action_h5.py /workspace/datasets/ogb_cube_single "$AH5" \
    > "$L/grid_h5.log" 2>&1 || die "action h5 failed"; }
log "P1: $(tail -1 "$L/grid_h5.log" 2>/dev/null || echo 'h5 cached')"

# ------------------------------------------------------- P2 caches + TD
if [ ! -f "$QF1" ]; then
  log "P2: caching fs1 under PLDM (~25 min)"
  [ -f "$QF1F" ] || CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$TRM/cache_latents.py" \
    --wm "$PLDM" --dataset "$EXPERT" --out "$QF1F" --state-key privileged_block_0_pos \
    > "$L/grid_cache.log" 2>&1 || { tail -5 "$L/grid_cache.log"; die "cache failed"; }
  python3 /workspace/filter_cache_eprange.py "$QF1F" "$QF1" --lo 0 --hi "$EPHI" \
    > "$L/grid_filter.log" 2>&1 || die "filter failed"
  grep -q FILTER_CACHE_DONE "$L/grid_filter.log" || die "filter incomplete"
fi
[ -f "$QF5" ] || python3 "$TRM/subsample_cache.py" --in "$QF1" --out "$QF5" --frameskip 5 \
  > "$L/grid_fs5.log" 2>&1 || die "fs5 failed"
[ -f "$QTD" ] || { log "P2: TD teacher";
  CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_metric.py" \
    --cache "$QF1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
    --steps 6000 --seed 0 --out "$QTD" > "$L/grid_td.log" 2>&1 || die "TD failed"; }
log "P2: caches + TD ready"

# ------------------------------------------------------------- eval helper
ev(){ # name solver-args... ; uses $d
  local nm=$1; shift
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$PLDM" "$@" output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" \
    || { log "  $nm: ep_range NOT APPLIED"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
}

# --------------------------------------------- P3 ANCHOR + references
# Anchor: old per-draw card had latent+CEM 64/66/68 on draws 42/43/44.
log "P3: anchor + references (stack reproduction check)"
for d in 42 43 44; do ev "ref_cem_e${d}" solver=cem; done
A43=$(sc ref_cem_e43)
awk -v x="${A43:-0}" 'BEGIN{exit !(x>=64 && x<=68)}' \
  || log "  *** ANCHOR WARNING: latent+CEM draw 43 = ${A43:-NA}, old = 66.0. Old-vs-new PLDM cells may not be comparable -- flag in any writeup."
for d in 42 43 44; do ev "ref_cemtd_e${d}" solver=cem "solver.cost_path=$QTD"; done
log "P3: refs done"

# ------------------------------------------------------------ grid definition
MW="0.1 0.3 1.0"; AMAX="3.5 4.5"; ALR="3e-4 1e-4"
cfg_tag(){ echo "mw$(echo "$1" | tr -d .)_a$(echo "$2" | tr -d .)_lr$3"; }
train_cfg(){ # gpu mw amax alr seed
  local gpu=$1 mw=$2 am=$3 alr=$4 s=$5
  local alrf; case "$alr" in 3e-4) alrf=3e-5;; 1e-4) alrf=1e-5;; *) alrf=3e-5;; esac
  local tag; tag=$(cfg_tag "$mw" "$am" "$alr")
  local out=/workspace/actors/lip4_pldm_${tag}_s${s}.pt
  [ -f "$out" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$am" \
    --mean-weight "$mw" --actor-lr "$alr" --actor-lr-final "$alrf" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --arch v4 --seed "$s" --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/grid_train_${tag}_s${s}.log" 2>&1 || log "  $tag/s$s TRAIN FAILED"
}

# ------------------------------------------------ P4 stage A: screen on draw 42
log "P4: stage A -- 12 configs, seed 0, ${NGPU} GPUs"
g=0
for mw in $MW; do for am in $AMAX; do for alr in $ALR; do
  train_cfg "$g" "$mw" "$am" "$alr" 0 &
  g=$((g+1))
  [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done; done
wait
log "P4: stage A trains done"
while pgrep -f "train_lip_a[c]|train_metri[c]" >/dev/null; do sleep 30; done
d=42
for mw in $MW; do for am in $AMAX; do for alr in $ALR; do
  tag=$(cfg_tag "$mw" "$am" "$alr")
  ev "grid_${tag}_s0_e42" solver=lip "solver.actor_path=/workspace/actors/lip4_pldm_${tag}_s0.pt"
done; done; done
log "P4: stage A screen complete"

# ---------------------------------- P5 stage B: top 3 on seeds 0-2, draws 43+44
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
log "P5: stage B on top-3: $TOP"
g=0
for tag in $TOP; do for s in 1 2; do
  mw=$(echo "$tag" | sed -E 's/mw([0-9]+)_.*/\1/'); mw=$(echo "scale=1; $mw/10" | bc)
  am=$(echo "$tag" | sed -E 's/.*_a([0-9]+)_.*/\1/'); am=$(echo "scale=1; $am/10" | bc)
  alr=$(echo "$tag" | sed -E 's/.*_lr(.*)$/\1/')
  train_cfg "$g" "$mw" "$am" "$alr" "$s" &
  g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done
wait
while pgrep -f "train_lip_a[c]" >/dev/null; do sleep 30; done
log "P5: stage B trains done"
for tag in $TOP; do for s in 0 1 2; do for dd in 43 44; do
  d=$dd; ev "grid_${tag}_s${s}_e${dd}" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_pldm_${tag}_s${s}.pt"
done; done; done

# ------------------------------------------------------------------ P6 card
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, re, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
def mean(xs): return sum(xs)/len(xs)
print("=== PLDM GRID CARD (frozen PLDM, held-out 8000:10000, egl) ===")
ref_cem = [rows.get(f"ref_cem_e{d}") for d in (42,43,44)]
ref_td  = [rows.get(f"ref_cemtd_e{d}") for d in (42,43,44)]
if all(v is not None for v in ref_cem):
    print(f"  latent+CEM  {ref_cem} -> 3-draw {mean(ref_cem):.1f} | 43/44 {mean(ref_cem[1:]):.1f}  (old 66.0)")
if all(v is not None for v in ref_td):
    print(f"  TD+CEM      {ref_td} -> 3-draw {mean(ref_td):.1f} | 43/44 {mean(ref_td[1:]):.1f}  (old 73.3)")
print("  --- stage A screen (seed 0, draw 42 = SELECTOR, not reportable) ---")
scr = sorted(((v, k) for k, v in rows.items() if k.startswith("grid_") and k.endswith("_s0_e42")), reverse=True)
for v, k in scr:
    print(f"    {k.replace('grid_','').replace('_s0_e42',''):22s} {v:5.1f}")
print("  --- stage B honest report (draws 43+44, never used for selection) ---")
tags = sorted({re.sub(r"grid_(.*)_s\d_e\d\d", r"\1", k) for k in rows if re.match(r"grid_.*_s\d_e4[34]$", k)})
best = (None, -1, None)
for t in tags:
    per_seed = []
    for s in (0,1,2):
        vs = [rows.get(f"grid_{t}_s{s}_e{d}") for d in (43,44)]
        if all(v is not None for v in vs): per_seed.append(mean(vs))
    if len(per_seed) == 3:
        m = mean(per_seed); sd = math.sqrt(sum((x-m)**2 for x in per_seed)/2)
        three = [mean([rows[f"grid_{t}_s{s}_e{d}"] for d in (42,43,44)])
                 for s in (0,1,2) if all(f"grid_{t}_s{s}_e{d}" in rows for d in (42,43,44))]
        extra = f" | 3-draw {mean(three):.1f}*" if len(three) == 3 else ""
        print(f"    {t:22s} {' '.join(f'{x:5.1f}' for x in per_seed)}  mean {m:.1f} sd {sd:.2f}{extra}")
        if m > best[1]: best = (t, m, (per_seed, sd))
if best[0] and all(v is not None for v in ref_td):
    bar = mean(ref_td[1:]); per_seed, sd = best[2]
    t_stat = (best[1]-bar)/(sd/math.sqrt(3)) if sd > 0 else float("inf")
    print(f"  WINNER {best[0]}: {best[1]:.1f} vs TD+CEM(43/44) {bar:.1f} -> "
          f"{best[1]-bar:+.1f}, t={t_stat:.2f} (df=2) "
          f"{'SIGNIFICANT' if t_stat > 2.92 else 'not significant'} (one-sided 5%)")
print("  * 3-draw means include draw 42 which selected the config -- optimistic,")
print("    printed only for comparability with the older 3-draw cards.")
print("PLDM_GRID_DONE")
PY
log "done"
