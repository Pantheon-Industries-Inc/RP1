#!/bin/bash
# PLDM FULL LANE, TAKE 2 -- runs the entire PLDM pipeline strictly AFTER the
# composite finishes (its lane B died on the transformers key-layout mismatch;
# weights fixed in place by invert_pldm_keys.py, INVERT_OK 01:34 UTC).
# Replaces pldm_bench_followup.sh: same trigger, but also builds the caches/TD
# and trains the LIPv4 actors the composite never got to, so nothing runs
# beside the composite's eval queue and the box stays single-purpose.
#
#   wait COMPOSITE_DONE + quiet box
#   -> caches fs1/fs5 under PLDM (GPU 1) -> TD
#   -> LIPv4 s0-2 (GPUs 1-3, canonical recipe, amax 1.6)
#   -> serial evals (GPU 0): 9x pldm_lip + 3x pldm_cem + 3x pldm_cemtd
#   -> complete PLDM card (LIP vs CEM vs TD+CEM, frozen base, held-out)
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_split
SUM=$R/summary_dynasplit.csv; DRV=$L/driver_pldm2.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; AMAX=1.6; DRAWS="42 43 44"; SEEDS="0 1 2"
QF1F=/workspace/caches/pldm_full_fs1.pt; QF1=/workspace/caches/pldm_tr${EPHI}_fs1.pt
QF5=/workspace/caches/pldm_tr${EPHI}_fs5.pt; QTD=/workspace/metrics/pldm_TD.pt
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== PLDM lane take-2 (pid $$): waiting for the composite ==="
T0=$(date +%s)
until grep -q "COMPOSITE_DONE" "$L/driver_composite.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && die "composite never finished in 12h"
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 120; done
log "composite done, box quiet"

# ---- P1 caches + TD
if [ ! -f "$QF1" ]; then
  log "P1: caching fs1 under PLDM (GPU 1)"
  [ -f "$QF1F" ] || CUDA_VISIBLE_DEVICES=1 timeout 28800 python3 "$TRM/cache_latents.py" \
    --wm "$PLDM" --dataset "$EXPERT" --out "$QF1F" --state-key privileged_block_0_pos \
    > "$L/p2_cache_pldm_fs1.log" 2>&1 || die "PLDM cache failed -- see p2_cache_pldm_fs1.log"
  python3 /workspace/filter_cache_eprange.py "$QF1F" "$QF1" --lo 0 --hi "$EPHI" \
    > "$L/p2_filter_pldm.log" 2>&1 || die "filter failed"
fi
[ -f "$QF5" ] || python3 "$TRM/subsample_cache.py" --in "$QF1" --out "$QF5" --frameskip 5 \
  > "$L/p2_cache_pldm_fs5.log" 2>&1 || die "fs5 failed"
[ -f "$QTD" ] || CUDA_VISIBLE_DEVICES=1 timeout 28800 python3 "$P/train_metric.py" \
  --cache "$QF1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
  --steps 6000 --seed 0 --out "$QTD" > "$L/p2_td_pldm.log" 2>&1 || die "TD failed"
log "P1: caches + TD ready"

# ---- P2 LIPv4 wave
log "P2: PLDM LIPv4 seeds $SEEDS (GPUs 1-3)"
g=1; for s in $SEEDS; do
  out=/workspace/actors/lip4_pldm_s${s}.pt
  if [ ! -f "$out" ]; then
    CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
      --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
      --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$s" \
      --out "$out" --out-value "${out%.pt}_value.pt" \
      > "$L/p2_lip_pldm_s${s}.log" 2>&1 || log "  pldm/s$s TRAIN FAILED" &
  fi
  g=$((g+1))
done; wait
log "P2: actors done"

# ---- P3 serial evals
run_eval(){ # name extra...
  local nm=$1; shift
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$PLDM" "$@" output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" \
    || { log "  $nm: ep_range NOT APPLIED"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
}
log "P3: eval queue (15 cells, sequential)"
while pgrep -f "train_lip_a[c]|train_metri[c]" >/dev/null; do sleep 60; done
for s in $SEEDS; do for d in $DRAWS; do
  run_eval "pldm_lip_s${s}_e${d}" seed=$d solver=lip \
    "solver.actor_path=/workspace/actors/lip4_pldm_s${s}.pt"
done; done
for d in $DRAWS; do run_eval "pldm_cem_e${d}"   seed=$d solver=cem; done
for d in $DRAWS; do run_eval "pldm_cemtd_e${d}" seed=$d solver=cem "+metric=$QTD"; done

# ---- P4 card
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
def m3(pref):
    vs = [rows.get(f"{pref}_e{d}") for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else vs
print("=== PLDM CARD (frozen PLDM_OgBench_lewm, held-out 8000:10000, amax 1.6, egl) ===")
cem, ctd = m3("pldm_cem"), m3("pldm_cemtd")
if cem: print(f"  latent+CEM  {'/'.join(f'{v:.0f}' for v in cem)} -> {sum(cem)/3:.1f}")
if ctd: print(f"  TD+CEM      {'/'.join(f'{v:.0f}' for v in ctd)} -> {sum(ctd)/3:.1f}")
lips = []
for s in (0, 1, 2):
    lip = m3(f"pldm_lip_s{s}")
    if lip:
        lips.append(sum(lip)/3)
        print(f"  LIP s{s}      {'/'.join(f'{v:.0f}' for v in lip)} -> {sum(lip)/3:.1f}")
if lips: print(f"  LIP 3-seed  {sum(lips)/len(lips):.1f}")
print("PLDM_NIGHT2_DONE")
PY
log "done"
