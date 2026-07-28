#!/bin/bash
# PLDM REFERENCE BENCHMARKS -- the two arms the composite does not produce:
# PLDM+CEM (latent cost) and PLDM+TD+CEM, held-out 8000:10000, same protocol
# as every other cell in summary_dynasplit.csv. Together with the composite's
# pldm_lip_* rows this completes the PLDM benchmark card (LIP vs CEM vs
# TD+CEM on a frozen PLDM base -- the ~450x-compute comparison).
#
# SAFE-BY-WAITING: launched any time; it sleeps until the composite prints
# COMPOSITE_DONE and no training/eval process remains, because evals render
# through EGL and concurrency corrupts silently (the 8-pt lesson). Touches no
# GPU until then.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
SUM=$R/summary_dynasplit.csv; DRV=$L/driver_pldmbench.log
PLDM=/workspace/models/PLDM_OgBench_lewm
QTD=/workspace/metrics/pldm_TD.pt
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== PLDM reference bench (pid $$): waiting for the composite to finish ==="
T0=$(date +%s)
until grep -q "COMPOSITE_DONE" "$L/driver_composite.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && { log "FATAL: composite never finished in 12h"; exit 1; }
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 120; done
log "composite done, box eval-safe"
[ -f "$QTD" ] || { log "FATAL: $QTD missing -- PLDM lane was skipped; nothing to benchmark against"; exit 1; }

run_eval(){ # name extra-hydra-args...
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
for d in $DRAWS; do run_eval "pldm_cem_e${d}"   seed=$d solver=cem; done
for d in $DRAWS; do run_eval "pldm_cemtd_e${d}" seed=$d solver=cem "+metric=$QTD"; done

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
print("=== PLDM BENCH CARD (frozen PLDM_OgBench_lewm, held-out 8000:10000, egl) ===")
cem, ctd = m3("pldm_cem"), m3("pldm_cemtd")
if cem: print(f"  latent+CEM  {'/'.join(f'{v:.0f}' for v in cem)} -> {sum(cem)/3:.1f}")
if ctd: print(f"  TD+CEM      {'/'.join(f'{v:.0f}' for v in ctd)} -> {sum(ctd)/3:.1f}")
for s in (0, 1, 2):
    lip = m3(f"pldm_lip_s{s}")
    if lip: print(f"  LIP s{s}      {'/'.join(f'{v:.0f}' for v in lip)} -> {sum(lip)/3:.1f}")
print("PLDM_BENCH_DONE")
PY
log "done"
