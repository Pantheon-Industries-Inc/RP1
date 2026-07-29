#!/bin/bash
# PLDM stage 3: the verdict runs.
#   (a) a4.5 seeds 3-5  -> 6-seed one-sample t vs TD+CEM 73.3 (the
#       "significantly beats" test; 3-seed mean 75.3 came with spread 8.0,
#       one hot seed -- not yet a claim).
#   (b) a5.5 seeds 0-2  -> completes the amax curve (69.8 -> 73.3 -> 73.6 ->
#       75.3 and still climbing; is 4.5 the peak or mid-slope?).
#   (c) h3/h2 re-evals with MATCHED solver horizon -- stage 2's h-arms
#       trained fine but died at eval: the v4 actor emits horizon-length
#       plans and the solver demanded 5-step plans from 3/2-step actors.
#       Solver horizon is a planner-internal property (like CEM's population);
#       task, budget and success criterion are unchanged. Caveat printed.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_dynasplit.csv
DRV=$L/driver_pldm_stage3.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== PLDM stage 3 (pid $$): a45 x s3-5, a55 probe, h-arm matched-horizon evals ==="
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 120; done
for f in "$QF1" "$QF5" "$QTD" "$AH5"; do [ -e "$f" ] || die "missing $f"; done
log "P0: box quiet"

train_one(){ # gpu seed amax out
  CUDA_VISIBLE_DEVICES=$1 timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$3" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$2" \
    --out "$4" --out-value "${4%.pt}_value.pt" \
    > "$L/pldm_lip_$(basename "$4" .pt).log" 2>&1 || log "  $(basename "$4") TRAIN FAILED"
}
log "P1: a45 seeds 3-5 (GPUs 0-2)"
g=0; for s in 3 4 5; do
  out=/workspace/actors/lip4_pldm_a45_s${s}.pt
  [ -f "$out" ] || train_one "$g" "$s" 4.5 "$out" & g=$((g+1))
done; wait
log "P1: a45 extension done"
log "P2: a55 seeds 0-2 (GPUs 0-2)"
g=0; for s in 0 1 2; do
  out=/workspace/actors/lip4_pldm_a55_s${s}.pt
  [ -f "$out" ] || train_one "$g" "$s" 5.5 "$out" & g=$((g+1))
done; wait
log "P2: a55 probe done"

run_eval(){ # name actor draw extra-overrides...
  local nm=$1 actor=$2 d=$3; shift 3
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  [ -f "$actor" ] || { log "  $nm: actor missing, FAIL"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$PLDM" solver=lip "solver.actor_path=$actor" "$@" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" \
    || { log "  $nm: ep_range NOT APPLIED"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
}
log "P3: eval queue (36 cells, sequential, egl)"
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" >/dev/null; do sleep 60; done
for s in 3 4 5; do for d in $DRAWS; do
  run_eval "pldm_a45_s${s}_e${d}" "/workspace/actors/lip4_pldm_a45_s${s}.pt" "$d"
done; done
for s in 0 1 2; do for d in $DRAWS; do
  run_eval "pldm_a55_s${s}_e${d}" "/workspace/actors/lip4_pldm_a55_s${s}.pt" "$d"
done; done
# h-arm re-evals: matched solver horizon (actors emit horizon-length plans)
for s in 0 1 2; do for d in $DRAWS; do
  run_eval "pldm_h3m_s${s}_e${d}" "/workspace/actors/lip4_pldm_h3_s${s}.pt" "$d" \
    plan_config.horizon=3 plan_config.receding_horizon=3
  run_eval "pldm_h2m_s${s}_e${d}" "/workspace/actors/lip4_pldm_h2_s${s}.pt" "$d" \
    plan_config.horizon=2 plan_config.receding_horizon=2
done; done

python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
def sm(arm, s):
    vs = [rows.get(f"{arm}_s{s}_e{d}") for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else sum(vs) / 3
print("=== PLDM VERDICT CARD (frozen PLDM, held-out 8000:10000, egl) ===")
print("  refs: latent+CEM 66.0 | TD+CEM 73.3")
print("  amax curve (h5): 1.6->69.8  2.6->73.3  3.5->73.6")
for arm, label, seeds in (("pldm_a45", "a4.5/h5", range(6)),
                          ("pldm_a55", "a5.5/h5", range(3)),
                          ("pldm_h3m", "h3 (matched hor.)", range(3)),
                          ("pldm_h2m", "h2 (matched hor.)", range(3))):
    ms = [sm(arm, s) for s in seeds]
    ok = [m for m in ms if m is not None]
    if ok:
        print(f"  {label:18s} {' '.join(f'{m:5.1f}' if m is not None else '   NA' for m in ms)}"
              f"   mean {sum(ok)/len(ok):.1f} (n={len(ok)})")
a45 = [sm("pldm_a45", s) for s in range(6)]
if all(m is not None for m in a45):
    n = 6; mean = sum(a45)/n
    sd = math.sqrt(sum((x-mean)**2 for x in a45)/(n-1))
    t = (mean - 73.3)/(sd/math.sqrt(n))
    print(f"  a4.5 6-seed vs TD+CEM 73.3: mean {mean:.2f} sd {sd:.2f} t {t:.2f} (df=5)"
          f" -> {'SIGNIFICANT' if t > 2.02 else 'not significant'} (one-sided 5%)")
print("  caveat: h-arm cells use solver horizon = training horizon (planner-"
      "internal); task, budget, success criterion identical to all other cells.")
print("PLDM_STAGE3_DONE")
PY
log "done"
