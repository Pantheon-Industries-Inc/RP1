#!/bin/bash
# Stage 1 for the DINO-WM TD campaign: prepare caches, train the canonical TD,
# and prove the metric wiring end-to-end on 6 tasks before spending sweep hours.
#
# The smoke eval is the step that matters: it is where a residual shape bug in
# the patch-token -> pooled-latent path would surface. Watch for the
# `[metric-hook] pooled to pred (...) goal (...)` line -- the last dim must be
# 384 and there must be no patch axis left.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; M=/workspace/metrics; C=/workspace/caches
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
CKPT=/workspace/ckpts/dinowm_noprop_cube
FULL=$C/dino_full_fs1.pt; TR=$C/dino_tr8000_fs1.pt; TR5=$C/dino_tr8000_fs5.pt
TD=$M/dino_td_canon.pt
mkdir -p "$L" "$M"; cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_dino_stage1.log"; }

log "=== waiting for the full cache ==="
for i in $(seq 1 120); do
  grep -q "latents (dim=" "$L/cache_dino.log" && break
  sleep 30
done
grep -q "latents (dim=" "$L/cache_dino.log" || { log "FATAL cache never finished"; exit 1; }
log "  $(grep -E 'latents \(dim=' "$L/cache_dino.log" | tail -1)"

# -------------------------------------------------- split: train on 0-7999 only
if [ ! -f "$TR" ]; then
  log "=== filter cache to episodes 0-7999 ==="
  python3 /workspace/filter_cache_eprange.py "$FULL" "$TR" --lo 0 --hi 8000 \
    > "$L/dino_filter.log" 2>&1 || { log "FATAL filter failed"; tail -5 "$L/dino_filter.log"; exit 1; }
  log "  $(tail -1 "$L/dino_filter.log")"
fi

# fs5 is only needed by LIP later, but it is cheap and keeps stage 2 unblocked
if [ ! -f "$TR5" ]; then
  log "=== fs5 subsample (for LIPv4 later) ==="
  python3 "$TRM/subsample_cache.py" --in "$TR" --out "$TR5" --frameskip 5 \
    > "$L/dino_fs5.log" 2>&1 || log "  WARNING fs5 failed (LIP stage will need it)"
fi
ls -la "$TR" "$TR5" 2>/dev/null | tee -a "$L/driver_dino_stage1.log"

# ------------------------------------------------------- canonical TD (anchor)
if [ ! -f "$TD" ]; then
  log "=== canonical TD: expectile 0.03, lr 1e-3, n-step 50, 6000 steps ==="
  CUDA_VISIBLE_DEVICES=0 python3 "$P/train_metric_sweep.py" --cache "$TR" \
    --learner td --head quasimetric --expectile 0.03 --lr 1e-3 \
    --n-step 50 --steps 6000 --seed 0 --out "$TD" \
    > "$L/dino_td_canon.log" 2>&1 || { log "FATAL TD failed"; tail -8 "$L/dino_td_canon.log"; exit 1; }
  log "  $(grep -E 'TD trained|final loss' "$L/dino_td_canon.log" | tail -1)"
fi

log "=== offline probe of the canonical TD (held-out eps >= 8000) ==="
CUDA_VISIBLE_DEVICES=0 python3 /workspace/probe_metric.py --cache "$FULL" \
  --metric "$TD" --ep-lo 8000 --tag canon 2>/dev/null | grep "^PROBE" | tee -a "$L/driver_dino_stage1.log"

# --------------------------------------------- 6-task smoke: does the hook fire
log "=== TD+CEM smoke, 6 tasks, held-out split ==="
CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 "$P/eval_wm_dino.py" --config-name cube \
  seed=42 eval.dataset_name="$EXPERT" eval.img_size=196 eval.goal_offset_steps=25 \
  eval.eval_budget=50 eval.num_eval=6 "+eval.ep_range=8000:10000" \
  policy="$CKPT" solver=cem "+metric=$TD" "+metric_pool_dim=384" \
  output.filename=dino_tdcem_smoke.txt > "$L/dino_tdcem_smoke.log" 2>&1
log "  hook: $(grep -E 'metric-hook' "$L/dino_tdcem_smoke.log" | tr '\n' ' ')"
log "  plan-score: $(grep -E 'plan-score metric' "$L/dino_tdcem_smoke.log" | tail -1)"
log "  result: $(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/dino_tdcem_smoke.log" | tail -1)"
grep -qE "Error|Traceback" "$L/dino_tdcem_smoke.log" && log "  SMOKE HAD ERRORS: $(grep -E 'Error' "$L/dino_tdcem_smoke.log" | tail -2)"
log "DINO_TD_STAGE1_DONE"
