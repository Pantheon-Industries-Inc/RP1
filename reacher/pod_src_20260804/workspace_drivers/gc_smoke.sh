#!/bin/bash
# Smoke-test the geometric lambda schedules before committing 36 trainings.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
for sc in uniform geom-early geom-late; do
  echo "=== $sc ==="
  CUDA_VISIBLE_DEVICES=0 timeout 900 python3 /workspace/swm_cem/scripts/plan/train_lip_ac.py \
    --cache /workspace/caches/canon_lejepa_fs5.pt \
    --cache-td /workspace/caches/canon_lejepa_fs1.pt \
    --h5 /workspace/reacher_slim.h5 \
    --wm /workspace/swm_home/checkpoints/lejepa_reacher \
    --pad-context --init-value /workspace/metrics/window3_lejepa.pt \
    --arch v4 --amax 2.2 --iters 8 --horizon 5 --max-delta 12 \
    --steps 20 --batch 32 --n-step 50 --expectile 0.1 \
    --lambda-schedule "$sc" --lambda-base 0.5 --seed 0 \
    --out /tmp/gc_${sc}.pt --out-value /tmp/gc_${sc}_v.pt 2>&1 \
    | grep -E "^step 0:|vframes|Error|Traceback" | head -3
  echo "  saved: $([ -f /tmp/gc_${sc}.pt ] && echo yes || echo NO)"
done
echo GC_SMOKE_DONE
