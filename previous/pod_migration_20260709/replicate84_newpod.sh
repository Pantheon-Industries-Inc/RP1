#!/bin/bash
# Replicate the documented 84.0 (Latent+CEM, v2WM) — THEIR exact protocol on the FULL
# lewm-cube dataset. New-pod paths: dataset on the persistent volume.
# Pre-verified: seed-42 draw == their printed 50 row indices; physics/render/scaler parity.
set -x
SNAP=/workspace/snapshot2
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5
CK=/workspace/ckpts/ogbench_cube_single_v2WM
export MUJOCO_GL=osmesa PYTHONPATH=$SNAP TQDM_DISABLE=1

# wait for extraction (up to 3h)
for i in $(seq 1 360); do [ -f /workspace/datasets/lewm_cube_full/extract.done ] && break; sleep 30; done
[ -f /workspace/datasets/lewm_cube_full/extract.done ] || { echo "extract never finished"; exit 1; }

python - <<'PY' || exit 1
import h5py
f = h5py.File("/workspace/datasets/lewm_cube_full/cube_single_expert.h5", "r")
ln = f["ep_len"][:]
assert ln.sum() == 2010000 and (ln == 201).all(), (int(ln.sum()), int((ln == 0).sum()))
print("h5 integrity OK: 10000 eps x 201")
PY

cd $SNAP
ev () { # NAME SEED GPU EXTRA...
  local NAME=$1 S=$2 GPU=$3; shift 3
  grep -qa success_rate /workspace/ev_rep84_${NAME}_s$S.log 2>/dev/null && return
  CUDA_VISIBLE_DEVICES=$GPU python scripts/plan/eval_wm.py --config-name cube \
    seed=$S eval.dataset_name=$H5 ++bf16=true eval.img_size=224 "$@" \
    output.filename=ev_rep84_${NAME}_s$S.txt > /workspace/ev_rep84_${NAME}_s$S.log 2>&1 &
}
ev latent 42 0 policy=$CK solver=cem
ev latent 43 1 policy=$CK solver=cem
ev latent 44 2 policy=$CK solver=cem
ev random 42 3 policy=random
wait
echo done > /workspace/replicate84.done
grep -a "success_rate" /workspace/ev_rep84_*_s4?.log | head
