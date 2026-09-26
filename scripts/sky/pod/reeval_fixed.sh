#!/usr/bin/env bash
# Re-evaluate specific actor SNAPSHOTS (fixed-stop protocol) on the report draws, on a pod job's own
# environment. Writes W&B keys eval/<EVAL_TAG>/* into the row's run (resumed via the checkpoint's
# wandb_id), so the deployed row's eval/rh5/* stays untouched.
#   scripts/sky/pod/reeval_fixed.sh <host:port> <jobname> <env> <base> <offset> <snapshot.pt> [...]
#   e.g. ... root@69.30.85.162:22186 rlp-tw-uniJ tworoom lejepa 100 \
#            /checkpoints/armin@pantheon.inc/rlp-tw-uniJ-20260921/actors/g_tworoom_lejepa_unig_ctrl_a2.5_t9000_s1_step2000.pt
set -euo pipefail
HP=$1; JOB=$2; ENVN=$3; BASE=$4; OFF=$5; shift 5
EVAL_TAG=${EVAL_TAG:-rh5fixed}
for SNAP in "$@"; do
  ssh -o BatchMode=yes -o ConnectTimeout=25 -p "${HP##*:}" -i ~/.ssh/id_ed25519 "${HP%%:*}" \
    "JOB=$JOB ENVN=$ENVN BASE=$BASE OFF=$OFF SNAP=$SNAP EVAL_TAG=$EVAL_TAG EVAL_SEEDS='${EVAL_SEEDS:-42 43 44}' bash -s" <<'REMOTE'
set -euo pipefail
export HOME=/root/homes/$JOB
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}   # 362-thread evals thrashed the pod
[ -d "$HOME/repo/stable-worldmodel" ] || export HOME=/root      # jobs launched without --home
for v in WANDB_API_KEY GIT_TOKEN; do [ -n "${!v:-}" ] || export "$v=$(xargs -0 -n1 printf '%s\n' < /proc/1/environ | sed -n "s/^$v=//p" | head -1)"; done
source /root/podjobs/$JOB/env.sh >/dev/null 2>&1 || true
export WANDB_DIR=/mnt/raid0/rlp-wandb/${EXPERIMENT_TAG}/reeval; mkdir -p "$WANDB_DIR"
cd ~/repo/stable-worldmodel
export STABLEWM_HOME=~/gh PYTHONPATH=$PWD HDF5_PLUGIN_PATH=$(python3 -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')
H5=$(ls ~/gd/*.h5 | head -1)
if [ "$ENVN" = tworoom ]; then CK=~/gh/checkpoints/${BASE}_tworoom; EXTRA="--override +eval.ep_range=8000:10000"
else CK=~/gh/checkpoints/${BASE}_cube; EXTRA="--override ++bf16=true +eval.ep_range=8000:10000"; export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl; fi
[ -f "$SNAP" ] || { echo "missing snapshot $SNAP" >&2; exit 1; }
RAWD=/checkpoints/armin@pantheon.inc/${EXPERIMENT_TAG}/rawlogs_fixed; mkdir -p "$RAWD"
echo "[reeval] $(basename "$SNAP") offsets $OFF draws ${EVAL_SEEDS} -> eval/$EVAL_TAG/*"
CUDA_VISIBLE_DEVICES=0 python3 scripts/plan/eval_log.py --actor "$SNAP" --env-config "$ENVN" --h5 "$H5" --policy "$CK" \
  --rh 5 --seeds ${EVAL_SEEDS:-42 43 44} --offsets $OFF $EXTRA --log-dir "$RAWD" --tag "$EVAL_TAG" 2>&1 | grep -E "\[summary\]|rror|Traceback" | tail -3
REMOTE
done
