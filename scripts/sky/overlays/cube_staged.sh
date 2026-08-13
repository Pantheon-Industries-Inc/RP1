#!/usr/bin/env bash
set -euo pipefail

# Three-shard Cube protocol:
#   nine configs at actor seed 0 -> select top three on task seeds 50/51 ->
#   train only those finalists at actor seeds 1/2 -> report all three actor
#   seeds on task seeds 42/43/44.  Selection and report draws are disjoint.

cd ~/repo/stable-worldmodel
export STABLEWM_HOME=~/gh PYTHONPATH=$PWD TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HDF5_PLUGIN_PATH=$(python3 -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

SHARD_INDEX=${SHARD_INDEX:?set SHARD_INDEX}
SHARD_COUNT=${SHARD_COUNT:-3}
STAGE_TOPK=${STAGE_TOPK:-3}
(( SHARD_INDEX >= 0 && SHARD_INDEX < SHARD_COUNT ))
(( SHARD_COUNT == 3 && STAGE_TOPK == 3 ))

H5=$(ls ~/gd/*.h5 | head -1)
CK=$(ls ~/gh/checkpoints | head -1)
CRIT=$(ls ~/gm/td_*.pt | head -1)
PERSIST=/checkpoints/${PANTHEON_USER}/${EXPERIMENT_TAG}
STAGE=$PERSIST/stages/seed0-top3-v1
SLOT=~/gl/slots
mkdir -p "$PERSIST/actors" "$PERSIST/values" "$PERSIST/logs" \
         "$STAGE/scores" "$STAGE/final" "$SLOT"

AMFIX=1.6; [[ "$BASE" = pldm ]] && AMFIX=4.5
CFG=()
for MW in 0.0 0.1 0.3; do
  for LR in 1e-4 3e-4 1e-3; do
    CFG+=("olr_a${AMFIX}_m${MW}_l${LR} $AMFIX $MW $LR")
  done
done

acquire() { local s; while :; do
  for s in $(seq 0 $((MAXPAR - 1))); do
    mkdir "$SLOT/$s" 2>/dev/null && { echo "$s"; return; }
  done
  sleep 10
done; }

train_actor() {
  local gpu=$1 nm=$2 am=$3 mw=$4 lr=$5 seed=$6
  local out=$PERSIST/actors/g_cube_${BASE}_${nm}_s${seed}.pt
  local val=$PERSIST/values/g_cube_${BASE}_${nm}_s${seed}_value.pt
  if [[ ! -s "$out" ]]; then
    local lrf
    lrf=$(python3 -c "print('%g' % (float('$lr') / 10))")
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu \
      python3 scripts/plan/train_lip_ac.py \
        --h5 "$H5" --wm "$CK" --cache ~/gc/fs5.pt --cache-td ~/gc/fs1.pt \
        --init-value "$CRIT" --arch v4 --horizon 5 --iters 8 --max-delta 12 \
        --amax "$am" --mean-weight "$mw" --actor-lr "$lr" --actor-lr-final "$lrf" \
        --align-mode randhorizon --align-weight 0 \
        --gamma 0.98 --replay-prob 0.5 --expand-weight 0 \
        --actor-only --reuse-refinement-rollouts --cache-device auto \
        --steps "${STEPS:-6000}" --batch "${BATCH:-128}" --n-step 50 --seed "$seed" \
        --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY" \
        --wandb-name "g_cube_${BASE}_${nm}_s${seed}" \
        --out "$out" --out-value "$val" \
        > "$PERSIST/logs/g_${nm}_s${seed}_train.log" 2>&1
  else
    echo "[reuse] $nm actor seed $seed" >&2
  fi
  [[ -s "$out" ]] || return 1
  printf '%s\n' "$out"
}

eval_actor() {
  local gpu=$1 actor=$2 nm=$3 seed=$4 tag=$5 draws=$6 log
  log=$PERSIST/logs/g_${nm}_s${seed}_${tag}_eval.log
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu \
    python3 scripts/plan/eval_log.py --actor "$actor" \
      --env-config cube --h5 "$H5" --policy "$CK" --rh 1 \
      --seeds $draws --offsets 25 100 \
      --override ++bf16=true +eval.ep_range=8000:10000 \
      --log-dir ~/gl --tag "$tag" > "$log" 2>&1
  grep '\[summary\]' "$log" | tail -1
}

echo "[stage1] cube/$BASE shard ${SHARD_INDEX}/${SHARD_COUNT}: seed-0 screen"
idx=0
for row in "${CFG[@]}"; do
  if (( idx % SHARD_COUNT != SHARD_INDEX )); then idx=$((idx + 1)); continue; fi
  idx=$((idx + 1))
  set -- $row; NM=$1; AM=$2; MW=$3; LR=$4
  SCORE=$STAGE/scores/${NM}.tsv
  [[ -s "$SCORE" ]] && continue
  S=$(acquire)
  (
    OUT=$(train_actor "$S" "$NM" "$AM" "$MW" "$LR" 0)
    SUMMARY=$(eval_actor "$S" "$OUT" "$NM" 0 sel "50 51")
    python3 - "$SUMMARY" "$NM" "$AM" "$MW" "$LR" "$SCORE" <<'PY'
import json, os, sys
line, nm, am, mw, lr, out = sys.argv[1:]
data = json.loads(line.split("[summary]", 1)[1].strip())
score = float(data["mean_rh1_all"])
tmp = out + f".tmp.{os.getpid()}"
with open(tmp, "w") as f: f.write(f"{score}\t{nm}\t{am}\t{mw}\t{lr}\n")
os.replace(tmp, out)
print(f"[stage1-score] {nm}: {score:.3f}", flush=True)
PY
    rmdir "$SLOT/$S" 2>/dev/null || true
  ) &
done
wait

deadline=$(( $(date +%s) + 43200 ))
while (( $(find "$STAGE/scores" -name '*.tsv' | wc -l) < ${#CFG[@]} )); do
  (( $(date +%s) < deadline )) || { echo "stage1 barrier timed out"; exit 1; }
  echo "[stage1-barrier] $(find "$STAGE/scores" -name '*.tsv' | wc -l)/${#CFG[@]}"
  sleep 60
done

SELECTED=$STAGE/selected.tsv
if [[ ! -s "$SELECTED" ]] && mkdir "$STAGE/select.lock" 2>/dev/null; then
  python3 - "$STAGE/scores" "$STAGE_TOPK" "$SELECTED" <<'PY'
import glob, os, sys
root, topk, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = []
for path in glob.glob(os.path.join(root, "*.tsv")):
    score, nm, am, mw, lr = open(path).read().strip().split("\t")
    rows.append((float(score), nm, am, mw, lr))
rows.sort(key=lambda x: (-x[0], x[1]))
assert len(rows) == 9
tmp = out + f".tmp.{os.getpid()}"
with open(tmp, "w") as f:
    for rank, row in enumerate(rows[:topk]):
        f.write("\t".join([str(rank), *map(str, row)]) + "\n")
os.replace(tmp, out)
print("[selected]", [(r[1], r[0]) for r in rows[:topk]], flush=True)
PY
fi
while [[ ! -s "$SELECTED" ]]; do sleep 10; done

# Shard i owns finalist rank i.  It reports seed 0 and trains/reports seeds 1/2.
while IFS=$'\t' read -r RANK SCORE NM AM MW LR; do
  (( RANK == SHARD_INDEX )) || continue
  for SEED in 0 1 2; do
    DONE=$STAGE/final/${NM}_s${SEED}.done
    [[ -f "$DONE" ]] && continue
    S=$(acquire)
    (
      OUT=$(train_actor "$S" "$NM" "$AM" "$MW" "$LR" "$SEED")
      eval_actor "$S" "$OUT" "$NM" "$SEED" rh1 "42 43 44" >/dev/null
      touch "$DONE"
      rmdir "$SLOT/$S" 2>/dev/null || true
    ) &
  done
  wait
done < "$SELECTED"

echo "[cube-staged] cube/$BASE shard $SHARD_INDEX complete"
