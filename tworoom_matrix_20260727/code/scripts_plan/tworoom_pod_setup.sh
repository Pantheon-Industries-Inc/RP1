#!/bin/bash
# Bring a fresh pod up to the point where tworoom_matrix.sh can run its anchor.
#
# Steps (each idempotent, safe to re-run):
#   1. env sanity + python deps
#   2. fetch the AUTHORS' canonical TwoRoom dataset from HF and extract it
#   3. preflight: print tworoom.h5's real columns and assert the ones we need
#   4. convert the three pretrained bases -> lejepa_tworoom / pldm_tworoom /
#      dinowm_tworoom
#
# Run AFTER the repo and checkpoints/tworoom/pretrained/*.tar.zst are synced.
# rsync from the laptop needs -rlptDz, NOT -a: this network FS forbids chown.
#
# Usage: tworoom_pod_setup.sh [gpu]
set -u
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/swm_home}
export CODE=${CODE:-/workspace/code/stable-worldmodel}
export PYTHONPATH=$CODE
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1   # the 3.4 GB pull is painful without it
export TQDM_DISABLE=1
export MUJOCO_GL=${MUJOCO_GL:-osmesa}

GPU=${1:-0}
PY=python3
CANON_DIR=${CANON_DIR:-/workspace/datasets_canon/tworoom}
CANON=$CANON_DIR/tworoom.h5
HF_REPO=quentinll/lewm-tworooms        # NOTE: plural "tworooms"
HF_FILE=tworoom.tar.zst
LOGS=/workspace/logs
STATE_KEY=${STATE_KEY:-pos_agent}

mkdir -p "$LOGS" "$CANON_DIR" "$STABLEWM_HOME"
DRV=$LOGS/pod_setup.log
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][setup] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }

# ------------------------------------------------------------------- 1. env
log "host $(hostname); $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
log "torch: $($PY -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1)"
$PY -c "import hdf5plugin" 2>/dev/null || pip install -q hdf5plugin
$PY -c "import hf_transfer" 2>/dev/null || pip install -q hf_transfer
$PY -c "import lance" 2>/dev/null || die "pylance missing -- install the repo deps first"
export HDF5_PLUGIN_PATH=$($PY -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
log "hdf5 plugins at $HDF5_PLUGIN_PATH"

# --------------------------------------------------------------- 2. dataset
# The authors' tworoom.h5 -- the dataset the released bases were TRAINED on and
# the one their benchmark evaluates against. Our tworoom_play.lance is a noised
# in-house variant (ExpertPolicy noise 2.0) and is NOT interchangeable: the
# eval draws tasks by replaying dataset rows, so the dataset is the eval set.
if [ ! -f "$CANON" ]; then
  TARBALL=$CANON_DIR/$HF_FILE
  if [ ! -f "$TARBALL" ]; then
    log "downloading $HF_REPO/$HF_FILE (3.43 GB)"
    $PY - "$HF_REPO" "$HF_FILE" "$CANON_DIR" <<'PY' 2>&1 | tail -3 | tee -a "$DRV"
import sys
from huggingface_hub import hf_hub_download
repo, fname, dest = sys.argv[1:4]
p = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset",
                    local_dir=dest)
print("downloaded ->", p)
PY
    [ -f "$TARBALL" ] || die "download failed"
  fi
  log "extracting $HF_FILE"
  tar -I zstd --no-same-owner --no-same-permissions -xf "$TARBALL" -C "$CANON_DIR" \
    || die "extract failed"
  # the tarball may nest one level; surface tworoom.h5 at $CANON
  if [ ! -f "$CANON" ]; then
    FOUND=$(find "$CANON_DIR" -name "tworoom.h5" -o -name "tworoom.lance" -type d | head -1)
    [ -n "$FOUND" ] || die "no tworoom.h5/.lance in the tarball; contents: $(ls -R "$CANON_DIR" | head -20)"
    log "found dataset at $FOUND"
    CANON=$FOUND
  fi
fi
log "canonical dataset: $CANON ($(du -sh "$CANON" | cut -f1))"

# -------------------------------------------------------------- 3. preflight
# Column names differ by convention (authors' pos_agent vs our play-data
# state), and a wrong state key caches silently wrong states for hours.
$PY - "$CANON" "$STATE_KEY" <<'PY' 2>&1 | tee -a "$DRV"
import sys
import stable_worldmodel as swm
path, key = sys.argv[1], sys.argv[2]
ds = swm.data.load_dataset(path, transform=None)
cols = list(getattr(ds, "column_names", None) or ds.schema.names)
print(f"[preflight] {path}")
print(f"[preflight] {len(cols)} columns -> {cols}")
try:
    print(f"[preflight] rows={ds.count_rows()}")
except Exception:
    pass
missing = [k for k in (key, "pixels", "action") if k not in cols]
if missing:
    raise SystemExit(
        f"[preflight] FATAL missing {missing}. Available: {cols}. "
        "Set STATE_KEY to whichever column holds the agent position, and use "
        "the same value for eval.callables in the plan config."
    )
extra = [k for k in ("proprio",) if k not in cols]
if extra:
    print(f"[preflight] WARNING no proprio column -- the dinowm (proprio) base "
          f"cannot be evaluated on this dataset; only dinowm_noprop could.")
print(f"[preflight] ok: '{key}', pixels, action present")
PY
grep -q "preflight. ok" "$DRV" || die "preflight failed -- fix the state key before caching"

# ------------------------------------------------------------- 4. conversion
# lewm.tar.zst -> lejepa_tworoom (the ckpt inside is named lejepa_weights.ckpt)
if [ ! -d "$STABLEWM_HOME/checkpoints/dinowm_tworoom" ]; then
  log "converting pretrained bases (lejepa, pldm, dinowm)"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$CODE/scripts/plan/convert_tworoom_bases.py" \
    --only lejepa,pldm,dinowm > "$LOGS/convert_bases.log" 2>&1 \
    || die "conversion failed (see $LOGS/convert_bases.log)"
fi
for c in lejepa_tworoom pldm_tworoom dinowm_tworoom; do
  [ -d "$STABLEWM_HOME/checkpoints/$c" ] && log "ckpt ok: $c" || log "MISSING ckpt: $c"
done

log "SETUP DONE. Next:"
log "  CANON=$CANON tworoom_matrix.sh lejepa 0 8 anchor   # 3 evals, expect ~87"
log "  CANON=$CANON tworoom_matrix.sh pldm   1 8 anchor   #          expect ~97"
log "  CANON=$CANON tworoom_matrix.sh dinowm 2 8 anchor   #          expect ~100"
log "  then, only if the anchors land: tworoom_phase1_matrix.sh <base> <gpu>"
log "  followed by tworoom_matrix.sh <base> <gpu> 8 full"
