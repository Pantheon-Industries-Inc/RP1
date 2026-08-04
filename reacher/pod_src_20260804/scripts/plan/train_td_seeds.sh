#!/bin/bash
# 3-seed TD values, so the TD rows get the same seed treatment as LIP.
# Seed 0 already exists under the legacy filename; alias it to _s0 and train
# s1, s2. Covers pre-Dyna and post-Dyna WMs for both bases.
# Caches were built on dmc/reacher_random.lance, verified row-for-row identical
# to the canonical reacher.h5 content (our row t == canonical row t+1, action
# pairing consistent to 3e-08), so they are valid training inputs.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PLAN=/workspace/stable-worldmodel/scripts/plan
MET=/workspace/metrics
C=/workspace/caches
D=/workspace/logs/train_td_seeds.log
mkdir -p "$MET" /workspace/logs
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$D"; }

run(){ # tag cache legacy gpu
  local tag=$1 cache=$2 legacy=$3 gpu=$4
  [ -f "$cache" ] || { log "$tag: MISSING cache $cache"; return 1; }
  if [ ! -f "$MET/td_${tag}_s0.pt" ]; then
    [ -f "$legacy" ] && cp "$legacy" "$MET/td_${tag}_s0.pt" && log "$tag s0: aliased from legacy"
  fi
  for s in 1 2; do
    local out=$MET/td_${tag}_s${s}.pt
    [ -f "$out" ] && { log "$tag s$s: cached"; continue; }
    CUDA_VISIBLE_DEVICES=$gpu python3 "$PLAN/train_metric.py" --cache "$cache" \
      --learner td --head quasimetric --expectile 0.1 --n-step 50 --steps 6000 \
      --seed "$s" --out "$out" > "/workspace/logs/td_${tag}_s${s}.log" 2>&1 \
      && log "$tag s$s: done" || log "$tag s$s: FAILED"
  done
}

log "=== training TD seeds (pre-Dyna bases)"
run lejepa_pre "$C/reacher_lejepa_fs1.pt" "$MET/td_reacher_lejepa_e0.1_n50.pt" 0 &
run pldm_pre   "$C/reacher_pldm_fs1.pt"   "$MET/td_reacher_pldm_e0.1_n50.pt"   1 &
wait
log "=== training TD seeds (post-Dyna WMs)"
run lejepa_post "$C/reacher_r1_8020_fs1.pt" "$MET/td_reacher_r1_8020_e0.1_n50.pt" 0 &
run pldm_post   "$C/reacher_r1_pldm_fs1.pt" "$MET/td_reacher_r1_pldm_e0.1_n50.pt" 1 &
wait
log "TD values now present: $(ls "$MET"/td_*_s[012].pt 2>/dev/null | wc -l) (expect 12)"
log "TD_SEEDS_DONE"
