#!/bin/bash
# TD teacher sweep on mean-pooled upstream DINO-WM latents.
#
# Axes: --expectile (the IQL tau; low = optimistic toward the min, which is what
# a cost-to-go/quasimetric wants) x --lr. Everything else is the canonical cube
# recipe: quasimetric head, n-step 50, 6000 steps, seed 0.
#
# NB TDConfig also has a field literally called `tau` -- that is the
# target-network Polyak rate, exposed here as --polyak-tau and held at its
# 0.005 default. It is NOT the expectile.
#
# Training is 95 s/cell, so all 12 cells are cheap and run 4-way across GPUs.
# A full TD+CEM eval is ~33 min/cell, so cells are screened offline first
# (probe_metric.py on HELD-OUT episodes >= 8000) and only the best few get
# scored by the real planner.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export TQDM_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; M=/workspace/metrics
FULL=/workspace/caches/dino_full_fs1.pt
TR=/workspace/caches/dino_tr8000_fs1.pt
SUM=$R/summary_dino_td.csv
mkdir -p "$L" "$R" "$M"; touch "$SUM"; cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_dino_td.log"; }

EXPECTILES="0.01 0.03 0.1 0.5"
LRS="3e-4 1e-3 3e-3"

[ -f "$TR" ] || { log "FATAL: $TR missing (run the filter first)"; exit 1; }

# ------------------------------------------------------------------ train, 4-way
log "=== TD sweep: expectile x lr, 12 cells, canonical head/n-step/steps ==="
i=0
for e in $EXPECTILES; do for lr in $LRS; do
  tag="e${e}_lr${lr}"
  out=$M/dino_td_${tag}.pt
  if [ -f "$out" ]; then log "  $tag cached"; continue; fi
  g=$((i % 4))
  CUDA_VISIBLE_DEVICES=$g python3 "$P/train_metric_sweep.py" --cache "$TR" \
    --learner td --head quasimetric --expectile "$e" --lr "$lr" \
    --n-step 50 --steps 6000 --seed 0 --out "$out" \
    > "$L/dino_td_${tag}.log" 2>&1 &
  i=$((i + 1))
  [ $((i % 4)) -eq 0 ] && wait
done; done
wait
log "  trained: $(ls -1 $M/dino_td_e*_lr*.pt 2>/dev/null | wc -l)/12"

# ------------------------------------------------------------------ offline screen
log "=== offline screen on held-out episodes >= 8000 ==="
for e in $EXPECTILES; do for lr in $LRS; do
  tag="e${e}_lr${lr}"
  out=$M/dino_td_${tag}.pt
  [ -f "$out" ] || { log "  $tag MISSING (train failed)"; continue; }
  grep -q "^${tag}," "$SUM" && continue
  line=$(CUDA_VISIBLE_DEVICES=0 python3 /workspace/probe_metric.py --cache "$FULL" \
    --metric "$out" --ep-lo 8000 --tag "$tag" 2>/dev/null | grep "^PROBE" | tail -1)
  if [ -z "$line" ]; then log "  $tag probe FAILED"; echo "${tag},NA,NA,NA" >> "$SUM"; continue; fi
  sp=$(echo "$line" | grep -oE "spearman=[-0-9.]+" | cut -d= -f2)
  pa=$(echo "$line" | grep -oE "pair_acc=[-0-9.]+" | cut -d= -f2)
  mo=$(echo "$line" | grep -oE "monotone=[-0-9.]+" | cut -d= -f2)
  echo "${tag},${sp},${pa},${mo}" >> "$SUM"
  log "  $tag spearman=$sp pair_acc=$pa monotone=$mo"
done; done

log ""
log "=== screen ranked by pair_acc (planner-relevant ordering) ==="
sort -t, -k3 -gr "$SUM" | head -14 | tee -a "$L/driver_dino_td.log"
log "DINO_TD_SWEEP_DONE"
