#!/bin/bash
# COMPOSITE NIGHT: three queued runs packed onto one box, using the co-residency
# pattern the repair driver and the f13 chain proved safe (training touches no
# EGL; only renders serialize).
#
#   lane A (GPU 0):   POOL CARD -- phi=0.08 mix on the COMBINED pool
#                     (onpolicy_full_lab + chained; 132.5k distinct rows,
#                     K_fail~5 vs POST-full's uniform K=18) -> fine-tune ->
#                     caches -> TD -> [join] LIP pool s0-2 on GPUs 0-2.
#                     Tests POOL QUALITY (distinctness + chained depth) at
#                     fixed mixture vs dsp_postfull. The variable both the
#                     f13 null and full-thin=+2.0 point at.
#   lane B (GPU 1-3): PLDM base bring-up -- caches/TD under PLDM_OgBench_lewm
#                     (train split), LIPv4 s0-2, canonical recipe. CAVEAT
#                     (pre-registered): amax 1.6 was tuned on LeWM; the old
#                     PLDM campaign preferred 3.5 under the old recipe. A low
#                     PLDM-LIP here motivates an amax mini-sweep, not a
#                     conclusion. Then the lr_sweep arms (it16/s12k/alr,
#                     amax 1.6, train-split caches) -- does harder
#                     optimization help now that the clip caps exploitation?
#   lane C (GPU 0):   one strictly-serial egl eval queue: 9 pool + 9 pldm +
#                     27 lrsw cells, ep_range 8000:10000 asserted per cell.
#
# ETA ~7.5 h: FT 4-4.5 h dominates lane A; lane B finishes inside that window.
#
# Usage: bash composite_pool_pldm_lrsw.sh [check]   env: SKIP_PGREP=1
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_split
SUM=$R/summary_dynasplit.csv; DRV=$L/driver_composite.log
V2WM=/workspace/models/v2WM; V2W=$V2WM/weights_epoch_22.pt
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; AMAX=1.6; DRAWS="42 43 44"; SEEDS="0 1 2"
# v2 train-split assets (exist)          # pool products
FS1=/workspace/caches/v2_tr${EPHI}_fs1.pt;   POOLMIX=$D/mix_pool_f08_5050.lance
FS5=/workspace/caches/v2_tr${EPHI}_fs5.pt;   WMP=/workspace/models/dyna_pool_5050
TD=/workspace/metrics/v2_tr${EPHI}_TD.pt;    PF1F=/workspace/caches/pool_full_fs1.pt
PF1=/workspace/caches/pool_tr${EPHI}_fs1.pt; PF5=/workspace/caches/pool_tr${EPHI}_fs5.pt
PTD=/workspace/metrics/pool_TD.pt
# pldm products
QF1F=/workspace/caches/pldm_full_fs1.pt; QF1=/workspace/caches/pldm_tr${EPHI}_fs1.pt
QF5=/workspace/caches/pldm_tr${EPHI}_fs5.pt; QTD=/workspace/metrics/pldm_TD.pt
mkdir -p "$L" "$R" "$D"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

# ------------------------------------------------------------------ preflight
log "=== COMPOSITE (pid $$): pool card + PLDM LIPv4 + lr_sweep ==="
for f in "$FS1" "$FS5" "$TD" "$V2W" "$V2WM/config.json" "$AH5" \
         /workspace/build_dyna_mix.py; do
  [ -e "$f" ] || die "preflight: missing $f"
done
[ -d "$EXPERT" ] || die "preflight: missing expert"
# PLDM is allowed to still be uploading at launch: nothing in lane A needs it,
# and lane B runs the lrsw waves first, then waits for the EXACT byte size
# (a partial scp is a valid-looking file; size is the only honest gate).
PLDM_BYTES=72266017
pldm_ready(){ [ -f "$PLDM/config.json" ] && \
  [ "$(stat -c%s "$PLDM/weights.pt" 2>/dev/null || echo 0)" = "$PLDM_BYTES" ]; }
if pldm_ready; then log "P0: PLDM checkpoint present"
else log "P0: PLDM upload incomplete ($(stat -c%s "$PLDM/weights.pt" 2>/dev/null || echo 0)/$PLDM_BYTES bytes) -- lane B will wait for it"; fi
for a in 0 1 2; do
  [ -d "$D/onpolicy_full_a${a}_lab.lance" ] || die "preflight: labeled lance a$a missing"
  [ -d "$D/chained_a${a}.lance" ] || die "preflight: chained lance a$a missing"
done
grep -q "failure-frac" /workspace/build_dyna_mix.py || die "preflight: old mixer on pod"
if [ "${SKIP_PGREP:-0}" != 1 ]; then
  pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null && die "preflight: something is already running"
fi
log "P0: preflight OK"
[ "${1:-}" = check ] && { log "check mode: exiting"; exit 0; }

lip_train(){ # gpu seed wm cache_fs5 cache_fs1 td out tag extra...
  local gpu=$1 seed=$2 wm=$3 c5=$4 c1=$5 td=$6 out=$7 tag=$8; shift 8
  [ -f "$out" ] && { log "  $tag/s$seed reusing"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$c5" --cache-td "$c1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
    --horizon 5 --n-step 50 --amax "$AMAX" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --arch v4 --seed "$seed" "$@" \
    --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/cmp_lip_${tag}_s${seed}.log" 2>&1 || { log "  $tag/s$seed TRAIN FAILED"; return 1; }
  log "  $tag/s$seed trained"
}
BASE_OPT=(--iters 8 --steps 6000 --actor-lr 3e-4 --actor-lr-final 3e-5)

# ================================================================ lane B (bg)
lane_b(){
  # ---- B0: lrsw arms FIRST -- they need only v2 assets, so they run while
  # the PLDM checkpoint may still be uploading.
  local arm g s
  for arm in it16 s12k alr; do
    case "$arm" in
      it16) EXTRA=(--iters 16 --steps 6000  --actor-lr 3e-4 --actor-lr-final 3e-5) ;;
      s12k) EXTRA=(--iters 8  --steps 12000 --actor-lr 3e-4 --actor-lr-final 3e-5) ;;
      alr)  EXTRA=(--iters 8  --steps 6000  --actor-lr 1e-4 --actor-lr-final 1e-5) ;;
    esac
    log "B0: lrsw $arm wave"
    g=1
    for s in $SEEDS; do
      lip_train "$g" "$s" "$V2WM" "$FS5" "$FS1" "$TD" \
        "/workspace/actors/lip4_lrsw_${arm}_s${s}.pt" "lrsw_${arm}" "${EXTRA[@]}" & g=$((g+1))
    done; wait
  done
  log "B0: lrsw waves done"
  # ---- B0b: wait for the PLDM upload (exact size), cap 3h
  local T0; T0=$(date +%s)
  until pldm_ready; do
    if [ $(( $(date +%s) - T0 )) -gt 10800 ]; then
      log "B0b: PLDM checkpoint never completed -- SKIPPING the PLDM lane"
      touch "$D/COMPOSITE_B_DONE"; return 0
    fi
    sleep 60
  done
  log "B0b: PLDM checkpoint complete"
  # ---- B1: PLDM caches + TD (GPU 1)
  if [ ! -f "$QF1" ]; then
    log "B1: caching fs1 under PLDM (GPU 1)"
    [ -f "$QF1F" ] || CUDA_VISIBLE_DEVICES=1 timeout 28800 python3 "$TRM/cache_latents.py" \
      --wm "$PLDM" --dataset "$EXPERT" --out "$QF1F" --state-key privileged_block_0_pos \
      > "$L/cmp_cache_pldm_fs1.log" 2>&1 || { log "B1: PLDM cache FAILED"; return 1; }
    python3 /workspace/filter_cache_eprange.py "$QF1F" "$QF1" --lo 0 --hi "$EPHI" \
      > "$L/cmp_filter_pldm.log" 2>&1 || { log "B1: PLDM filter FAILED"; return 1; }
  fi
  [ -f "$QF5" ] || python3 "$TRM/subsample_cache.py" --in "$QF1" --out "$QF5" --frameskip 5 \
    > "$L/cmp_cache_pldm_fs5.log" 2>&1 || { log "B1: PLDM fs5 FAILED"; return 1; }
  [ -f "$QTD" ] || CUDA_VISIBLE_DEVICES=1 timeout 28800 python3 "$P/train_metric.py" \
    --cache "$QF1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
    --steps 6000 --seed 0 --out "$QTD" > "$L/cmp_td_pldm.log" 2>&1 \
    || { log "B1: PLDM TD FAILED"; return 1; }
  log "B1: PLDM caches + TD ready"
  # ---- B2: PLDM LIPv4 wave (GPUs 1-3)
  log "B2: PLDM LIPv4 seeds $SEEDS"
  local g=1 s
  for s in $SEEDS; do
    lip_train "$g" "$s" "$PLDM" "$QF5" "$QF1" "$QTD" \
      "/workspace/actors/lip4_pldm_s${s}.pt" pldm "${BASE_OPT[@]}" & g=$((g+1))
  done; wait
  log "B2: PLDM actors done"
  touch "$D/COMPOSITE_B_DONE"
}
rm -f "$D/COMPOSITE_B_DONE"
lane_b &
LANE_B_PID=$!

# ================================================================ lane A (fg)
# ---- A1: phi=0.08 mix on the combined pool (per-lance keeps, K_fail~5)
if [ ! -f "$POOLMIX/.done" ]; then
  log "A1: mix phi=0.08 on combined pool (6 labeled lances)"
  python3 /workspace/build_dyna_mix.py --expert "$EXPERT" \
    --onpolicy $D/onpolicy_full_a0_lab.lance $D/onpolicy_full_a1_lab.lance $D/onpolicy_full_a2_lab.lance \
               $D/chained_a0.lance $D/chained_a1.lance $D/chained_a2.lance \
    --out "$POOLMIX" --onpolicy-frac 0.5 --failure-frac 0.08 --expert-ep-hi "$EPHI" \
    > "$L/cmp_mix_pool.log" 2>&1 || die "pool mix failed"
  grep -q BUILD_MIX_DONE "$L/cmp_mix_pool.log" || die "pool mix incomplete"
  grep -q "\[split\] expert restricted" "$L/cmp_mix_pool.log" || die "expert slice NOT restricted"
  touch "$POOLMIX/.done"
fi
log "A1: $(grep -h 'dup K_fail=' "$L/cmp_mix_pool.log" | tail -1)"
log "A1: $(grep -h 'achieved:' "$L/cmp_mix_pool.log" | tail -1)"
ACH_ON=$(grep -h 'achieved:' "$L/cmp_mix_pool.log" | tail -1 | grep -oE 'on-policy [0-9.]+' | grep -oE '[0-9.]+')
awk -v x="${ACH_ON:-0}" 'BEGIN{exit !(x>=0.48 && x<=0.52)}' \
  || die "A1: achieved on-policy ${ACH_ON:-unparsed} outside [0.48,0.52]"

# ---- A2: fine-tune v2WM -> dyna_pool_5050
if [ ! -f "$WMP/weights_epoch_1.pt" ]; then
  log "A2: fine-tune v2WM -> dyna_pool_5050 (~4-4.5h)"
  INIT_WEIGHTS=$V2W CUDA_VISIBLE_DEVICES=0 timeout 86400 python3 \
    scripts/train/lewm_expert.py data=ogb data.dataset.name="$POOLMIX" \
    "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
    "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
    trainer.devices=1 output_model_name=dyna_pool_5050 subdir=dyna_pool_5050 \
    +action_stats_pin=expert wandb.enabled=false > "$L/cmp_ft_pool.log" 2>&1 || die "fine-tune failed"
  CK=/workspace/swm_home/checkpoints/dyna_pool_5050
  mkdir -p "$WMP"
  cp "$CK/weights_epoch_1.pt" "$WMP/" || die "no weights_epoch_1.pt in $CK"
  cp "$V2WM/config.json" "$WMP/config.json" || die "no arch config"
fi
cmp -s "$WMP/config.json" "$V2WM/config.json" || die "pool config differs from arch config"
log "A2: WM_pool ready"

# ---- A3: caches + TD under WM_pool (GPU 0)
if [ ! -f "$PF1" ]; then
  [ -f "$PF1F" ] || { log "A3: caching fs1 under WM_pool";
    CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$TRM/cache_latents.py" \
      --wm "$WMP" --dataset "$EXPERT" --out "$PF1F" --state-key privileged_block_0_pos \
      > "$L/cmp_cache_pool_fs1.log" 2>&1 || die "pool cache failed"; }
  python3 /workspace/filter_cache_eprange.py "$PF1F" "$PF1" --lo 0 --hi "$EPHI" \
    > "$L/cmp_filter_pool.log" 2>&1 || die "pool filter failed"
fi
[ -f "$PF5" ] || python3 "$TRM/subsample_cache.py" --in "$PF1" --out "$PF5" --frameskip 5 \
  > "$L/cmp_cache_pool_fs5.log" 2>&1 || die "pool fs5 failed"
[ -f "$PTD" ] || CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_metric.py" \
  --cache "$PF1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
  --steps 6000 --seed 0 --out "$PTD" > "$L/cmp_td_pool.log" 2>&1 || die "pool TD failed"
log "A3: pool caches + TD ready"

# ---- join, then A4: LIP pool wave on GPUs 0-2
log "join: waiting for lane B (PID $LANE_B_PID)"
while [ ! -f "$D/COMPOSITE_B_DONE" ]; do
  kill -0 "$LANE_B_PID" 2>/dev/null || { log "WARNING: lane B died without marker; continuing"; break; }
  sleep 120
done
log "A4: LIP pool wave (GPUs 0-2)"
g=0; for s in $SEEDS; do
  lip_train "$g" "$s" "$WMP" "$PF5" "$PF1" "$PTD" \
    "/workspace/actors/lip4_pool_s${s}.pt" pool "${BASE_OPT[@]}" & g=$((g+1))
done; wait
log "A4: pool actors done"

# ================================================================ lane C evals
run_eval(){ # name wm actor draw
  local nm=$1 wm=$2 actor=$3 d=$4
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  [ -f "$actor" ] || { log "  $nm: actor missing, FAIL"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$wm" solver=lip "solver.actor_path=$actor" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" \
    || { log "  $nm: ep_range NOT APPLIED"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
}
log "C: eval queue (45 cells, sequential, egl, held-out $EVAL_RANGE)"
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" >/dev/null; do sleep 60; done
for s in $SEEDS; do for d in $DRAWS; do
  run_eval "dsp_pool_s${s}_e${d}" "$WMP" "/workspace/actors/lip4_pool_s${s}.pt" "$d"
done; done
for s in $SEEDS; do for d in $DRAWS; do
  run_eval "pldm_lip_s${s}_e${d}" "$PLDM" "/workspace/actors/lip4_pldm_s${s}.pt" "$d"
done; done
for arm in it16 s12k alr; do for s in $SEEDS; do for d in $DRAWS; do
  run_eval "lrsw_${arm}_s${s}_e${d}" "$V2WM" "/workspace/actors/lip4_lrsw_${arm}_s${s}.pt" "$d"
done; done; done

# ================================================================ final card
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
def seeds3(arm):
    ms = [sm(arm, s) for s in (0, 1, 2)]
    return None if any(m is None for m in ms) else ms
print("=== COMPOSITE CARD (held-out 8000:10000, amax 1.6, egl) ===")
for arm in ("dsp_pre", "dsp_postfull", "dsp_f13", "dsp_pool", "pldm_lip",
            "lrsw_it16", "lrsw_s12k", "lrsw_alr"):
    ms = seeds3(arm)
    if ms: print(f"  {arm:14s} {' '.join(f'{m:5.1f}' for m in ms)}   mean {sum(ms)/3:.1f}")
pool, full = seeds3("dsp_pool"), seeds3("dsp_postfull")
if pool and full:
    d = [a - b for a, b in zip(pool, full)]
    m = sum(d)/3; sd = math.sqrt(sum((x-m)**2 for x in d)/2)
    t = m/(sd/math.sqrt(3)) if sd > 0 else float("inf")
    print(f"  POOL vs FULL (same phi=0.08; K 18 -> ~5, +chained depth): "
          f"delta {m:+.2f} sd {sd:.2f} t {t:.2f} (df=2)")
    print("  reading: pool > full => distinctness/coverage is the lever, "
          "chain more; pool ~ full => WM fine-tune saturated at this mixture.")
pre6 = [sm("dsp_pre", s) for s in range(6)]
if all(v is not None for v in pre6):
    print(f"  reference dsp_pre 6-seed mean: {sum(pre6)/6:.1f}")
print("COMPOSITE_DONE")
PY
log "done"
