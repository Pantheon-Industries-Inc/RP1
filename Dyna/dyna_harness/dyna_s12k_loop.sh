#!/bin/bash
# DYNA LOOP ON THE s12k BASE -- the best-results protocol (= the POST-full arm:
# episode-disjoint 0:7999/8000:9999, terminate_at_goal=False collection,
# uniform-K 50/50 mix, lr 1e-5 x 2 epochs with epoch 1 pre-registered), with
# the s12k recipe (--steps 12000) carried through to the POST actors.
#
# PRE arm = lrsw_s12k (88.7, rows already in the CSV) -- no PRE evals needed.
# The on-policy data comes from the lrsw_s12k actors themselves (that is the
# loop). Collection seed band 2000+ (dsp used 1000+) for clean provenance.
#
# Doubles as a plateau probe: Dyna is worth ~+4 (3 replications) and s12k was
# +0.9 over 6k-PRE. Additive => ~92.5-93; if 92 is the eval's ceiling, this
# lands ~92 = POST-full. Either answer is informative.
#
# WAITS for the PLDM take-2 lane to finish (collection renders; the box must
# be single-purpose). ~1.4h collect + ~3.5h FT + 25min caches/TD + ~1.6h
# s12k wave + ~20min evals => card ~7h after it starts (~11:30 UTC).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_split
SUM=$R/summary_dynasplit.csv; DRV=$L/driver_s12kloop.log
V2WM=/workspace/models/v2WM; V2W=$V2WM/weights_epoch_22.pt
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; COLLECT_RANGE="0:${EPHI}"
AMAX=1.6; SEEDS="0 1 2"; DRAWS="42 43 44"; NCALL=12
MIX=$D/mix_s12k_5050.lance
WMS=/workspace/models/dyna_s12k_5050
SF1F=/workspace/caches/s12k_full_fs1.pt
SF1=/workspace/caches/s12k_tr${EPHI}_fs1.pt
SF5=/workspace/caches/s12k_tr${EPHI}_fs5.pt
STD=/workspace/metrics/s12k_TD.pt
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

# ------------------------------------------------------------------ P0 wait+preflight
log "=== DYNA LOOP, s12k BASE (pid $$): waiting for the PLDM lane ==="
T0=$(date +%s)
until grep -q "PLDM_NIGHT2_DONE" "$L/driver_pldm2.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 21600 ] && die "PLDM lane never finished in 6h"
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 120; done
log "box quiet"
for s in $SEEDS; do
  [ -f "/workspace/actors/lip4_lrsw_s12k_s${s}.pt" ] || die "s12k actor s$s missing"
done
for f in "$V2W" "$V2WM/config.json" "$AH5" /workspace/build_dyna_mix.py; do
  [ -e "$f" ] || die "missing $f"
done
[ -d "$EXPERT" ] || die "missing expert"
grep -q "SWM_RECORD_OUTCOME" stable_worldmodel/world/world.py || die "world.py lacks the outcome recorder"
log "P0: preflight OK"
[ "${1:-}" = check ] && { log "check mode: exiting"; exit 0; }

# ------------------------------------------------- P1 collection (s12k actors)
if [ ! -f "$D/S12K_COLLECT_DONE" ]; then
  log "P1: collection, ${NCALL} calls x 3 s12k actors, terminate_at_goal=False, ep_range=$COLLECT_RANGE"
  for i in $(seq 0 $((NCALL-1))); do for a in $SEEDS; do
    rec="$D/onpolicy_s12k_a${a}.lance"; lg="$L/s12kcol_a${a}_c${i}.log"
    grep -q "kept=" "$lg" 2>/dev/null && continue
    SWM_RECORD_PATH=$rec SWM_RECORD_OUTCOME=1 CUDA_VISIBLE_DEVICES=0 timeout 10800 \
    python3 "$P/eval_wm.py" --config-name cube seed=$((2000 + a*100 + i)) \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=50 \
      "+eval.ep_range=$COLLECT_RANGE" world.terminate_at_goal=False \
      policy="$V2WM" solver=lip \
      "solver.actor_path=/workspace/actors/lip4_lrsw_s12k_s${a}.pt" \
      output.filename="s12kcol_a${a}_c${i}.txt" > "$lg" 2>&1
    grep -q "ep_range 0:${EPHI}" "$lg" || die "call a$a/c$i: ep_range not applied"
    rec_line=$(grep -h "\[record\]" "$lg" | tail -1)
    echo "$rec_line" | grep -q "kept=50 dropped=0" || die "call a$a/c$i: unexpected keep/drop -- $rec_line"
    echo "$rec_line" | grep -q "kept_success=" || die "call a$a/c$i: outcome label missing -- $rec_line"
    log "  a$a/c$i: $(echo "$rec_line" | grep -oE 'kept_success=[0-9]+')"
  done; done
  touch "$D/S12K_COLLECT_DONE"
fi
log "P1: collection done"

# --------------------------------------------------------------- P2 mix (uniform K)
if [ ! -f "$MIX/.done" ]; then
  log "P2: 50/50 mix, uniform K (the best-results recipe), expert < $EPHI"
  python3 /workspace/build_dyna_mix.py --expert "$EXPERT" \
    --onpolicy $D/onpolicy_s12k_a0.lance $D/onpolicy_s12k_a1.lance $D/onpolicy_s12k_a2.lance \
    --out "$MIX" --onpolicy-frac 0.5 --expert-ep-hi "$EPHI" > "$L/s12k_mix.log" 2>&1 || die "mix failed"
  grep -q BUILD_MIX_DONE "$L/s12k_mix.log" || die "mix incomplete"
  grep -q "\[split\] expert restricted" "$L/s12k_mix.log" || die "expert slice NOT restricted"
  touch "$MIX/.done"
fi
log "P2: $(grep -h 'dup K=' "$L/s12k_mix.log" | tail -1)"

# ------------------------------------------------------------ P3 WM fine-tune
if [ ! -f "$WMS/weights_epoch_1.pt" ]; then
  log "P3: fine-tune v2WM -> dyna_s12k_5050 (lr 1e-5, 2 epochs, epoch 1)"
  INIT_WEIGHTS=$V2W CUDA_VISIBLE_DEVICES=0 timeout 86400 python3 \
    scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
    "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
    "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
    trainer.devices=1 output_model_name=dyna_s12k_5050 subdir=dyna_s12k_5050 \
    +action_stats_pin=expert wandb.enabled=false > "$L/s12k_ft.log" 2>&1 || die "fine-tune failed"
  CK=/workspace/swm_home/checkpoints/dyna_s12k_5050
  mkdir -p "$WMS"
  cp "$CK/weights_epoch_1.pt" "$WMS/" || die "no weights_epoch_1.pt"
  cp "$V2WM/config.json" "$WMS/config.json" || die "no arch config"
fi
cmp -s "$WMS/config.json" "$V2WM/config.json" || die "config differs from arch config"
log "P3: WM_s12k ready"

# ------------------------------------------------------- P4 caches + TD
if [ ! -f "$SF1" ]; then
  [ -f "$SF1F" ] || { log "P4: caching fs1 under WM_s12k";
    CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$TRM/cache_latents.py" \
      --wm "$WMS" --dataset "$EXPERT" --out "$SF1F" --state-key privileged_block_0_pos \
      > "$L/s12k_cache_fs1.log" 2>&1 || die "cache failed"; }
  python3 /workspace/filter_cache_eprange.py "$SF1F" "$SF1" --lo 0 --hi "$EPHI" \
    > "$L/s12k_filter.log" 2>&1 || die "filter failed"
fi
[ -f "$SF5" ] || python3 "$TRM/subsample_cache.py" --in "$SF1" --out "$SF5" --frameskip 5 \
  > "$L/s12k_cache_fs5.log" 2>&1 || die "fs5 failed"
[ -f "$STD" ] || CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_metric.py" \
  --cache "$SF1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
  --steps 6000 --seed 0 --out "$STD" > "$L/s12k_td.log" 2>&1 || die "TD failed"
log "P4: caches + TD ready"

# -------------------------------------------- P5 POST actors (s12k recipe)
log "P5: POST actors, --steps 12000, GPUs 0-2"
g=0; for s in $SEEDS; do
  out=/workspace/actors/lip4_s12kpost_s${s}.pt
  if [ ! -f "$out" ]; then
    CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
      --cache "$SF5" --cache-td "$SF1" --h5 "$AH5" --wm "$WMS" --init-value "$STD" \
      --horizon 5 --iters 8 --steps 12000 --n-step 50 --amax "$AMAX" \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$s" \
      --out "$out" --out-value "${out%.pt}_value.pt" \
      > "$L/s12k_lip_post_s${s}.log" 2>&1 || log "  post/s$s TRAIN FAILED" &
  fi
  g=$((g+1))
done; wait
log "P5: POST actors done"

# ---------------------------------------------------------------- P6 evals
run_eval(){ # name actor draw
  local nm=$1 actor=$2 d=$3
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  [ -f "$actor" ] || { log "  $nm: actor missing, FAIL"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$WMS" solver=lip "solver.actor_path=$actor" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" \
    || { log "  $nm: ep_range NOT APPLIED"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
}
log "P6: POST evals (PRE = existing lrsw_s12k rows)"
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" >/dev/null; do sleep 60; done
for s in $SEEDS; do for d in $DRAWS; do
  run_eval "s12kpost_s${s}_e${d}" "/workspace/actors/lip4_s12kpost_s${s}.pt" "$d"
done; done

# ------------------------------------------------------------------ P7 card
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
print("=== s12k DYNA CARD (held-out 8000:10000, amax 1.6, egl) ===")
pre = [sm("lrsw_s12k", s) for s in (0, 1, 2)]
post = [sm("s12kpost", s) for s in (0, 1, 2)]
for nm, ms in (("PRE  (lrsw_s12k)", pre), ("POST (s12kpost)", post)):
    if all(m is not None for m in ms):
        print(f"  {nm}: {' '.join(f'{m:5.1f}' for m in ms)}   mean {sum(ms)/3:.1f}")
if all(m is not None for m in pre + post):
    d = [a - b for a, b in zip(post, pre)]
    m = sum(d)/3; sd = math.sqrt(sum((x-m)**2 for x in d)/2)
    t = m/(sd/math.sqrt(3)) if sd > 0 else float("inf")
    print(f"  Dyna on s12k: delta {m:+.2f} sd {sd:.2f} t {t:.2f} (df=2)")
    ref = [sm("dsp_postfull", s) for s in (0, 1, 2)]
    if all(r is not None for r in ref):
        print(f"  reference POST-full (6k recipe): {sum(ref)/3:.1f} -- s12kpost above it "
              "=> the 92-plateau was recipe-bound; equal => eval ceiling.")
print("S12K_LOOP_DONE")
PY
log "done"
