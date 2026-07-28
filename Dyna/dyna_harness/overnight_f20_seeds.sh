#!/bin/bash
# OVERNIGHT CHAIN: seeds 3-5 (PRE + POST-full) AND the phi=0.20 failure-fraction
# cell, composed the way dyna_repair_downstream.sh proved safe: the WM fine-tune
# holds GPU 0 while the LIP waves run on GPUs 1-3 (LIP trains on caches, no EGL),
# and every eval runs strictly serial at the end.
#
#   lane A (GPU 0):  relabel gate -> mix phi=0.20 -> fine-tune -> caches -> TD
#   lane B (GPU 1-3): PRE s3-5 wave, then POST-full s3-5 wave
#   join:            LIP f20 s0-2 on GPUs 0-2 (needs A's TD and B's GPUs)
#   lane C (GPU 0):  27 eval cells, sequential, egl, ep_range asserted per cell
#
# ETA (measured stage times, HANDOFF_20260728.md §0): FT 3.5-4.5 h dominates;
# total ~6.5-7 h.
#
# Usage:  bash overnight_f20_seeds.sh check   # preflight only, exit
#         bash overnight_f20_seeds.sh         # run (idempotent, phases skip
#                                             # on existing outputs)
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_split
SUM=$R/summary_dynasplit.csv; DRV=$L/driver_overnight.log
V2WM=/workspace/models/v2WM; V2W=$V2WM/weights_epoch_22.pt
WM2=/workspace/models/dyna_full_5050
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; AMAX=1.6; DRAWS="42 43 44"
NEW_SEEDS="3 4 5"; F20_SEEDS="0 1 2"
# PRE assets (arm 1)                # POST-full assets (repair driver)
FS1=/workspace/caches/v2_tr${EPHI}_fs1.pt;  RF1=/workspace/caches/rep_full_tr${EPHI}_fs1.pt
FS5=/workspace/caches/v2_tr${EPHI}_fs5.pt;  RF5=/workspace/caches/rep_full_tr${EPHI}_fs5.pt
TD=/workspace/metrics/v2_tr${EPHI}_TD.pt;   RTD=/workspace/metrics/rep_full_TD.pt
# phi=0.20 products
MIX=$D/mix_f20_5050.lance
WMF=/workspace/models/dyna_f20_5050
F20F1F=/workspace/caches/f20_full_fs1.pt
F20F1=/workspace/caches/f20_tr${EPHI}_fs1.pt
F20F5=/workspace/caches/f20_tr${EPHI}_fs5.pt
F20TD=/workspace/metrics/f20_TD.pt
mkdir -p "$L" "$R" "$D"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
cd "$CODE"

# ------------------------------------------------------------------ P0 preflight
log "=== OVERNIGHT CHAIN (pid $$): seeds 3-5 + phi=0.20 ==="
for f in "$FS1" "$FS5" "$TD" "$RF1" "$RF5" "$RTD" "$V2W" "$V2WM/config.json" \
         "$WM2/weights_epoch_1.pt" "$WM2/config.json" "$AH5" \
         /workspace/relabel_onpolicy.py /workspace/build_dyna_mix.py; do
  [ -e "$f" ] || die "preflight: missing $f"
done
[ -d "$EXPERT" ] || die "preflight: missing $EXPERT"
for a in 0 1 2; do [ -d "$D/onpolicy_full_a${a}.lance" ] || die "preflight: missing onpolicy_full_a${a}.lance"; done
for s in 0 1 2; do [ -f "/workspace/actors/lip4_dsp_pre_s${s}.pt" ] || die "preflight: PRE s${s} actor missing"; done
grep -q "failure-frac" /workspace/build_dyna_mix.py || die "preflight: pod build_dyna_mix.py lacks --failure-frac (deploy the new one)"
grep -q "ep_range" "$P/eval_wm.py" || die "preflight: eval_wm.py has no ep_range"
grep -q "_row_in_range" "$P/eval_wm.py" || die "preflight: eval_wm.py lacks the ep_range KeyError fix"
pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null && die "preflight: something is already running"
log "P0: preflight OK"
[ "${1:-}" = check ] && { log "check mode: exiting before any work"; exit 0; }

# ================================================================ lane B (bg)
# PRE s3-5 then POST-full s3-5 on GPUs 1-3, concurrent with lane A's fine-tune.
lip_train(){ # gpu seed wm cache_fs5 cache_fs1 td out tag
  local gpu=$1 seed=$2 wm=$3 c5=$4 c1=$5 td=$6 out=$7 tag=$8
  [ -f "$out" ] && { log "  $tag/s$seed reusing"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$c5" --cache-td "$c1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$seed" \
    --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/ovr_lip_${tag}_s${seed}.log" 2>&1 || { log "  $tag/s$seed TRAIN FAILED"; return 1; }
  log "  $tag/s$seed trained"
}
lane_b(){
  local g
  log "B1: PRE seeds $NEW_SEEDS on GPUs 1-3 (v2WM, train-split caches)"
  g=1; for s in $NEW_SEEDS; do
    lip_train "$g" "$s" "$V2WM" "$FS5" "$FS1" "$TD" \
      "/workspace/actors/lip4_dsp_pre_s${s}.pt" pre & g=$((g+1))
  done; wait
  log "B1: PRE wave done"
  log "B2: POST-full seeds $NEW_SEEDS on GPUs 1-3 (WM2, rep_full caches)"
  g=1; for s in $NEW_SEEDS; do
    lip_train "$g" "$s" "$WM2" "$RF5" "$RF1" "$RTD" \
      "/workspace/actors/lip4_rep_full_s${s}.pt" postfull & g=$((g+1))
  done; wait
  log "B2: POST-full wave done"
  touch "$D/OVR_LANE_B_DONE"
}
rm -f "$D/OVR_LANE_B_DONE"
lane_b &
LANE_B_PID=$!

# ================================================================ lane A (fg)
# --------------------------------------------- A1 relabel gate (CPU, no GPU)
LAB0=$D/onpolicy_full_a0_lab.lance
if [ ! -d "$LAB0" ]; then
  log "A1: relabelling on-policy lances (gates A/B/C inside)"
  python3 /workspace/relabel_onpolicy.py \
    --expert "$EXPERT" \
    --onpolicy $D/onpolicy_full_a0.lance $D/onpolicy_full_a1.lance $D/onpolicy_full_a2.lance \
    --arm1-logs "$L/dsp_col_a{a}_c{c}.log" \
    --ep-range "0:${EPHI}" --goal-offset 25 --num-eval 50 --seed-base 1000 \
    > "$L/ovr_relabel.log" 2>&1
  grep -q "RELABEL_DONE" "$L/ovr_relabel.log" || die "relabel gate FAILED -- see $L/ovr_relabel.log; aborting the whole chain"
fi
log "A1: labels ready ($(grep -h '\[total\]' "$L/ovr_relabel.log" 2>/dev/null | tail -1))"

# --------------------------------------------------- A2 mix at failure 0.20
if [ ! -f "$MIX/.done" ]; then
  log "A2: 50/50 mix at --failure-frac 0.20 (expect K_fail=22, K_succ~13)"
  python3 /workspace/build_dyna_mix.py --expert "$EXPERT" \
    --onpolicy $D/onpolicy_full_a0_lab.lance $D/onpolicy_full_a1_lab.lance $D/onpolicy_full_a2_lab.lance \
    --out "$MIX" --onpolicy-frac 0.5 --failure-frac 0.20 --expert-ep-hi "$EPHI" \
    > "$L/ovr_mix_f20.log" 2>&1 || die "mix failed -- see $L/ovr_mix_f20.log"
  grep -q BUILD_MIX_DONE "$L/ovr_mix_f20.log" || die "mix incomplete"
  grep -q "\[split\] expert restricted" "$L/ovr_mix_f20.log" || die "expert slice NOT restricted"
  grep -q "dup K_fail=" "$L/ovr_mix_f20.log" || die "outcome-split duplication NOT applied (uniform-K fallback?)"
  touch "$MIX/.done"
fi
log "A2: $(grep -h 'dup K_fail=' "$L/ovr_mix_f20.log" | tail -1)"
log "A2: $(grep -h 'achieved:' "$L/ovr_mix_f20.log" | tail -1)"

# ------------------------------------------------------------ A3 WM fine-tune
if [ ! -f "$WMF/weights_epoch_1.pt" ]; then
  log "A3: fine-tune v2WM -> dyna_f20_5050 (lr 1e-5, 2 epochs, epoch 1 pre-registered; ~3.5-4.5h)"
  INIT_WEIGHTS=$V2W CUDA_VISIBLE_DEVICES=0 timeout 86400 python3 \
    scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
    "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
    "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
    trainer.devices=1 output_model_name=dyna_f20_5050 subdir=dyna_f20_5050 \
    +action_stats_pin=expert wandb.enabled=false > "$L/ovr_ft_f20.log" 2>&1 || die "fine-tune failed"
  CK=/workspace/swm_home/checkpoints/dyna_f20_5050
  mkdir -p "$WMF"
  cp "$CK/weights_epoch_1.pt" "$WMF/" || die "no weights_epoch_1.pt in $CK"
  # ARCH config, not the checkpoint's training config.json (ops gotcha §6):
  # fine-tuning does not change architecture; v2WM/config.json is the loader's.
  cp "$V2WM/config.json" "$WMF/config.json" || die "no arch config"
fi
grep -q "load_state_dict" "$WMF/config.json" || die "config.json in $WMF is the TRAINING config, not the arch config"
log "A3: WM_f20 ready"

# ------------------------------------------------------- A4 caches, A5 TD
if [ ! -f "$F20F1" ]; then
  [ -f "$F20F1F" ] || { log "A4: caching fs1 under WM_f20";
    CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$TRM/cache_latents.py" \
      --wm "$WMF" --dataset "$EXPERT" --out "$F20F1F" --state-key privileged_block_0_pos \
      > "$L/ovr_cache_f20_fs1.log" 2>&1 || die "cache failed"; }
  python3 /workspace/filter_cache_eprange.py "$F20F1F" "$F20F1" --lo 0 --hi "$EPHI" \
    > "$L/ovr_filter_f20.log" 2>&1 || die "cache filter failed"
  grep -q FILTER_CACHE_DONE "$L/ovr_filter_f20.log" || die "cache filter incomplete"
fi
[ -f "$F20F5" ] || python3 "$TRM/subsample_cache.py" --in "$F20F1" --out "$F20F5" --frameskip 5 \
  > "$L/ovr_cache_f20_fs5.log" 2>&1 || die "fs5 subsample failed"
if [ ! -f "$F20TD" ]; then
  log "A5: TD teacher on WM_f20 train-split cache"
  CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_metric.py" \
    --cache "$F20F1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
    --steps 6000 --seed 0 --out "$F20TD" > "$L/ovr_td_f20.log" 2>&1 || die "TD failed"
fi
log "A4/A5: caches + TD ready"

# ---------------------------------------------- join: wait for lane B's GPUs
log "join: waiting for lane B (PID $LANE_B_PID) before taking GPUs 1-2"
while [ ! -f "$D/OVR_LANE_B_DONE" ]; do
  kill -0 "$LANE_B_PID" 2>/dev/null || { log "WARNING: lane B exited without marker -- its actors may be missing; continuing"; break; }
  sleep 120
done

# --------------------------------------------------------- A6 LIP f20 wave
log "A6: LIP f20 seeds $F20_SEEDS on GPUs 0-2"
g=0; for s in $F20_SEEDS; do
  lip_train "$g" "$s" "$WMF" "$F20F5" "$F20F1" "$F20TD" \
    "/workspace/actors/lip4_f20_s${s}.pt" f20 & g=$((g+1))
done; wait
log "A6: f20 actors done"

# ================================================================ lane C evals
run_eval(){ # name wm actor draw
  local nm=$1 wm=$2 actor=$3 d=$4
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  [ -f "$actor" ] || { log "  $nm: actor missing, recording FAIL"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$wm" solver=lip "solver.actor_path=$actor" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" \
    || { log "  $nm: ep_range NOT APPLIED -- refusing to record"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
}
log "C: eval queue (strictly sequential, egl, held-out $EVAL_RANGE)"
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" >/dev/null; do sleep 60; done
for s in $NEW_SEEDS; do for d in $DRAWS; do
  run_eval "dsp_pre_s${s}_e${d}" "$V2WM" "/workspace/actors/lip4_dsp_pre_s${s}.pt" "$d"
done; done
for s in $NEW_SEEDS; do for d in $DRAWS; do
  run_eval "dsp_postfull_s${s}_e${d}" "$WM2" "/workspace/actors/lip4_rep_full_s${s}.pt" "$d"
done; done
for s in $F20_SEEDS; do for d in $DRAWS; do
  run_eval "dsp_f20_s${s}_e${d}" "$WMF" "/workspace/actors/lip4_f20_s${s}.pt" "$d"
done; done

# ================================================================ final card
seed_mean(){ m3 "$(sc ${1}_s${2}_e42)" "$(sc ${1}_s${2}_e43)" "$(sc ${1}_s${2}_e44)"; }
log ""
log "=== OVERNIGHT CARD (held-out $EVAL_RANGE, amax $AMAX, egl) ==="
log "  arm            s0     s1     s2     s3     s4     s5"
for a in dsp_pre dsp_postfull dsp_f20; do
  row="  $(printf '%-12s' "$a")"
  for s in 0 1 2 3 4 5; do row="$row  $(printf '%5s' "$(seed_mean "$a" "$s")")"; done
  log "$row"
done
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln:
        continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"):
        rows[k] = float(v)
def seed_mean(arm, s):
    vs = [rows.get(f"{arm}_s{s}_e{d}") for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else sum(vs) / 3
def arm_seeds(arm, seeds):
    ms = [seed_mean(arm, s) for s in seeds]
    return None if any(m is None for m in ms) else ms
def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    return m, sd, t
pre6, full6 = arm_seeds("dsp_pre", range(6)), arm_seeds("dsp_postfull", range(6))
if pre6 and full6:
    m, sd, t = paired(full6, pre6)
    print(f"  Dyna full-vs-pre, 6 seeds: delta {m:+.2f} sd {sd:.2f} t {t:.2f} (df=5)")
f20 = arm_seeds("dsp_f20", range(3))
full3 = arm_seeds("dsp_postfull", range(3))
if f20 and full3:
    m, sd, t = paired(f20, full3)
    print(f"  f20-vs-full (phi 0.20 vs 0.08), seeds 0-2: delta {m:+.2f} sd {sd:.2f} t {t:.2f} (df=2)")
    print("  reading: f20 > full => more failure data helps; run phi=0.40 next."
          " f20 < full => 0.08 already past optimum; thin's deficit was duplication.")
PY
log "OVERNIGHT_DONE"
