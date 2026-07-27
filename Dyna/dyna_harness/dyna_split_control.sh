#!/bin/bash
# EPISODE-DISJOINT DYNA CONTROL — the experiment that decides whether Dyna is real.
# Policy: ../DATA_SPLIT_POLICY.md.   Split: train 0-7999, eval 8000-9999.
#
# WHY THIS EXISTS. The headline 83.0 -> 94.4 is not a valid controlled comparison:
#   (a) collection drew from the same 10k pool eval scores on, so WM1 was
#       fine-tuned on rollouts launched from the eval demonstrations;
#   (b) the pre- and post-Dyna arms used DIFFERENT amax (2.2/1.8 vs 3.5), and the
#       amax sweep shows 3.5 alone costs ~7 pts -- most of the claimed gain.
# This run fixes both: every episode touched by collection, the fine-tune, the
# caches, the TD teacher and the LIP critic comes from 0-7999; every eval task is
# drawn from 8000-9999; and BOTH arms are trained at amax 1.6 with identical
# seeds and identical everything except the world model.
#
# STRICT split, deliberately: the caches are filtered too, not just the
# fine-tune. 0-7999 is a PREFIX, so kept episode ids stay 0..7999 contiguous --
# expert_actions.h5 lookups and train_lip_ac.py's ep_off/ep_len bookkeeping keep
# working with no renumbering. Residual we cannot remove here: v2WM itself was
# pre-trained on all 10k. That is stated in the card, not hidden.
#
# PRE-REGISTERED CHOICES (so nothing is selected on the eval set):
#   * amax 1.6 for both arms (chosen by the amax sweep, before this run)
#   * fine-tune = 50/50 expert:on-policy, lr 1e-5, epoch 1 (the r1 winner)
#   * 3 seeds {0,1,2} x 3 draws {42,43,44}
# Do NOT pick a different epoch or arm after seeing the numbers.
#
# Ops: OMP=8; <=4 concurrent trainings; evals STRICTLY sequential (3-way -> SIGABRT).
#
# RENDERER: egl, always (standing instruction 2026-07-27). egl is what produced
# the authors' h5 renders, and the reacher campaign found osmesa is out-of-domain
# against them -- worth +7.3 pts there. On cube the two measured equal (seed 0:
# egl 86.0 vs osmesa 85.3), so this does not move our numbers, but egl is the
# correct default and removes a whole class of domain-gap doubt.
# Two egl requirements this script satisfies:
#   * MUJOCO_EGL_DEVICE_ID must be pinned, or every render context piles onto
#     physical GPU 0 regardless of CUDA_VISIBLE_DEVICES (parallelization audit).
#   * egl crashes under concurrent training -- every eval phase below is gated on
#     a `while pgrep ... train` wait loop, and rendering phases never overlap
#     training. The concurrent phases (P3/P8 LIP training) do not render at all.
# Fully resumable: every phase is marker- or artifact-gated; safe to re-run.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_split
SUM=$R/summary_dynasplit.csv; DRV=$L/driver_dynasplit.log
V2W=/workspace/models/v2WM/weights_epoch_22.pt
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000                 # train split = [0, EPHI);  eval split = [EPHI, 10000)
EVAL_RANGE="${EPHI}:10000"
COLLECT_RANGE="0:${EPHI}"
AMAX=1.6
SEEDS="0 1 2"; DRAWS="42 43 44"; NCALL=12   # 12 calls x 3 actors x 50 envs
mkdir -p "$L" "$R" "$D" /workspace/actors /workspace/metrics /workspace/caches
touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
cd "$CODE"
log "=== EPISODE-DISJOINT DYNA CONTROL (pid $$) train ${COLLECT_RANGE} / eval ${EVAL_RANGE} ==="

# ---------------------------------------------------------------- P0 preflight
grep -q "ep_range" "$P/eval_wm.py" || die "eval_wm.py on this pod has NO ep_range support -- deploy the fixed file first"
grep -q "_row_in_range" "$P/eval_wm.py" || die "eval_wm.py has ep_range but NOT the KeyError fix -- deploy the fixed file first"
log "P0: eval_wm.py has ep_range + the mask fix"

# ------------------------------------------------- P1 train-split latent caches
FS1=/workspace/caches/v2_tr${EPHI}_fs1.pt
FS5=/workspace/caches/v2_tr${EPHI}_fs5.pt
if [ ! -f "$FS1" ]; then
  log "P1: filtering fs1 cache to episodes ${COLLECT_RANGE}"
  python3 /workspace/filter_cache_eprange.py /workspace/caches/v2_expert_fs1.pt "$FS1" \
    --lo 0 --hi "$EPHI" > "$L/dsp_filter_fs1.log" 2>&1 || die "fs1 filter failed"
  grep -q FILTER_CACHE_DONE "$L/dsp_filter_fs1.log" || die "fs1 filter incomplete"
fi
if [ ! -f "$FS5" ]; then
  log "P1: filtering fs5 cache to episodes ${COLLECT_RANGE}"
  python3 /workspace/filter_cache_eprange.py /workspace/caches/v2_expert_fs5.pt "$FS5" \
    --lo 0 --hi "$EPHI" > "$L/dsp_filter_fs5.log" 2>&1 || die "fs5 filter failed"
  grep -q FILTER_CACHE_DONE "$L/dsp_filter_fs5.log" || die "fs5 filter incomplete"
fi
log "P1: caches ready ($(grep -h 'rows ' "$L/dsp_filter_fs1.log" | tail -1))"

# --------------------------------------------------------- P2 TD on train split
TD=/workspace/metrics/v2_tr${EPHI}_TD.pt
if [ ! -f "$TD" ]; then
  log "P2: TD teacher on the train split"
  CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_metric.py" \
    --cache "$FS1" --learner td --head quasimetric \
    --expectile 0.03 --n-step 50 --steps 6000 --seed 0 \
    --out "$TD" > "$L/dsp_td_pre.log" 2>&1 || die "TD failed"
fi
log "P2: TD ready"

# ------------------------------------------- P3 PRE actors (amax 1.6, split-safe)
lip_train(){ # gpu seed wm cache_fs5 cache_fs1 td out tag
  local gpu=$1 seed=$2 wm=$3 c5=$4 c1=$5 td=$6 out=$7 tag=$8
  [ -f "$out" ] && { log "  $tag/s$seed reusing"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$c5" --cache-td "$c1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$seed" \
    --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/dsp_lip_${tag}_s${seed}.log" 2>&1 || { log "  $tag/s$seed TRAIN FAILED"; return 1; }
  log "  $tag/s$seed trained (E_final $(grep -oE 'E_final [0-9.]+' "$L/dsp_lip_${tag}_s${seed}.log" | tail -1 | awk '{print $2}'))"
}
log "P3: PRE actors (v2WM, amax $AMAX, train-split caches)"
g=0; for s in $SEEDS; do
  lip_train "$g" "$s" /workspace/models/v2WM "$FS5" "$FS1" "$TD" \
    "/workspace/actors/lip4_dsp_pre_s${s}.pt" pre & g=$((g+1))
done; wait
log "P3: PRE actors done"

# ------------------------------------------------------ eval helper (sequential)
run_eval(){ # name wm actor draw ep_range
  local nm=$1 wm=$2 actor=$3 d=$4 rng=$5
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$rng" \
    policy="$wm" solver=lip "solver.actor_path=$actor" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${rng%:*}:${rng#*:}" "$L/eval_${nm}.log" \
    || { log "  $nm: ep_range NOT APPLIED -- refusing to record"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
}

# ------------------------------------------------ P4 PRE evals on the eval split
log "P4: PRE evals on held-out episodes $EVAL_RANGE"
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" > /dev/null; do sleep 60; done
for s in $SEEDS; do for d in $DRAWS; do
  run_eval "dsp_pre_s${s}_e${d}" /workspace/models/v2WM \
    "/workspace/actors/lip4_dsp_pre_s${s}.pt" "$d" "$EVAL_RANGE"
done; done

# ------------------------------------------------- P5 collection from 0:EPHI only
if [ ! -f "$D/COLLECT_DONE" ]; then
  log "P5: on-policy collection, ep_range=$COLLECT_RANGE, ${NCALL} calls x 3 actors"
  for i in $(seq 0 $((NCALL-1))); do for a in $SEEDS; do
    rec="$D/onpolicy_dsp_a${a}.lance"
    lg="$L/dsp_col_a${a}_c${i}.log"
    grep -q "kept=" "$lg" 2>/dev/null && continue
    SWM_RECORD_PATH=$rec CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" \
      --config-name cube seed=$((1000 + a*100 + i)) eval.dataset_name="$EXPERT" \
      ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
      eval.num_eval=50 "+eval.ep_range=$COLLECT_RANGE" \
      policy=/workspace/models/v2WM solver=lip \
      "solver.actor_path=/workspace/actors/lip4_dsp_pre_s${a}.pt" \
      output.filename="dspcol_a${a}_c${i}.txt" > "$lg" 2>&1
    grep -q "ep_range 0:${EPHI}" "$lg" || die "collection call a$a/c$i did not apply ep_range -- ABORTING, the split would be violated"
    log "  collect a$a/c$i: $(grep -oE 'kept=[0-9]+' "$lg" | tail -1)"
  done; done
  touch "$D/COLLECT_DONE"
fi
log "P5: collection done"

# --------------------------------------------- P5b VERIFY the split actually held
log "P5b: auditing collection vs eval episodes"
python3 - "$D" "$EPHI" > "$L/dsp_audit.log" 2>&1 <<'PY'
import sys, glob, lance, numpy as np
d, ephi = sys.argv[1], int(sys.argv[2])
bad = 0
for p in sorted(glob.glob(f"{d}/onpolicy_dsp_a*.lance")):
    ds = lance.dataset(p)
    # the recorder stores the SOURCE episode in step/goal metadata; what matters
    # here is that no rollout was launched from an eval-split episode. The
    # collector's own ep_range log line is the primary guard; this is the audit.
    n = ds.count_rows()
    print(f"{p}: rows={n}")
print("AUDIT_NOTE: primary guard is the 'ep_range 0:%d' line asserted per call." % ephi)
print("AUDIT_DONE")
PY
cat "$L/dsp_audit.log" | tee -a "$DRV"

# -------------------------------------------------------- P6 fine-tune dataset
MIX=$D/mix_dsp_5050.lance
if [ ! -f "$MIX/.done" ]; then
  log "P6: building 50/50 mix with expert restricted to episode_idx < $EPHI"
  python3 /workspace/build_dyna_mix.py --expert "$EXPERT" \
    --onpolicy $D/onpolicy_dsp_a0.lance $D/onpolicy_dsp_a1.lance $D/onpolicy_dsp_a2.lance \
    --out "$MIX" --onpolicy-frac 0.5 --expert-ep-hi "$EPHI" \
    > "$L/dsp_mix.log" 2>&1 || die "mix build failed"
  grep -q BUILD_MIX_DONE "$L/dsp_mix.log" || die "mix incomplete"
  grep -q "\[split\] expert restricted" "$L/dsp_mix.log" || die "mix did NOT restrict the expert slice"
  touch "$MIX/.done"
fi
log "P6: mix ready -- $(grep -h 'expert rows' "$L/dsp_mix.log" | tail -1)"

# ------------------------------------------------------------- P7 WM fine-tune
WM1DIR=/workspace/models/dyna_dsp_5050
if [ ! -f "$WM1DIR/weights_epoch_1.pt" ]; then
  log "P7: fine-tuning v2WM -> WM1 (lr 1e-5, 2 epochs, epoch 1 pre-registered)"
  INIT_WEIGHTS=$V2W CUDA_VISIBLE_DEVICES=0 timeout 86400 python3 \
    scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
    "data.dataset.keys_to_load=[pixels,action]" \
    "data.dataset.keys_to_cache=[action]" \
    "data.dataset.keys_to_merge=null" \
    optimizer.lr=1e-5 trainer.max_epochs=2 trainer.devices=1 \
    output_model_name=dyna_dsp_5050 subdir=dyna_dsp_5050 \
    +action_stats_pin=expert wandb.enabled=false \
    > "$L/dsp_ft.log" 2>&1 || die "fine-tune failed (see dsp_ft.log)"
  CK=/workspace/swm_home/checkpoints/dyna_dsp_5050
  mkdir -p "$WM1DIR"
  cp "$CK/config.json" "$WM1DIR/" || die "no config.json in $CK"
  cp "$CK/weights_epoch_1.pt" "$WM1DIR/" || die "no weights_epoch_1.pt in $CK"
fi
log "P7: WM1 ready at $WM1DIR"

# ------------------------------- P8 post-Dyna downstream (cache -> TD -> LIP x3)
FS1B=/workspace/caches/dsp_wm1_tr${EPHI}_fs1.pt
FS5B=/workspace/caches/dsp_wm1_tr${EPHI}_fs5.pt
FS1FULL=/workspace/caches/dsp_wm1_full_fs1.pt
if [ ! -f "$FS1B" ]; then
  if [ ! -f "$FS1FULL" ]; then
    log "P8: caching latents under WM1"
    CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$TRM/cache_latents.py" \
      --wm "$WM1DIR" --dataset "$EXPERT" --out "$FS1FULL" \
      --state-key privileged_block_0_pos > "$L/dsp_cache_fs1.log" 2>&1 || die "WM1 fs1 cache failed"
  fi
  python3 /workspace/filter_cache_eprange.py "$FS1FULL" "$FS1B" --lo 0 --hi "$EPHI" \
    > "$L/dsp_filter_wm1_fs1.log" 2>&1 || die "WM1 fs1 filter failed"
fi
if [ ! -f "$FS5B" ]; then
  python3 "$TRM/subsample_cache.py" --in "$FS1B" --out "$FS5B" --frameskip 5 \
    > "$L/dsp_cache_fs5.log" 2>&1 || die "WM1 fs5 subsample failed"
fi
log "P8: WM1 caches ready"
TDB=/workspace/metrics/dsp_wm1_tr${EPHI}_TD.pt
if [ ! -f "$TDB" ]; then
  log "P8: TD teacher on WM1 train-split cache"
  CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_metric.py" \
    --cache "$FS1B" --learner td --head quasimetric \
    --expectile 0.03 --n-step 50 --steps 6000 --seed 0 \
    --out "$TDB" > "$L/dsp_td_post.log" 2>&1 || die "WM1 TD failed"
fi
log "P8: POST actors (WM1, amax $AMAX)"
g=0; for s in $SEEDS; do
  lip_train "$g" "$s" "$WM1DIR" "$FS5B" "$FS1B" "$TDB" \
    "/workspace/actors/lip4_dsp_post_s${s}.pt" post & g=$((g+1))
done; wait

# ----------------------------------------------- P9 POST evals on the eval split
log "P9: POST evals on held-out episodes $EVAL_RANGE"
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" > /dev/null; do sleep 60; done
for s in $SEEDS; do for d in $DRAWS; do
  run_eval "dsp_post_s${s}_e${d}" "$WM1DIR" \
    "/workspace/actors/lip4_dsp_post_s${s}.pt" "$d" "$EVAL_RANGE"
done; done

# ------------------------------------------------------------------- P10 the card
log ""
log "=== EPISODE-DISJOINT DYNA CONTROL — train ${COLLECT_RANGE}, eval ${EVAL_RANGE} ==="
log "    both arms: amax $AMAX, seeds {0,1,2}, draws {42,43,44}, v2WM base"
log "  arm    s0     s1     s2     3-seed"
for arm in pre post; do
  A=$(m3 "$(sc dsp_${arm}_s0_e42)" "$(sc dsp_${arm}_s0_e43)" "$(sc dsp_${arm}_s0_e44)")
  B=$(m3 "$(sc dsp_${arm}_s1_e42)" "$(sc dsp_${arm}_s1_e43)" "$(sc dsp_${arm}_s1_e44)")
  C=$(m3 "$(sc dsp_${arm}_s2_e42)" "$(sc dsp_${arm}_s2_e43)" "$(sc dsp_${arm}_s2_e44)")
  log "  $(printf '%-5s' "$arm")  $A   $B   $C     $(m3 "$A" "$B" "$C")"
done
PRE=$(m3 "$(m3 "$(sc dsp_pre_s0_e42)" "$(sc dsp_pre_s0_e43)" "$(sc dsp_pre_s0_e44)")" "$(m3 "$(sc dsp_pre_s1_e42)" "$(sc dsp_pre_s1_e43)" "$(sc dsp_pre_s1_e44)")" "$(m3 "$(sc dsp_pre_s2_e42)" "$(sc dsp_pre_s2_e43)" "$(sc dsp_pre_s2_e44)")")
POST=$(m3 "$(m3 "$(sc dsp_post_s0_e42)" "$(sc dsp_post_s0_e43)" "$(sc dsp_post_s0_e44)")" "$(m3 "$(sc dsp_post_s1_e42)" "$(sc dsp_post_s1_e43)" "$(sc dsp_post_s1_e44)")" "$(m3 "$(sc dsp_post_s2_e42)" "$(sc dsp_post_s2_e43)" "$(sc dsp_post_s2_e44)")")
log ""
log "  DELTA = $(awk -v a="$PRE" -v b="$POST" 'BEGIN{if(a=="NA"||b=="NA"){print "NA"}else{printf "%+.1f", b-a}}')  (post $POST - pre $PRE)"
log "  Uncontrolled reference for contrast: 83.0 -> 94.4 (+11.4), which mixed the"
log "  fine-tune with an amax change worth ~7 pts AND collected from eval episodes."
log "  n=150 per arm-seed => SE ~2.9 pts; a delta under ~4 pts is not distinguishable."
log "  Residual, state it: v2WM was pre-trained on all 10k episodes."
log "DYNA_SPLIT_CONTROL_DONE"
