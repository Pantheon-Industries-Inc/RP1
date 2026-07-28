#!/bin/bash
# DYNA CONTROL, ARM 2 — full-length on-policy episodes (the K=18 arm).
# Chained after dyna_split_control.sh (arm 1, K=87). Same split, same PRE actors,
# same amax 1.6, same eval seeds. The ONLY difference is how the data was collected.
#
# THE PROBLEM ARM 1 EXPOSED. The WM needs a 25-frame window (span 20 + frameskip
# 5), so `episodes < 25 steps` are structurally unusable and get dropped. With
# terminate_at_goal=True and an ~88%-success actor, the successes are exactly the
# episodes that end early -- so the filter throws away the successes and keeps the
# failures. Measured on arm 1: 407 of 1800 episodes kept (22%), 72% of them
# sitting at the 50-step budget cap. Hitting the pre-registered 50/50 row split
# from 18,500 on-policy rows then needs K=87 duplication, versus K=25 in the r1
# run this recipe was validated on. ~10k distinct on-policy windows seen ~174
# times over 2 epochs is a memorisation risk, and it would make a null result
# uninterpretable: "Dyna adds nothing once amax is fixed" and "407 episodes was
# not enough data" would look identical.
#
# THE FIX. world.terminate_at_goal=False during collection: every episode runs the
# full 50-step budget, so all 1800 are >= 25 steps and survive the filter.
# ~90,000 rows -> K~18, back in r1's regime, at 2.3x the sim steps (~3.3h).
#
# CONTROLLED BY CONSTRUCTION: identical collection seeds to arm 1, so both arms
# roll out from the SAME start states. Arm1 vs Arm2 therefore isolates the
# duplication/termination effect itself, not a data reshuffle.
#
# PRE arm is SHARED -- lip4_dsp_pre_s{0,1,2} are reused and their held-out cells
# (dsp_pre_*) are already in the CSV, so no PRE work is repeated.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_split
SUM=$R/summary_dynasplit.csv; DRV=$L/driver_dyna_arm2.log
V2W=/workspace/models/v2WM/weights_epoch_22.pt
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; COLLECT_RANGE="0:${EPHI}"
AMAX=1.6; SEEDS="0 1 2"; DRAWS="42 43 44"; NCALL=12
mkdir -p "$L" "$R" "$D"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
cd "$CODE"

log "waiting for arm 1 to finish (GPU is single-tenant here)"
while ! grep -q "DYNA_SPLIT_CONTROL_DONE" "$L/driver_dynasplit.log" 2>/dev/null; do
  pgrep -f "dyna_split_control|dsp_chain" >/dev/null || { log "arm1 driver gone without DONE"; break; }
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 120; done
log "=== ARM 2 (full-length episodes, terminate_at_goal=False) start ==="

for s in $SEEDS; do [ -f "/workspace/actors/lip4_dsp_pre_s${s}.pt" ] || die "PRE actor s$s missing"; done

# ------------------------------------------------- P1 collection, full episodes
if [ ! -f "$D/COLLECT_FULL_DONE" ]; then
  log "P1: collection with terminate_at_goal=False, ep_range=$COLLECT_RANGE"
  for i in $(seq 0 $((NCALL-1))); do for a in $SEEDS; do
    rec="$D/onpolicy_full_a${a}.lance"; lg="$L/dsp_colfull_a${a}_c${i}.log"
    grep -q "kept=" "$lg" 2>/dev/null && continue
    SWM_RECORD_PATH=$rec CUDA_VISIBLE_DEVICES=0 timeout 10800 python3 "$P/eval_wm.py" \
      --config-name cube seed=$((1000 + a*100 + i)) eval.dataset_name="$EXPERT" \
      ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
      eval.num_eval=50 "+eval.ep_range=$COLLECT_RANGE" world.terminate_at_goal=False \
      policy=/workspace/models/v2WM solver=lip \
      "solver.actor_path=/workspace/actors/lip4_dsp_pre_s${a}.pt" \
      output.filename="dspcolfull_a${a}_c${i}.txt" > "$lg" 2>&1
    grep -q "ep_range 0:${EPHI}" "$lg" || die "call a$a/c$i did not apply ep_range -- split would be violated"
    log "  collect-full a$a/c$i: $(grep -oE 'kept=[0-9]+' "$lg" | tail -1)"
  done; done
  touch "$D/COLLECT_FULL_DONE"
fi
python3 - "$D" <<'PY' | tee -a "$DRV"
import glob, lance, numpy as np
tot_r = tot_e = 0
for p in sorted(glob.glob(f"{__import__('sys').argv[1]}/onpolicy_full_a*.lance")):
    d = lance.dataset(p); n = d.count_rows()
    e = np.asarray(d.take(list(range(n)), columns=["episode_idx"]).to_pydict()["episode_idx"]).reshape(-1)
    u, c = np.unique(e, return_counts=True); tot_r += n; tot_e += len(u)
    print(f"  {p.split('/')[-1]}: rows={n} episodes={len(u)} meanlen={c.mean():.1f}")
print(f"  ARM2 TOTAL rows={tot_r} episodes={tot_e}   (arm1: 18500 rows / 407 eps)")
PY

# --------------------------------------------------------------- P2 mix (K~18)
MIX=$D/mix_full_5050.lance
if [ ! -f "$MIX/.done" ]; then
  log "P2: 50/50 mix, expert restricted to episode_idx < $EPHI"
  python3 /workspace/build_dyna_mix.py --expert "$EXPERT" \
    --onpolicy $D/onpolicy_full_a0.lance $D/onpolicy_full_a1.lance $D/onpolicy_full_a2.lance \
    --out "$MIX" --onpolicy-frac 0.5 --expert-ep-hi "$EPHI" > "$L/dsp_mix_full.log" 2>&1 || die "mix failed"
  grep -q BUILD_MIX_DONE "$L/dsp_mix_full.log" || die "mix incomplete"
  grep -q "\[split\] expert restricted" "$L/dsp_mix_full.log" || die "expert slice NOT restricted"
  touch "$MIX/.done"
fi
log "P2: $(grep -h 'dup K=' "$L/dsp_mix_full.log" | tail -1)"

# ------------------------------------------------------------ P3 WM fine-tune
WM2=/workspace/models/dyna_full_5050
if [ ! -f "$WM2/weights_epoch_1.pt" ]; then
  log "P3: fine-tune v2WM -> WM2 (lr 1e-5, 2 epochs, epoch 1 pre-registered)"
  INIT_WEIGHTS=$V2W CUDA_VISIBLE_DEVICES=0 timeout 86400 python3 \
    scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
    "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
    "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
    trainer.devices=1 output_model_name=dyna_full_5050 subdir=dyna_full_5050 \
    +action_stats_pin=expert wandb.enabled=false > "$L/dsp_ft_full.log" 2>&1 || die "fine-tune failed"
  CK=/workspace/swm_home/checkpoints/dyna_full_5050
  mkdir -p "$WM2"
  cp "$CK/weights_epoch_1.pt" "$WM2/" || die "no weights_epoch_1.pt in $CK"
  # ARCH config, not the checkpoint's config.json: lewm_expert.py writes the
  # TRAINING config there (output_model_name, num_workers, ...) and the loader
  # needs the architecture one (_target_, encoder, predictor, ...). Fine-tuning
  # does not change architecture, and v2WM/config.json is byte-identical to the
  # r1 run's packaged config (sha fd35a47c). Copying the wrong one fails with
  # "ConfigAttributeError: Missing key load_state_dict" at cache_latents.
  cp /workspace/models/v2WM/config.json "$WM2/config.json" || die "no arch config"
fi
log "P3: WM2 ready"

# ------------------------------------------- P4 downstream: cache -> TD -> LIPx3
F1F=/workspace/caches/dsp_wm2_full_fs1.pt
F1=/workspace/caches/dsp_wm2_tr${EPHI}_fs1.pt
F5=/workspace/caches/dsp_wm2_tr${EPHI}_fs5.pt
if [ ! -f "$F1" ]; then
  [ -f "$F1F" ] || CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$TRM/cache_latents.py" \
    --wm "$WM2" --dataset "$EXPERT" --out "$F1F" --state-key privileged_block_0_pos \
    > "$L/dsp_cache_full_fs1.log" 2>&1 || die "WM2 cache failed"
  python3 /workspace/filter_cache_eprange.py "$F1F" "$F1" --lo 0 --hi "$EPHI" \
    > "$L/dsp_filter_wm2.log" 2>&1 || die "WM2 filter failed"
fi
[ -f "$F5" ] || python3 "$TRM/subsample_cache.py" --in "$F1" --out "$F5" --frameskip 5 \
  > "$L/dsp_cache_full_fs5.log" 2>&1 || die "WM2 fs5 failed"
TD2=/workspace/metrics/dsp_wm2_tr${EPHI}_TD.pt
[ -f "$TD2" ] || CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_metric.py" \
  --cache "$F1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
  --steps 6000 --seed 0 --out "$TD2" > "$L/dsp_td_full.log" 2>&1 || die "WM2 TD failed"
log "P4: caches + TD ready; training POST-full actors"
g=0; for s in $SEEDS; do
  out=/workspace/actors/lip4_dsp_postfull_s${s}.pt
  if [ ! -f "$out" ]; then
    CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
      --cache "$F5" --cache-td "$F1" --h5 "$AH5" --wm "$WM2" --init-value "$TD2" \
      --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$s" \
      --out "$out" --out-value "${out%.pt}_value.pt" \
      > "$L/dsp_lip_postfull_s${s}.log" 2>&1 &
  fi
  g=$((g+1))
done; wait
log "P4: POST-full actors done"

# --------------------------------------------------- P5 evals on the eval split
log "P5: POST-full evals on held-out $EVAL_RANGE"
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" >/dev/null; do sleep 60; done
for s in $SEEDS; do for d in $DRAWS; do
  nm="dsp_postfull_s${s}_e${d}"
  c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; continue; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$WM2" solver=lip "solver.actor_path=/workspace/actors/lip4_dsp_postfull_s${s}.pt" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" || { log "  $nm: ep_range NOT applied"; echo "${nm},FAIL" >> "$SUM"; continue; }
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
done; done

# ------------------------------------------------------------------- P6 the card
arm(){ m3 "$(m3 "$(sc ${1}_s0_e42)" "$(sc ${1}_s0_e43)" "$(sc ${1}_s0_e44)")" \
          "$(m3 "$(sc ${1}_s1_e42)" "$(sc ${1}_s1_e43)" "$(sc ${1}_s1_e44)")" \
          "$(m3 "$(sc ${1}_s2_e42)" "$(sc ${1}_s2_e43)" "$(sc ${1}_s2_e44)")"; }
log ""
log "=== DYNA, EPISODE-DISJOINT, BOTH DATA REGIMES (eval $EVAL_RANGE, amax $AMAX) ==="
log "  arm                       s0     s1     s2    3-seed"
for a in dsp_pre dsp_post dsp_postfull; do
  log "  $(printf '%-22s' "$a")  $(m3 "$(sc ${a}_s0_e42)" "$(sc ${a}_s0_e43)" "$(sc ${a}_s0_e44)")   $(m3 "$(sc ${a}_s1_e42)" "$(sc ${a}_s1_e43)" "$(sc ${a}_s1_e44)")   $(m3 "$(sc ${a}_s2_e42)" "$(sc ${a}_s2_e43)" "$(sc ${a}_s2_e44)")    $(arm "$a")"
done
PRE=$(arm dsp_pre); P1=$(arm dsp_post); P2=$(arm dsp_postfull)
log ""
log "  pre                 $PRE"
log "  post thin (K=87)    $P1   delta $(awk -v a="$PRE" -v b="$P1" 'BEGIN{if(a=="NA"||b=="NA")print "NA";else printf "%+.1f",b-a}')"
log "  post full (K~18)    $P2   delta $(awk -v a="$PRE" -v b="$P2" 'BEGIN{if(a=="NA"||b=="NA")print "NA";else printf "%+.1f",b-a}')"
log "  thin-vs-full        $(awk -v a="$P1" -v b="$P2" 'BEGIN{if(a=="NA"||b=="NA")print "NA";else printf "%+.1f",b-a}')  <- the duplication effect, same start states"
log "  n=150/arm-seed => SE ~2.9 pts; deltas under ~4 pts are not distinguishable."
log "  If BOTH deltas are ~0, 'Dyna adds nothing once amax 1.6 fixes the"
log "  catastrophic-seed mode' is a clean finding, not a data artifact."
log "DYNA_ARM2_DONE"
