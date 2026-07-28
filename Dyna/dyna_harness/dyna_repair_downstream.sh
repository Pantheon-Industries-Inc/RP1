#!/bin/bash
# REPAIR + FINISH both Dyna arms after the arch-vs-training config.json bug.
#
# THE BUG. lewm_expert.py writes a config.json into its checkpoint dir, but that
# is the TRAINING config (output_model_name, num_workers, train_split, data, ...).
# The model loader wants the ARCHITECTURE config (_target_, encoder, predictor,
# projector, action_encoder). Both arms' packaging step copied the wrong one, so
# cache_latents.py died instantly with:
#     omegaconf.errors.ConfigAttributeError: Missing key load_state_dict
# Fine-tuning does not change architecture, so the arch config is just the base
# model's. Verified: v2WM/config.json is byte-identical (sha fd35a47c, 1114 B) to
# dyna_r1_5050b/config.json, i.e. exactly how the r1 run packaged its WM.
# This is the "arch-vs-training config.json" gotcha already recorded in memory.
#
# WHAT THIS DOES. Arm 1 died at P8 with WM1 weights intact; arm 2's fine-tune is
# still running (its driver was retired, but the timeout-wrapped trainer sits in
# its own process group and survives). So:
#   A  now, on GPUs 1-3:  arm1 cache -> filter -> fs5 -> TD -> LIP x3
#      (arm2's fine-tune holds only GPU 0 and is CPU-bound at ~0% GPU duty, so
#       this is genuine free parallelism, not contention)
#   B  when arm2's fine-tune exits: package WM2 with the ARCH config, then its
#      cache -> TD -> LIP x3
#   C  once everything is quiet: 18 evals, strictly sequential, egl, GPU 0
#   D  combined card: pre / post-thin(K=87) / post-full(K~18)
# Resumable: every step is artifact-gated.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results
SUM=$R/summary_dynasplit.csv; DRV=$L/driver_repair.log
ARCH=/workspace/models/v2WM/config.json          # the architecture config
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; AMAX=1.6
SEEDS="0 1 2"; DRAWS="42 43 44"
touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
cd "$CODE"

# package a fine-tuned WM correctly: weights from the checkpoint, ARCH config from v2WM
package(){ # ckpt_dir model_dir
  local ck=$1 md=$2
  [ -f "$ck/weights_epoch_1.pt" ] || die "no weights_epoch_1.pt in $ck"
  mkdir -p "$md"
  cp "$ck/weights_epoch_1.pt" "$md/"
  cp "$ARCH" "$md/config.json"           # <-- the fix
  python3 -c "
import json,sys
j=json.load(open('$md/config.json'))
need={'_target_','encoder','predictor','projector','action_encoder'}
missing=need-set(j)
sys.exit(f'arch config missing {missing}' if missing else 0)" || die "packaged config is not an arch config"
  log "  packaged $md (arch config from v2WM)"
}

# cache -> filter -> fs5 -> TD -> LIP x3 for one arm
downstream(){ # tag model_dir gpu_a gpu_b gpu_c
  local tag=$1 md=$2 ga=$3 gb=$4 gc=$5
  local f1f=/workspace/caches/rep_${tag}_full_fs1.pt
  local f1=/workspace/caches/rep_${tag}_tr${EPHI}_fs1.pt
  local f5=/workspace/caches/rep_${tag}_tr${EPHI}_fs5.pt
  local td=/workspace/metrics/rep_${tag}_TD.pt
  if [ ! -f "$f1" ]; then
    if [ ! -f "$f1f" ]; then
      log "  [$tag] caching latents (gpu $ga)"
      CUDA_VISIBLE_DEVICES=$ga timeout 28800 python3 "$TRM/cache_latents.py" --wm "$md" \
        --dataset "$EXPERT" --out "$f1f" --state-key privileged_block_0_pos \
        > "$L/rep_cache_${tag}.log" 2>&1 || { log "  [$tag] CACHE FAILED"; tail -5 "$L/rep_cache_${tag}.log" | tee -a "$DRV"; return 1; }
    fi
    python3 /workspace/filter_cache_eprange.py "$f1f" "$f1" --lo 0 --hi "$EPHI" \
      > "$L/rep_filter_${tag}.log" 2>&1 || { log "  [$tag] filter failed"; return 1; }
  fi
  [ -f "$f5" ] || python3 "$TRM/subsample_cache.py" --in "$f1" --out "$f5" --frameskip 5 \
      > "$L/rep_fs5_${tag}.log" 2>&1 || { log "  [$tag] fs5 failed"; return 1; }
  if [ ! -f "$td" ]; then
    log "  [$tag] TD teacher (gpu $ga)"
    CUDA_VISIBLE_DEVICES=$ga timeout 28800 python3 "$P/train_metric.py" --cache "$f1" \
      --learner td --head quasimetric --expectile 0.03 --n-step 50 --steps 6000 --seed 0 \
      --out "$td" > "$L/rep_td_${tag}.log" 2>&1 || { log "  [$tag] TD failed"; return 1; }
  fi
  log "  [$tag] LIP x3 (gpus $ga/$gb/$gc)"
  local gpus=("$ga" "$gb" "$gc")
  local i=0
  for s in $SEEDS; do
    local g=${gpus[$i]}
    local out=/workspace/actors/lip4_rep_${tag}_s${s}.pt
    if [ ! -f "$out" ]; then
      CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
        --cache "$f5" --cache-td "$f1" --h5 "$AH5" --wm "$md" --init-value "$td" \
        --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
        --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
        --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$s" \
        --out "$out" --out-value "${out%.pt}_value.pt" \
        > "$L/rep_lip_${tag}_s${s}.log" 2>&1 &
    fi
    i=$((i+1))
  done
  wait
  log "  [$tag] downstream complete ($(ls -1 /workspace/actors/lip4_rep_${tag}_s*.pt 2>/dev/null | wc -l)/3 actors)"
}

# ------------------------------------------------------- A: arm1, on the idle GPUs
log "=== A: arm1 (thin, K=87) downstream on gpus 1-3, alongside arm2's fine-tune ==="
package /workspace/swm_home/checkpoints/dyna_dsp_5050 /workspace/models/dyna_dsp_5050
downstream thin /workspace/models/dyna_dsp_5050 1 2 3 || log "arm1 downstream had failures"

# ------------------------------------------- B: wait for arm2's fine-tune, then it
log "=== B: waiting for arm2's fine-tune to exit ==="
while pgrep -f "lewm_exper[t].*dyna_full_5050|lewm_exper[t].*mix_full" >/dev/null; do sleep 120; done
while pgrep -f "lewm_exper[t]" >/dev/null; do sleep 120; done
log "arm2 fine-tune done"
package /workspace/swm_home/checkpoints/dyna_full_5050 /workspace/models/dyna_full_5050
downstream full /workspace/models/dyna_full_5050 1 2 3 || log "arm2 downstream had failures"

# --------------------------------------------- C: evals, strictly sequential, egl
log "=== C: 18 held-out evals, sequential, egl ==="
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|cache_latent[s]" >/dev/null; do sleep 60; done
ev(){ # cellname model actor draw
  local nm=$1 md=$2 A=$3 d=$4
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  [ -f "$A" ] || { log "  $nm actor missing, skip"; return 0; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 \
    eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" policy="$md" solver=lip \
    "solver.actor_path=$A" output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" || { log "  $nm: ep_range NOT applied"; echo "${nm},FAIL" >> "$SUM"; return 0; }
  grep -q "MUJOCO_GL=egl" "$L/eval_${nm}.log" || { log "  $nm: not egl"; echo "${nm},FAIL" >> "$SUM"; return 0; }
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
}
for s in $SEEDS; do for d in $DRAWS; do
  ev "dsp_post_s${s}_e${d}" /workspace/models/dyna_dsp_5050 /workspace/actors/lip4_rep_thin_s${s}.pt "$d"
done; done
for s in $SEEDS; do for d in $DRAWS; do
  ev "dsp_postfull_s${s}_e${d}" /workspace/models/dyna_full_5050 /workspace/actors/lip4_rep_full_s${s}.pt "$d"
done; done

# ------------------------------------------------------------------- D: the card
arm(){ m3 "$(m3 "$(sc ${1}_s0_e42)" "$(sc ${1}_s0_e43)" "$(sc ${1}_s0_e44)")" \
          "$(m3 "$(sc ${1}_s1_e42)" "$(sc ${1}_s1_e43)" "$(sc ${1}_s1_e44)")" \
          "$(m3 "$(sc ${1}_s2_e42)" "$(sc ${1}_s2_e43)" "$(sc ${1}_s2_e44)")"; }
log ""
log "=== DYNA, EPISODE-DISJOINT (train 0:7999 / eval ${EVAL_RANGE}), amax ${AMAX}, egl ==="
log "  arm                      s0     s1     s2    3-seed"
for a in dsp_pre dsp_post dsp_postfull; do
  log "  $(printf '%-21s' "$a")  $(m3 "$(sc ${a}_s0_e42)" "$(sc ${a}_s0_e43)" "$(sc ${a}_s0_e44)")   $(m3 "$(sc ${a}_s1_e42)" "$(sc ${a}_s1_e43)" "$(sc ${a}_s1_e44)")   $(m3 "$(sc ${a}_s2_e42)" "$(sc ${a}_s2_e43)" "$(sc ${a}_s2_e44)")    $(arm "$a")"
done
PRE=$(arm dsp_pre); T=$(arm dsp_post); F=$(arm dsp_postfull)
d(){ awk -v a="$1" -v b="$2" 'BEGIN{if(a=="NA"||b=="NA")print "NA";else printf "%+.1f",b-a}'; }
log ""
log "  pre                  $PRE"
log "  post thin  (K=87)    $T    delta $(d "$PRE" "$T")"
log "  post full  (K~18)    $F    delta $(d "$PRE" "$F")"
log "  full - thin          $(d "$T" "$F")   <- duplication effect, identical start states"
log "  n=150 per arm-seed => SE ~2.9 pts; deltas under ~4 pts are not distinguishable."
log "  Uncontrolled reference: 83.0 -> 94.4 (+11.4), which mixed the fine-tune with"
log "  an amax change worth ~7 pts AND collected from the eval episodes."
log "DYNA_REPAIR_DONE"
