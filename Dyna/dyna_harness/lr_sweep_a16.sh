#!/bin/bash
# Post-amax tuning at the chosen clip: amax 1.6 fixed, 3 arms x 3 seeds.
#
# WHY THESE ARMS. The amax sweep showed the clip removes the catastrophic-seed
# mode (spread 20.7 -> 4.7) and that the optimum is a flat plateau 1.4-2.2, so
# further clip tuning is exhausted. What is NOT known is whether the clip changed
# the OPTIMIZATION regime. Evidence it might have:
#   * at amax 3.5, --iters 16 gave the best imagined plans ever measured
#     (E_final 1.68 vs 7.82 baseline) and planned WORSE in reality (80.0) --
#     classic action-exploitation: harder optimization bought better fantasies.
#   * at amax 1.6, E_final is only 5.05 / 4.70 / 3.71 -- nowhere near 1.68, i.e.
#     there is optimization headroom left unused.
# If the clip bounds actions tightly enough that exploitation is no longer
# reachable, harder/longer optimization should now be SAFE and possibly helpful.
# That is the interaction this tests. it16 is the informative arm; s12k and alr
# separate "more optimization" from "more careful optimization".
#
#   it16  --iters 16                                (2x inner optimization)
#   s12k  --steps 12000                             (2x training)
#   alr   --actor-lr 1e-4 --actor-lr-final 1e-5      (3x lower actor lr)
# Baseline is the existing amax-1.6 card: 85.3 / 84.7 / 89.3 -> 86.4.
#
# PHASE A first, and it may invalidate everything: the reacher campaign found
# 2026-07-27 that MUJOCO_GL=osmesa renders are out-of-domain vs the authors' h5
# renders, worth +7.3 pts there. Every cube number to date is osmesa. Phase A
# re-evaluates the three existing amax-1.6 actors under egl; if egl wins, the
# arm evals run under egl and the 86.4 baseline must be restated.
# egl needs MUJOCO_EGL_DEVICE_ID pinned (else every render context piles onto
# physical GPU 0) and crashes under concurrent training -- so Phase A runs on an
# idle pod, before any training starts.
#
# Ops rules honoured: OMP=8, <=4 concurrent trainings (pid-quota), evals strictly
# sequential (3-way concurrent -> SIGABRT). Idempotent: every cell is cached in
# the summary CSV and every actor is skipped if its .pt exists.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export TQDM_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/workspace/code/stable-worldmodel/scripts/plan
L=/workspace/logs; SUM=/workspace/results/summary_lrsweep.csv; DRV=$L/driver_lrsweep.log
WM=/workspace/models/v2WM; TD=/workspace/metrics/v2_expert_TD.pt
FS1=/workspace/caches/v2_expert_fs1.pt; FS5=/workspace/caches/v2_expert_fs5.pt
AH5=/workspace/datasets/expert_actions.h5
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AMAX=1.6
mkdir -p /workspace/results /workspace/actors; touch "$SUM"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
cd /workspace/code/stable-worldmodel

# one eval; $1 name  $2 actor  $3 draw  $4 renderer(osmesa|egl)
run_eval(){
  local nm=$1 actor=$2 d=$3 rend=$4
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  local -a env=(MUJOCO_GL=$rend PYOPENGL_PLATFORM=$rend)
  [ "$rend" = egl ] && env+=(MUJOCO_EGL_DEVICE_ID=0)
  env "${env[@]}" CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" \
    --config-name cube seed=$d eval.dataset_name="$EXPERT" ++bf16=true \
    eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
    policy="$WM" solver=lip "solver.actor_path=$actor" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
}

# ---------------------------------------------------------------- PHASE A: egl?
log "### PHASE A: renderer check -- amax 1.6 actors under egl (osmesa ref 86.4) ###"
while pgrep -f "train_lip_a[c]|eval_w[m]" > /dev/null; do sleep 60; done
for s in 0 1 2; do for d in 42 43 44; do
  run_eval "eglchk_s${s}_e${d}" "/workspace/actors/lip4_amsw_a16_s${s}.pt" "$d" egl
done; done
EGL0=$(m3 "$(sc eglchk_s0_e42)" "$(sc eglchk_s0_e43)" "$(sc eglchk_s0_e44)")
EGL1=$(m3 "$(sc eglchk_s1_e42)" "$(sc eglchk_s1_e43)" "$(sc eglchk_s1_e44)")
EGL2=$(m3 "$(sc eglchk_s2_e42)" "$(sc eglchk_s2_e43)" "$(sc eglchk_s2_e44)")
EGLM=$(m3 "$EGL0" "$EGL1" "$EGL2")
log "=== EGL CHECK: $EGL0 / $EGL1 / $EGL2 -> $EGLM   vs osmesa 85.3 / 84.7 / 89.3 -> 86.4 ==="
# egl ALWAYS (standing instruction 2026-07-27): egl is what produced the authors'
# h5 renders, so it is the correct default regardless of which scores higher. The
# check above is kept as a measurement, not as a switch.
REND=egl
if [ "$EGLM" != NA ]; then
  log "egl $EGLM vs osmesa 86.4 -> delta $(awk -v e="$EGLM" 'BEGIN{printf "%+.1f", e-86.4}') pts."
  log "The reacher osmesa out-of-domain bug does NOT transfer to cube; record that."
fi
log "### renderer for arm evals: $REND ###"

# ------------------------------------------------------------- PHASE B: trainings
train_one(){ # gpu arm seed
  local gpu=$1 arm=$2 seed=$3
  local out=/workspace/actors/lip4_lrsw_${arm}_s${seed}.pt
  [ -f "$out" ] && { log "$arm/s$seed already trained, reusing"; return 0; }
  local -a extra=()
  case "$arm" in
    it16) extra=(--iters 16 --steps 6000) ;;
    s12k) extra=(--iters 8  --steps 12000) ;;
    alr)  extra=(--iters 8  --steps 6000 --actor-lr 1e-4 --actor-lr-final 1e-5) ;;
    *) log "unknown arm $arm"; return 1 ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$P/train_lip_ac.py" \
    --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
    --horizon 5 --n-step 50 --amax "$AMAX" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    "${extra[@]}" --arch v4 --seed "$seed" \
    --out "$out" --out-value /workspace/metrics/lip4_lrsw_${arm}_s${seed}_value.pt \
    > "$L/train_lrsw_${arm}_s${seed}.log" 2>&1 \
    || { log "$arm/s$seed TRAIN FAILED (see train_lrsw_${arm}_s${seed}.log)"; return 1; }
  log "$arm/s$seed trained (E_final $(grep -oE 'E_final [0-9.]+' "$L/train_lrsw_${arm}_s${seed}.log" | tail -1 | awk '{print $2}'))"
}

# alr defaults to actor-lr 3e-4 in the baseline, so 'alr' is the only lr arm;
# note that arms differ in wall clock (s12k ~2x), hence the simple 4-slot pool.
JOBS=""
for arm in it16 s12k alr; do for s in 0 1 2; do
  [ -f "/workspace/actors/lip4_lrsw_${arm}_s${s}.pt" ] || JOBS="$JOBS ${arm}:${s}"
done; done
set -- $JOBS
log "### PHASE B: $# trainings needed (4 concurrent max) ###"
while [ $# -gt 0 ]; do
  for gpu in 0 1 2 3; do
    [ $# -eq 0 ] && break
    j=$1; shift
    train_one "$gpu" "${j%%:*}" "${j##*:}" &
  done
  wait
  log "  training batch done"
done

# --------------------------------------------------------------- PHASE C: evals
log "### PHASE C: arm evals (sequential, $REND) ###"
for arm in it16 s12k alr; do for s in 0 1 2; do
  A=/workspace/actors/lip4_lrsw_${arm}_s${s}.pt
  [ -f "$A" ] || { log "  $arm/s$s actor missing, skip"; continue; }
  for d in 42 43 44; do run_eval "lrsw_${arm}_s${s}_e${d}" "$A" "$d" "$REND"; done
done; done

log ""
log "=== LR/OPT SWEEP CARD (amax 1.6, v2WM, h25, draws 42/43/44, $REND) ==="
log "  arm       s0     s1     s2    3-seed"
for arm in it16 s12k alr; do
  A=$(m3 "$(sc lrsw_${arm}_s0_e42)" "$(sc lrsw_${arm}_s0_e43)" "$(sc lrsw_${arm}_s0_e44)")
  B=$(m3 "$(sc lrsw_${arm}_s1_e42)" "$(sc lrsw_${arm}_s1_e43)" "$(sc lrsw_${arm}_s1_e44)")
  C=$(m3 "$(sc lrsw_${arm}_s2_e42)" "$(sc lrsw_${arm}_s2_e43)" "$(sc lrsw_${arm}_s2_e44)")
  log "  $(printf '%-8s' "$arm")  $A   $B   $C    $(m3 "$A" "$B" "$C")"
done
log "  baseline  85.3   84.7   89.3    86.4   (amax 1.6, iters 8, steps 6000, alr 3e-4)"
log "  NOTE: held-out selection (select draw 42 / report 43+44) applies here too --"
log "        run dyna_harness/amax_holdout_select.sh logic before quoting a winner."
log "LRSWEEP_DONE"
