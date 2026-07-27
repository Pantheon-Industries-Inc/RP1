#!/bin/bash
# Re-measure the full amax card under egl (standing egl instruction), chained to
# start only after the Dyna control finishes -- egl crashes under concurrent
# training and evals must be sequential.
#
# NO RETRAINING NEEDED: train_lip_ac.py reads cached latents built from the
# dataset's stored jpegs, so LIP training never renders and is renderer-
# independent. Only the evals change under egl. That is 41 evals, not 15
# retrainings + 45 evals.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/workspace/code/stable-worldmodel/scripts/plan
L=/workspace/logs; SUM=/workspace/results/summary_amaxegl.csv; DRV=$L/driver_amaxegl.log
WM=/workspace/models/v2WM
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
mkdir -p /workspace/results; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
m3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"}else{printf "%.1f",(a+b+c)/3}}'; }
cd /workspace/code/stable-worldmodel

log "waiting for the Dyna control to finish before touching the GPU"
while ! grep -q "DYNA_SPLIT_CONTROL_DONE" "$L/driver_dynasplit.log" 2>/dev/null; do
  pgrep -f "dyna_split_control|dsp_resume" >/dev/null || { log "Dyna driver gone without DONE -- proceeding anyway"; break; }
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 120; done
log "GPU idle; starting egl re-card"

# reuse the 4 egl cells already measured for the a16 actors during the renderer check
for c in $(grep "^eglchk_" /workspace/results/summary_lrsweep.csv 2>/dev/null); do
  n=${c%%,*}; v=${c##*,}; s=$(echo "$n" | sed -E 's/.*_s([0-9])_e.*/\1/'); d=${n##*_e}
  grep -q "^amaxegl_a16_s${s}_e${d}," "$SUM" || echo "amaxegl_a16_s${s}_e${d},${v}" >> "$SUM"
done
log "seeded $(grep -c '^amaxegl_a16' "$SUM") a16 cells from the renderer check"

for am in 1.0 1.2 1.4 1.6 1.8; do t=${am/./}; for s in 0 1 2; do
  A=/workspace/actors/lip4_amsw_a${t}_s${s}.pt
  [ -f "$A" ] || { log "  a${t}/s$s actor missing, skip"; continue; }
  for d in 42 43 44; do
    nm="amaxegl_a${t}_s${s}_e${d}"
    c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; continue; }
    CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
      seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 policy="$WM" solver=lip \
      "solver.actor_path=$A" output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
    grep -q "MUJOCO_GL=egl" "$L/eval_${nm}.log" || { log "  $nm: NOT egl -- refusing to record"; echo "${nm},FAIL" >> "$SUM"; continue; }
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
    echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
  done
done; done

log ""
log "=== AMAX CARD UNDER EGL (v2WM, h25, draws 42/43/44, full pool) ==="
log "  amax    s0     s1     s2    3-seed    osmesa ref"
for am in 1.0 1.2 1.4 1.6 1.8; do t=${am/./}
  A=$(m3 "$(sc amaxegl_a${t}_s0_e42)" "$(sc amaxegl_a${t}_s0_e43)" "$(sc amaxegl_a${t}_s0_e44)")
  B=$(m3 "$(sc amaxegl_a${t}_s1_e42)" "$(sc amaxegl_a${t}_s1_e43)" "$(sc amaxegl_a${t}_s1_e44)")
  C=$(m3 "$(sc amaxegl_a${t}_s2_e42)" "$(sc amaxegl_a${t}_s2_e43)" "$(sc amaxegl_a${t}_s2_e44)")
  case $am in 1.0) r=79.3;; 1.2) r=82.4;; 1.4) r=85.6;; 1.6) r=86.4;; 1.8) r=86.2;; esac
  log "  $am    $A   $B   $C    $(m3 "$A" "$B" "$C")      $r"
done
log "  If these track the osmesa column, egl==osmesa on cube is confirmed across"
log "  all 45 cells, not just the one seed the renderer check covered."
log "AMAX_EGL_RECARD_DONE"
