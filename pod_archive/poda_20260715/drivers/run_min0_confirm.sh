#!/bin/bash
# min0 3-draw confirm: add s43 to the existing s42+s44 evals so min0 gets a
# 3-draw h25 number directly comparable to the champion's 88.0 (88/96/80).
# min0 = minimal-input actor [A, grad V, E] (drop z0 + z_g), sweep winner.
# Arms: 4 gated seeds (s5_min0_s0/s1, s6_w_s2/s3) + 2 no-gate (s6_wng_s0/s1).
# CONTROL: champion schedamax on s43 — must reproduce ~96 (guards the harness /
# guards against evaluating the wrong checkpoint). Also fills champion s43 for the
# no-gate/goal comparisons. s42 and s44 already in summary.csv from the sweep.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa

PLAN=/workspace/code/stable-worldmodel/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
ACT=/workspace/actors
PY=python3
WM=/workspace/ckpts/ogbench_cube_single_v2WM
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver_min0confirm.log"; }
sc() { grep "^${1}," "$RES/summary.csv" | tail -1 | cut -d, -f2; }

run_eval() { # name gpu seed actor
  local name=$1 gpu=$2 seed=$3 actor=$4
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached ($(sc "$name"))"; return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu timeout 14400 $PY "$PLAN/eval_wm.py" \
    --config-name cube \
    seed="$seed" eval.dataset_name="$H5" ++bf16=true eval.img_size=224 \
    policy="$WM" eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=lip "solver.actor_path=$actor" output.filename="${name}.txt" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED (see log)"; echo "${name},FAIL" >> "$RES/summary.csv"; return 1; }
  echo "${name},${sr}" >> "$RES/summary.csv"
  log "eval ${name}: ${sr}"
}

# --- wave 1: 4 gated min0 seeds on s43 (one per GPU)
log "min0 confirm: s43 for 4 gated seeds"
run_eval min0confirm_g_s0_h25_s43 0 43 "$ACT/lip_ac90s5_min0_s0.pt" &
run_eval min0confirm_g_s1_h25_s43 1 43 "$ACT/lip_ac90s5_min0_s1.pt" &
run_eval min0confirm_g_s2_h25_s43 2 43 "$ACT/lip_ac90s6_w_s2.pt" &
run_eval min0confirm_g_s3_h25_s43 3 43 "$ACT/lip_ac90s6_w_s3.pt" &
wait
# --- wave 2: 2 no-gate min0 seeds + champion control
log "min0 confirm: s43 for no-gate seeds + champion control"
run_eval min0confirm_ng_s0_h25_s43 0 43 "$ACT/lip_ac90s6_wng_s0.pt" &
run_eval min0confirm_ng_s1_h25_s43 1 43 "$ACT/lip_ac90s6_wng_s1.pt" &
run_eval min0confirm_champ_ctrl_h25_s43 2 43 "$ACT/lip_ac90_schedamax.pt" &
wait

# --- 3-draw means. s42/s44 read from the sweep rows (gated: s5_min0/s6_w; ng: s6_wng)
log "=== control: champion schedamax s43 = $(sc min0confirm_champ_ctrl_h25_s43) (expect ~96; sweep s42=88 s44=80 -> 3-draw 88.0)"
mean3() { # label  s42  s43  s44
  awk -v a="$2" -v b="$3" -v c="$4" 'BEGIN{ if(a=="" || b=="" || c=="" || a=="FAIL" || b=="FAIL" || c=="FAIL"){print "NA"} else {printf "%.1f", (a+b+c)/3} }'
}
gtot=0; gn=0
declare -A G42=( [s0]=$(sc lipac90s5_min0_s0_h25_s42) [s1]=$(sc lipac90s5_min0_s1_h25_s42) [s2]=$(sc lipac90s6_w_s2_h25_s42) [s3]=$(sc lipac90s6_w_s3_h25_s42) )
declare -A G44=( [s0]=$(sc lipac90s5_min0_s0_h25_s44) [s1]=$(sc lipac90s5_min0_s1_h25_s44) [s2]=$(sc lipac90s6_w_s2_h25_s44) [s3]=$(sc lipac90s6_w_s3_h25_s44) )
for s in s0 s1 s2 s3; do
  m=$(mean3 "$s" "${G42[$s]}" "$(sc min0confirm_g_${s}_h25_s43)" "${G44[$s]}")
  log "min0 gated ${s}: s42=${G42[$s]} s43=$(sc min0confirm_g_${s}_h25_s43) s44=${G44[$s]} -> 3-draw ${m}"
  [ "$m" != "NA" ] && { gtot=$(awk "BEGIN{print $gtot + $m}"); gn=$((gn+1)); }
done
[ "$gn" -gt 0 ] && log "min0 gated 3-draw mean over ${gn} seeds = $(awk "BEGIN{printf \"%.1f\", $gtot/$gn}") (champion 88.0)"
for s in s0 s1; do
  a=$(sc lipac90s6_wng_${s}_h25_s42); c=$(sc lipac90s6_wng_${s}_h25_s44)
  log "min0 no-gate ${s}: s42=${a} s43=$(sc min0confirm_ng_${s}_h25_s43) s44=${c} -> 3-draw $(mean3 $s $a $(sc min0confirm_ng_${s}_h25_s43) $c)"
done
log "DONE. min0 3-draw confirm complete"
