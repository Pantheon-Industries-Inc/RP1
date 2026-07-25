#!/bin/bash
# Round S7: min0 hyper sweep — SCREEN phase (25 arms, seed 0, 3-draw eval each).
# Two-pod fleet (volumes NOT shared; each pod runs only its own queue):
#   a = 31.24.80.32:15419    12 arms: st16k st12k k4 k12 ax25 ax30 ax40 ax45 + grid mw01 row
#   b = 87.120.211.204:18099 13 arms: st20k [20k on pod b per user] + grid mw03/mw05 rows + ns25 ns100
# Grid: mean-weight{0.1,0.3,0.5} x tau{ref=0.1->0.03, f10=0.1 const, e001=0.1->0.01,
#       e205=0.2->0.05, f50=0.5 const=plain-TD control} minus banked baseline cell
#       (mw01/tref = min0 gated, 87.8 over 4 seeds).
# Base: gated min0 (--drop-z0 --drop-zg) on schedamax; seed 0. K is stored in the ckpt and
# the solver deploys ck["iters"], so k4/k12 eval at their trained K automatically.
# SELECT BY 3-DRAW MEAN ONLY — E_final is logged as an indicator, never selects (HANDOFF §6);
# tau arms rescale V so E is not even comparable across arms.
# Confirm phase (after both pods DONE, global top-3 chosen):
#   run_ac90s7_ogbench.sh <a|b> confirm "<arm> <arm> ..."   -> seed 1 + 3-draw evals. Idempotent.
set -u
MODE=${1:?usage: run_ac90s7_ogbench.sh a|b [confirm "<arms>"]}
PHASE=${2:-screen}
CARMS=${3:-}
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
WM=/workspace/ckpts/ogbench_cube_single_v2WM
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5
CACHE1=/workspace/caches/cube_full_fs1.pt
CACHE5=/workspace/caches/cube_full_fs5.pt
TD_WIN=$MET/cf_dE_t003n50.pt
SUM=$RES/summary_s7${MODE}.csv
DRV=$LOGS/driver_ac90s7${MODE}.log
TRAIN_TIMEOUT=28800
EVAL_TIMEOUT=14400
touch "$SUM"

log(){ echo "[$(date +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$RES"/summary_s7*.csv 2>/dev/null | tail -1 | cut -d, -f2; }

flags(){ case $1 in
  mw01tf10)  echo "--expectile-final 0.1";;
  mw01te001) echo "--expectile-final 0.01";;
  mw01te205) echo "--expectile 0.2 --expectile-final 0.05";;
  mw01tf50)  echo "--expectile 0.5 --expectile-final 0.5";;
  mw03tref)  echo "--mean-weight 0.3";;
  mw03tf10)  echo "--mean-weight 0.3 --expectile-final 0.1";;
  mw03te001) echo "--mean-weight 0.3 --expectile-final 0.01";;
  mw03te205) echo "--mean-weight 0.3 --expectile 0.2 --expectile-final 0.05";;
  mw03tf50)  echo "--mean-weight 0.3 --expectile 0.5 --expectile-final 0.5";;
  mw05tref)  echo "--mean-weight 0.5";;
  mw05tf10)  echo "--mean-weight 0.5 --expectile-final 0.1";;
  mw05te001) echo "--mean-weight 0.5 --expectile-final 0.01";;
  mw05te205) echo "--mean-weight 0.5 --expectile 0.2 --expectile-final 0.05";;
  mw05tf50)  echo "--mean-weight 0.5 --expectile 0.5 --expectile-final 0.5";;
  ax25) echo "--amax 2.5";;    ax30) echo "--amax 3.0";;
  ax40) echo "--amax 4.0";;    ax45) echo "--amax 4.5";;
  k4)  echo "--iters 4";;      k12) echo "--iters 12";;
  st12k) echo "--steps 12000";; st16k) echo "--steps 16000";; st20k) echo "--steps 20000";;
  ns25) echo "--n-step 25";;   ns100) echo "--n-step 100";;
  *) return 1;;
esac; }

# schedamax min0 base; per-arm flags are appended AFTER and argparse last-wins overrides.
BASE="--cache $CACHE5 --cache-td $CACHE1 --h5 $H5 --wm $WM --init-value $TD_WIN
      --horizon 5 --iters 8 --steps 8000 --n-step 50 --amax 3.5
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4
      --actor-lr 3e-4 --actor-lr-final 3e-5 --drop-z0 --drop-zg"

train_arm(){ # gpu arm seed
  local gpu=$1 arm=$2 seed=$3 fl out
  out="$ACT/lip_ac90s7_${arm}_s${seed}.pt"
  fl=$(flags "$arm") || { log "train ${arm}_s${seed}: UNKNOWN ARM"; return 1; }
  [ -f "$out" ] && { log "train ${arm}_s${seed}: cached"; return 0; }
  log "train ${arm}_s${seed}: start gpu${gpu} [${fl}]"
  CUDA_VISIBLE_DEVICES=$gpu timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" $BASE $fl \
    --seed "$seed" --out "$out" --out-value "$MET/lip_ac90s7_${arm}_s${seed}_value.pt" \
    > "$LOGS/train_lip_ac90s7_${arm}_s${seed}.log" 2>&1 \
    || { log "train ${arm}_s${seed}: FAILED (see log)"; return 1; }
  local ef; ef=$(grep -E "^step" "$LOGS/train_lip_ac90s7_${arm}_s${seed}.log" | tail -1 \
    | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  log "train ${arm}_s${seed}: done, E_final ${ef:-NA} (indicator only)"
}

eval_arm(){ # gpu arm seed draw
  local gpu=$1 arm=$2 seed=$3 s=$4 name p c sr
  name="lipac90s7_${arm}_s${seed}_h25_s${s}"; p="$ACT/lip_ac90s7_${arm}_s${seed}.pt"
  [ -f "$p" ] || { log "eval ${name}: no ckpt"; return 1; }
  c=$(sc "$name"); [ -n "$c" ] && { log "eval ${name}: cached (${c})"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name cube \
    seed="$s" eval.dataset_name="$H5" ++bf16=true eval.img_size=224 \
    policy="$WM" eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=lip "solver.actor_path=$p" output.filename="${name}.txt" \
    > "$LOGS/eval_${name}.log" 2>&1
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$SUM"; return 1; }
  echo "${name},${sr}" >> "$SUM"; log "eval ${name}: ${sr}"
}

do_arm(){ # gpu arm seed — train, then 3-draw eval, all on this GPU
  local gpu=$1 arm=$2 seed=$3 s
  train_arm "$gpu" "$arm" "$seed" || return 1
  for s in 42 43 44; do eval_arm "$gpu" "$arm" "$seed" "$s"; done
}

worker(){ # gpu arm...
  local gpu=$1 arm; shift
  for arm in "$@"; do do_arm "$gpu" "$arm" 0; done
  log "gpu${gpu} queue done"
}

mean3(){ # arm seed
  local x y z
  x=$(sc "lipac90s7_${1}_s${2}_h25_s42"); y=$(sc "lipac90s7_${1}_s${2}_h25_s43"); z=$(sc "lipac90s7_${1}_s${2}_h25_s44")
  awk -v a="${x:-}" -v b="${y:-}" -v c="${z:-}" \
    'BEGIN{if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL")print "NA"; else printf "%.1f",(a+b+c)/3}'
}

if [ "$PHASE" = confirm ]; then
  [ -n "$CARMS" ] || { log "confirm: no arms given"; exit 1; }
  log "confirm phase (${MODE}): seed 1 for: ${CARMS}"
  gpu=0; for arm in $CARMS; do do_arm "$gpu" "$arm" 1 & gpu=$(( (gpu+1) % 4 )); done; wait
  for arm in $CARMS; do log "confirm ${arm}: s0=$(mean3 "$arm" 0) s1=$(mean3 "$arm" 1)"; done
  touch "$RES/s7_${MODE}_confirm.DONE"; log "DONE confirm ${MODE}"; exit 0
fi

if [ "$MODE" = a ]; then
  ARMS_LOCAL="st16k st12k k4 k12 ax25 ax30 ax40 ax45 mw01tf10 mw01te001 mw01te205 mw01tf50"
  log "S7 screen pod a: 12 arms (queues balanced by est. duration)"
  worker 0 st16k ax25 k4 &
  worker 1 st12k ax30 mw01tf10 &
  worker 2 k12 ax40 mw01te001 &
  worker 3 ax45 mw01te205 mw01tf50 &
else
  ARMS_LOCAL="st20k mw03tref mw03tf10 mw03te001 mw03te205 mw03tf50 mw05tref mw05tf10 mw05te001 mw05te205 mw05tf50 ns25 ns100"
  log "S7 screen pod b: 13 arms (st20k here per directive)"
  worker 0 st20k mw03tref &
  worker 1 mw03tf10 mw03te001 mw03te205 mw03tf50 &
  worker 2 mw05tref mw05tf10 mw05te001 mw05te205 &
  worker 3 mw05tf50 ns25 ns100 &
fi
wait
log "=== screen ${MODE} local ranking (3-draw mean, seed 0; refs: min0 87.8 / champion 88.0) ==="
for arm in $ARMS_LOCAL; do log "  ${arm}: $(mean3 "$arm" 0)"; done
touch "$RES/s7_${MODE}.DONE"
log "DONE screen ${MODE} — global ranking + confirm dispatch happens after both pods finish"
