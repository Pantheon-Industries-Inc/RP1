#!/bin/bash
# Round S6 (input-sweep phase 3): gate test on the input-square WINNER + extra seeds.
# Winner of {zg0 (S3), z00, min0 (S5)} selected at runtime from summary.csv
# (mean over seeds 0/1 of h25 s42+s44). Arms:
#   wng_s0/s1: winner flags + --no-gate     (completes item 4: can the winner drop the gate?)
#   w_s2/s3:   winner flags, seeds 2/3      (robustness of the winner itself)
# Champion refs: schedamax s0=168 s1=154; champion-no-gate = S3 ng arms.
set -u
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
EVAL_TIMEOUT=14400

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver_ac90s6.log"; }
sc() { grep "^${1}," "$RES/summary.csv" | tail -1 | cut -d, -f2; }

run_eval() { # name gpu seed offset budget extra...
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5; shift 5
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached ($(sc "$name"))"; return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu timeout "$EVAL_TIMEOUT" $PY "$PLAN/eval_wm.py" \
    --config-name cube \
    seed="$seed" eval.dataset_name="$H5" ++bf16=true eval.img_size=224 \
    policy="$WM" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    output.filename="${name}.txt" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$RES/summary.csv"; return 1; }
  echo "${name},${sr}" >> "$RES/summary.csv"
  log "eval ${name}: ${sr}"
}

score() { # round-prefix arm -> mean over seeds 0/1 of (s42+s44), or -1
  local pre=$1 arm=$2 tot=0 n=0 x y sd
  for sd in 0 1; do
    x=$(sc "lipac90${pre}_${arm}_s${sd}_h25_s42"); y=$(sc "lipac90${pre}_${arm}_s${sd}_h25_s44")
    { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && continue
    tot=$(awk "BEGIN{print $tot + $x + $y}"); n=$((n + 1))
  done
  if [ "$n" -gt 0 ]; then awk "BEGIN{print $tot / $n}"; else echo -1; fi
}

ZG=$(score s3 zg0); Z0=$(score s5 z00); MN=$(score s5 min0)
best="zg0"; bs=$ZG; WFLAGS="--drop-zg"
awk "BEGIN{exit !($Z0 > $bs)}" && { best="z00"; bs=$Z0; WFLAGS="--drop-z0"; }
awk "BEGIN{exit !($MN > $bs)}" && { best="min0"; bs=$MN; WFLAGS="--drop-z0 --drop-zg"; }
log "input-square winner: ${best} (2-seed mean s42+s44 = ${bs}; zg0=${ZG} z00=${Z0} min0=${MN})"

BASE="--cache $CACHE5 --cache-td $CACHE1 --h5 $H5 --wm $WM --init-value $TD_WIN
      --horizon 5 --iters 8 --steps 8000 --n-step 50 --amax 3.5
      --expectile 0.1 --expectile-final 0.03 --critic-lr-final 1e-4
      --actor-lr 3e-4 --actor-lr-final 3e-5"

tr6() { # arm gpu extra...
  local arm=$1 gpu=$2; shift 2
  local out="$ACT/lip_ac90s6_${arm}.pt"
  [ -f "$out" ] && { log "train ${arm}: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip_ac.py" $BASE \
    --out "$out" --out-value "$MET/lip_ac90s6_${arm}_value.pt" "$@" \
    > "$LOGS/train_lip_ac90s6_${arm}.log" 2>&1 \
    || log "train ${arm} FAILED"
}
log "round S6: winner=${best} gate test (wng s0/s1) + winner seeds 2/3"
tr6 wng_s0 0 $WFLAGS --no-gate --seed 0 &
tr6 wng_s1 1 $WFLAGS --no-gate --seed 1 &
tr6 w_s2   2 $WFLAGS --seed 2 &
tr6 w_s3   3 $WFLAGS --seed 3 &
wait
log "round S6 trained"

ARMS="wng_s0 wng_s1 w_s2 w_s3"
for s in 42 44; do
  gpu=0
  for arm in $ARMS; do
    p="$ACT/lip_ac90s6_${arm}.pt"
    [ -f "$p" ] || continue
    run_eval "lipac90s6_${arm}_h25_s${s}" "$gpu" "$s" 25 50 solver=lip "solver.actor_path=$p" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

for arm in $ARMS; do
  x=$(sc "lipac90s6_${arm}_h25_s42"); y=$(sc "lipac90s6_${arm}_h25_s44")
  { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && continue
  log "arm ${arm} (${best}): s42+s44 = $(awk "BEGIN{print $x + $y}")"
done
log "(ref) champion schedamax: s0 = 168, s1 = 154"
log "(ref) champion no-gate (S3 ng): s0 = $(score s3 ng | cut -d. -f1) (2-seed mean)"
log "DONE. round S6 complete — input sweep + gate tests finished"
