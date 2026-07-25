#!/bin/bash
# PushT LIP-AC experiment: train TD critic + LIP actor in tandem (4 GPUs).
#
# Motivation: the sequential protocol picks the TD by CEM ranking, but LIP
# consumes the value's gradient field — on PushT the CEM-picked teacher
# (td_e0.03_n5: 76/54 on h25/h50 s42) yields LIP 66/36. Four tandem arms probe
# whether co-training closes that gap:
#   ac_fresh      critic from scratch (2k pretrain), teacher EMA, freeze @80%
#   ac_warm       critic warm-started from the DEEP-SWEEP TD winner (tau0.01 n1)
#   ac_schedamax  warm + cube-record recipe: critic-lr 1e-3->1e-4 cosine,
#                 expectile anneal 0.1->0.01, actor-lr cosine, amax 3.5
#   ac_expand03   warm + value expansion w=0.3 (TD backups on planner rollouts)
# Baselines to beat on s42 (h25+h50): teacher td2_e0.01_n1 = 154, LIP2 = 128,
# LIP2+R8 = 142. Cube lesson: NO restarts on AC actors (zero-init specialized);
# one probe cell checks that on PushT too.
# Then: LIP-eval each arm (selection on s42 h25+h50), CEM-eval each arm's
# teacher (did tandem training hurt/help the value itself?), confirm the best
# arm on all 6 cells. Sequential baselines already live in summary.csv
# (lipwin_k8_lr3e-4_st8000_*, tdwin_td_e0.03_n5_*).
#
# Prereqs on the pod (from the run_all_pusht.sh run): caches, h5, metrics/td_e0.03_n5.pt,
# and scripts/plan/train_lip_ac.py synced into /workspace/code/stable-worldmodel.
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
CACHES=/workspace/caches
PY=python3
WM=lewm_pusht_official
H5=/workspace/swm_home/datasets/pusht_expert_train.h5
CACHE1=$CACHES/pusht_official_fs1.pt
CACHE5=$CACHES/pusht_official_fs5.pt
TD_WIN=$MET/td2_e0.01_n1.pt
EVAL_TIMEOUT=14400

mkdir -p "$LOGS" "$RES" "$MET" "$ACT"
touch "$RES/summary.csv"

log() { echo "[$(date +%H:%M:%S)] [ac] $*" | tee -a "$LOGS/driver.log"; }
die() { log "FATAL: $*"; exit 1; }

[ -f "$CACHE1" ] || die "missing $CACHE1"
[ -f "$CACHE5" ] || die "missing $CACHE5"
[ -f "$TD_WIN" ] || die "missing sequential TD winner $TD_WIN (needed for warm arms)"
[ -f "$PLAN/train_lip_ac.py" ] || die "train_lip_ac.py not synced into $PLAN"

# run one eval; positional: name gpu seed offset budget, then extra hydra args
run_eval() {
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5; shift 5
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached ($(grep "^${name}," "$RES/summary.csv" | tail -1 | cut -d, -f2))"
    return 0
  fi
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu timeout "$EVAL_TIMEOUT" $PY "$PLAN/eval_wm.py" \
    --config-name pusht_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" \
    "+video_dir=$RES/videos_${name}" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "${sr:-}" ]; then
    log "eval ${name}: FAILED (no success_rate; see $LOGS/eval_${name}.log)"
    echo "${name},FAIL" >> "$RES/summary.csv"
    return 1
  fi
  echo "${name},${sr}" >> "$RES/summary.csv"
  log "eval ${name}: ${sr}"
}

# ---------------------------------------------------------------- phase 1: train
# All arms share the sequential LIP winner's actor recipe (K8, lr 3e-4, 8k steps)
# and the sequential TD winner's critic hypers (expectile 0.03, n-step 5).
ac_train() { # arm gpu extra-args...
  local arm=$1 gpu=$2; shift 2
  local out="$ACT/lip_ac_${arm}.pt" outv="$MET/lip_ac_${arm}_value.pt"
  [ -f "$out" ] && { log "train ${arm}: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm $WM \
    --out "$out" --out-value "$outv" \
    --horizon 5 --iters 8 --steps 8000 --actor-lr 3e-4 \
    --expectile 0.01 --n-step 1 "$@" \
    > "$LOGS/train_lip_ac_${arm}.log" 2>&1 \
    || log "train ${arm} FAILED"
}

log "phase 1: training 4 AC arms"
ac_train fresh    0 &
ac_train warm     1 --init-value "$TD_WIN" &
# schedamax: overrides the shared expectile via later flag (last wins in argparse)
[ -f "$ACT/lip_ac_schedamax.pt" ] || CUDA_VISIBLE_DEVICES=2 $PY "$PLAN/train_lip_ac.py" \
  --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm $WM \
  --out "$ACT/lip_ac_schedamax.pt" --out-value "$MET/lip_ac_schedamax_value.pt" \
  --horizon 5 --iters 8 --steps 8000 --actor-lr 3e-4 --actor-lr-final 3e-5 \
  --n-step 1 --init-value "$TD_WIN" \
  --critic-lr 1e-3 --critic-lr-final 1e-4 \
  --expectile 0.1 --expectile-final 0.01 --amax 3.5 \
  > "$LOGS/train_lip_ac_schedamax.log" 2>&1 &
ac_train expand03 3 --init-value "$TD_WIN" --expand-weight 0.3 &
wait
log "AC arms trained"

# ---------------------------------------------------------------- phase 2: select (s42)
ARMS="fresh warm schedamax expand03"
for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  gpu=0
  for arm in $ARMS; do
    a="$ACT/lip_ac_${arm}.pt"
    [ -f "$a" ] || continue
    run_eval "lipac_${arm}_${cname}_s42" "$gpu" 42 "$coff" "$cbud" \
      solver=lip "solver.actor_path=$a" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

# diagnostic: CEM on each tandem teacher — did co-training move the value itself?
for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  gpu=0
  for arm in $ARMS; do
    m="$MET/lip_ac_${arm}_value.pt"
    [ -f "$m" ] || continue
    run_eval "accem_${arm}_${cname}_s42" "$gpu" 42 "$coff" "$cbud" "+metric=$m" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

# ---------------------------------------------------------------- phase 3: confirm winner
best=""; best_score=-1
for arm in $ARMS; do
  x=$(grep "^lipac_${arm}_h25_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
  y=$(grep "^lipac_${arm}_h50_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
  { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && continue
  s=$(awk "BEGIN{print $x + $y}")
  log "arm ${arm}: h25 ${x} + h50 ${y} = ${s}"
  if awk "BEGIN{exit !($s > $best_score)}"; then best_score=$s; best=$arm; fi
done
[ -n "$best" ] || die "no AC arm produced two valid selection scores"
log "AC winner: ${best} (h25+h50 s42 = ${best_score}; LIP2=128 LIP2R8=142 teacher=154)"

for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  gpu=0
  for seed in 42 43 44; do
    run_eval "lipacwin_${best}_${cname}_s${seed}" "$gpu" "$seed" "$coff" "$cbud" \
      solver=lip "solver.actor_path=$ACT/lip_ac_${best}.pt" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

# restarts probe (cube: restarts HURT AC actors; verify on PushT, one cell)
run_eval "lipacwin_${best}_R8probe_h50_s42" 0 42 50 100 \
  solver=lip "solver.actor_path=$ACT/lip_ac_${best}.pt" solver.restarts=8 solver.restart_noise=0.5

log "AC ALL DONE. Compare lipacwin_${best}_* vs lip2R8_* / lip2win_* / td2win_* in $RES/summary.csv"
