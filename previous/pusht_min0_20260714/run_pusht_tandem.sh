#!/bin/bash
# PushT tandem-first min0 campaign (user directive: sweep IN tandem space,
# select the deployed object by LIP-eval — no CEM-mediated teacher selection).
#
# 6 min0 tandem arms (all --drop-z0 --drop-zg, schedamax recipe, warm-started
# from Phase-A standalone metrics; the h25 CEM half-sweep is the prior that
# picked these corners + the mechanism ablation):
#   old_pi   old teacher corner (t0.01 n1) + tandem imagination-matching
#   old      old teacher corner, pure tandem (the drafted run_ac_pusht arm)
#   im_pi    imagination-matched single-frame (h25 92 = old teacher)
#   wn3a_pi  win3 t0.01 n1 (h25 88)
#   wn3b_pi  win3 t0.03 n5 (backup-robust corner, h25 84)
#   vi_pi    delta t0.01 n1 (h25 86; grad through dz gives the ACTOR
#            arrival-momentum sensitivity CEM never queried)
#
# Selection: LIP-eval h25+h50 s42, one-shot AND R8 per arm (restarts were
# load-bearing for sequential LIP on PushT; cube says AC actors hate them —
# measure both). Winner: seed-1 guard, then 3-draw headline (both modes),
# then finalcem (winner's tandem value CEM 3-draw) as the value diagnostic.
# Baselines: teacher CEM 92+62=154, LIP2+R8 142 (88 h25), LIP2 128.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8  # evals are CPU-bound; uncapped OMP thrashes

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
WM=lewm_pusht_official
H5=/workspace/swm_home/datasets/pusht_expert_train.h5
CACHE1=/workspace/caches/pusht_official_fs1.pt
CACHE5=/workspace/caches/pusht_official_fs5.pt
EVAL_TIMEOUT=14400
STOP_AFTER=${STOP_AFTER:-}

mkdir -p "$LOGS" "$RES" "$MET" "$ACT" /workspace/locks
exec 9>/workspace/locks/tandem.lock
flock -n 9 || { echo "another tandem driver instance is running; exiting"; exit 1; }
touch "$RES/summary.csv"

log() { echo "[$(date +%H:%M:%S)] [tandem] $*" | tee -a "$LOGS/min0_driver.log"; }
die() { log "FATAL: $*"; exit 1; }

run_eval() { # name gpu seed offset budget, then extra hydra args
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5; shift 5
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached ($(grep "^${name}," "$RES/summary.csv" | tail -1 | cut -d, -f2))"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu timeout "$EVAL_TIMEOUT" $PY "$PLAN/eval_wm.py" \
    --config-name pusht_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1 9>&-
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "${sr:-}" ]; then
    log "eval ${name}: FAILED (see $LOGS/eval_${name}.log)"
    echo "${name},FAIL" >> "$RES/summary.csv"
    return 1
  fi
  echo "${name},${sr}" >> "$RES/summary.csv"
  log "eval ${name}: ${sr}"
}

sr_of() { grep "^${1}," "$RES/summary.csv" | tail -1 | cut -d, -f2; }
combined() {
  local x y; x=$(sr_of "$1"); y=$(sr_of "$2")
  { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && return 1
  awk "BEGIN{print $x + $y}"
}

R8ARGS="solver.restarts=8 solver.restart_noise=0.5"

# ---------------------------------------------------------------- arms
# all min0 arms are NO-GATE (cube: gate expressively redundant on the MLP,
# 87.8 gated / 87.3 no-gate tied; handoff prefers no-gate); old_pi_g is the
# single gated control since gate-neutrality was never measured on PushT
arm_meta() { # name -> "init tau n pimg gate"
  case "$1" in
    old_pi)   echo "$MET/td2_e0.01_n1.pt 0.01 1 1 0" ;;
    old)      echo "$MET/td2_e0.01_n1.pt 0.01 1 0 0" ;;
    im_pi)    echo "$MET/im_e0.01_n1.pt 0.01 1 1 0" ;;
    wn3a_pi)  echo "$MET/wn3_e0.01_n1.pt 0.01 1 1 0" ;;
    wn3b_pi)  echo "$MET/wn3_e0.03_n5.pt 0.03 5 1 0" ;;
    vi_pi)    echo "$MET/vi_e0.01_n1.pt 0.01 1 1 0" ;;
    old_pi_g) echo "$MET/td2_e0.01_n1.pt 0.01 1 1 1" ;;
    # win3 ("Markov-3") tandem sweep: tau x n x p-imag around the h25 prior
    wn3c_pi)  echo "$MET/wn3_e0.1_n1.pt 0.1 1 1 0" ;;
    wn3d_pi)  echo "$MET/wn3_e0.01_n3.pt 0.01 3 1 0" ;;
    wn3e_pi)  echo "$MET/wn3_e0.1_n15.pt 0.1 15 1 0" ;;
    wn3f_pi)  echo "$MET/wn3_e0.03_n1.pt 0.03 1 1 0" ;;
    wn3_nopi) echo "$MET/wn3_e0.01_n1.pt 0.01 1 0 0" ;;
    *) return 1 ;;
  esac
}
ARMS="old_pi old im_pi wn3a_pi wn3b_pi vi_pi old_pi_g wn3c_pi wn3d_pi wn3e_pi wn3f_pi wn3_nopi"

ac_train() { # arm gpu seed suffix
  local arm=$1 gpu=$2 seed=$3 sfx=${4:-}
  set -- $(arm_meta "$arm") || die "unknown arm $arm"
  local init=$1 tau=$2 n=$3 pimg=$4 gate=$5
  local extra=""
  [ "$pimg" = "1" ] && extra="--p-imag 0.5"
  [ "$gate" = "0" ] && extra="$extra --no-gate"
  local out="$ACT/tnd_${arm}${sfx}.pt" outv="$MET/tnd_${arm}${sfx}_value.pt"
  [ -f "$out" ] && { log "train ${arm}${sfx}: cached"; return 0; }
  [ -f "$init" ] || { log "train ${arm}: MISSING init $init"; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm $WM \
    --out "$out" --out-value "$outv" \
    --horizon 5 --iters 8 --steps 8000 \
    --init-value "$init" --n-step "$n" \
    --expectile 0.1 --expectile-final "$tau" \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --amax 3.5 \
    --drop-z0 --drop-zg $extra --seed "$seed" \
    > "$LOGS/train_tnd_${arm}${sfx}.log" 2>&1 9>&- \
    || log "train ${arm}${sfx} FAILED"
}

log "T1: training tandem min0 arms: ${ARMS}"
gpu=0
for arm in $ARMS; do
  ac_train "$arm" "$gpu" 0 &
  gpu=$(( (gpu + 1) % 4 )); [ "$gpu" -eq 0 ] && wait
done
wait
log "T1 trained"
[ "$STOP_AFTER" = "t1" ] && { log "STOP_AFTER=t1"; exit 0; }

# ---------------------------------------------------------------- T2: LIP evals
log "T2: LIP-eval all arms, one-shot + R8, h25+h50 s42"
for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  gpu=0
  for arm in $ARMS; do
    a="$ACT/tnd_${arm}.pt"
    [ -f "$a" ] || continue
    run_eval "tnd_${arm}_${cname}_s42" "$gpu" 42 "$coff" "$cbud" \
      solver=lip "solver.actor_path=$a" &
    gpu=$(( (gpu + 1) % 4 )); [ "$gpu" -eq 0 ] && wait
    run_eval "tnd_${arm}_R8_${cname}_s42" "$gpu" 42 "$coff" "$cbud" \
      solver=lip "solver.actor_path=$a" $R8ARGS &
    gpu=$(( (gpu + 1) % 4 )); [ "$gpu" -eq 0 ] && wait
  done
  wait
done

# rank: best of {one-shot, R8} per arm
log "T2 ranking (teacher CEM 154, LIP2+R8 142):"
WIN_ARM=""; WIN_MODE="plain"; WIN_SCORE=""
for arm in $ARMS; do
  for mode in plain R8; do
    if [ "$mode" = "plain" ]; then
      c=$(combined "tnd_${arm}_h25_s42" "tnd_${arm}_h50_s42") || continue
    else
      c=$(combined "tnd_${arm}_R8_h25_s42" "tnd_${arm}_R8_h50_s42") || continue
    fi
    log "  ${arm} ${mode}: ${c}"
    if [ -z "$WIN_SCORE" ] || awk "BEGIN{exit !($c > $WIN_SCORE)}"; then
      WIN_SCORE=$c; WIN_ARM=$arm; WIN_MODE=$mode
    fi
  done
done
[ -n "$WIN_ARM" ] || die "T2: no arm produced valid scores"
log "T2 winner: ${WIN_ARM} (${WIN_MODE}, ${WIN_SCORE})"
[ "$STOP_AFTER" = "t2" ] && { log "STOP_AFTER=t2"; exit 0; }

# ---------------------------------------------------------------- T3: seed guard
log "T3: seed-1 guard on ${WIN_ARM}"
ac_train "$WIN_ARM" 0 1 "_s1"
WFLAGS=""
[ "$WIN_MODE" = "R8" ] && WFLAGS=$R8ARGS
run_eval "tnd_${WIN_ARM}_s1_h25_s42" 0 42 25 50 solver=lip \
  "solver.actor_path=$ACT/tnd_${WIN_ARM}_s1.pt" $WFLAGS &
run_eval "tnd_${WIN_ARM}_s1_h50_s42" 1 42 50 100 solver=lip \
  "solver.actor_path=$ACT/tnd_${WIN_ARM}_s1.pt" $WFLAGS &
wait
WIN_CKPT=$ACT/tnd_${WIN_ARM}.pt
c1=$(combined "tnd_${WIN_ARM}_s1_h25_s42" "tnd_${WIN_ARM}_s1_h50_s42" || echo "")
if [ -n "$c1" ] && awk "BEGIN{exit !($c1 > $WIN_SCORE)}"; then
  WIN_CKPT=$ACT/tnd_${WIN_ARM}_s1.pt
  log "seed-1 beats seed-0 (${c1} vs ${WIN_SCORE}); confirming s1"
fi
echo "${WIN_ARM},${WIN_MODE},${WIN_SCORE},${WIN_CKPT},seed1=${c1:-n/a}" > "$RES/tandem_winner.txt"

# ---------------------------------------------------------------- T4: 3-draw headline
log "T4: 3-draw headline for ${WIN_ARM} (both inference modes)"
for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  gpu=0
  for seed in 42 43 44; do
    run_eval "final_${WIN_ARM}_${cname}_s${seed}" "$gpu" "$seed" "$coff" "$cbud" \
      solver=lip "solver.actor_path=$WIN_CKPT" &
    gpu=$(( gpu + 1 ))
    run_eval "final_${WIN_ARM}_R8_${cname}_s${seed}" "$gpu" "$seed" "$coff" "$cbud" \
      solver=lip "solver.actor_path=$WIN_CKPT" $R8ARGS &
    gpu=$(( (gpu + 1) % 4 )); [ "$gpu" -eq 0 ] && wait
  done
  wait
done
log "T4 headline rows:"
grep -E "^final_" "$RES/summary.csv" | while read -r l; do log "  $l"; done
[ "$STOP_AFTER" = "t4" ] && { log "STOP_AFTER=t4"; exit 0; }

# ---------------------------------------------------------------- T5: value diagnostic
log "T5: winner tandem value under CEM, 3-draw (did tandem move the value itself?)"
WVAL=$MET/tnd_${WIN_ARM}_value.pt
[ "$WIN_CKPT" = "$ACT/tnd_${WIN_ARM}_s1.pt" ] && WVAL=$MET/tnd_${WIN_ARM}_s1_value.pt
for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  gpu=0
  for seed in 42 43 44; do
    run_eval "finalcem_${WIN_ARM}_${cname}_s${seed}" "$gpu" "$seed" "$coff" "$cbud" \
      "+metric=$WVAL" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done
log "ALL DONE"
