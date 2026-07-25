#!/bin/bash
# TwoRoom min0 campaign v2 (2026-07-14, pod a 4xH100, 224 cores).
# Same protocol/row-names as v1; scheduler rewritten:
#   - evals are CPU-bound (~29 min h25 / ~55 min h50, torch env rendering)
#     -> work-queue with NWORK concurrent evals, OMP_NUM_THREADS capped
#   - all 6 screen trainings run CONCURRENTLY with the anchor evals (GPU vs CPU);
#     the anchor gate is checked before any screen eval spends CPU
#   - seeds/deliverable phase picks the best MIN0 arm (fullin = control only,
#     user directive 2026-07-14)
# Idempotent via /workspace/results/summary_tworoom.csv (FAIL rows must be
# scrubbed before relaunch; done by the launcher).
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
H5=/workspace/caches/tworoom_play.h5
CACHE1=/workspace/caches/tworoom_lewm_fs1.pt
CACHE5=/workspace/caches/tworoom_lewm_fs5.pt
TD=$MET/td_e0.1_n50.pt
SUM=$RES/summary_tworoom.csv
DRV=$LOGS/driver_tworoom_min0.log
TRAIN_TIMEOUT=10800
EVAL_TIMEOUT=7200
NWORK=${NWORK:-12}
EVAL_THREADS=${EVAL_THREADS:-18}
QUEUE=$RES/.evalq
QLOCK=$RES/.evalq.lock

mkdir -p "$LOGS" "$RES" "$MET" "$ACT"
touch "$SUM"

log(){ echo "[$(date +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

run_eval(){ # name gpu seed offset budget surface(std|hard) extra...
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5 surface=$6; shift 6
  if grep -q "^${name}," "$SUM"; then
    log "eval ${name}: cached ($(sc "$name"))"; return 0
  fi
  local hard=()
  [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$EVAL_THREADS MKL_NUM_THREADS=$EVAL_THREADS \
    timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name tworoom_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" \
    "+video_dir=$RES/videos_${name}" "${hard[@]}" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "${sr:-}" ]; then
    log "eval ${name}: FAILED (see $LOGS/eval_${name}.log)"
    echo "${name},FAIL" >> "$SUM"; return 1
  fi
  echo "${name},${sr}" >> "$SUM"
  log "eval ${name}: ${sr}"
}

# ---- work queue: lines "name|seed|offset|budget|surface|extra hydra args"
qadd(){ echo "$1" >> "$QUEUE"; }
qpop(){ ( flock 9; head -n1 "$QUEUE" 2>/dev/null; sed -i '1d' "$QUEUE" 2>/dev/null ) 9>"$QLOCK"; }
worker(){
  local wid=$1 gpu line name seed offset budget surface extra
  gpu=$((wid % 4))
  while :; do
    line=$(qpop); [ -z "$line" ] && break
    IFS='|' read -r name seed offset budget surface extra <<<"$line"
    # shellcheck disable=SC2086
    run_eval "$name" "$gpu" "$seed" "$offset" "$budget" "$surface" $extra
  done
}
qrun(){ # drain the queue with NWORK workers
  local i pids=()
  for ((i=0; i<NWORK; i++)); do worker "$i" & pids+=($!); done
  wait "${pids[@]}"
}
qcard(){ # tag actor -> enqueue 12 cells
  local tag=$1 actor=$2
  for surface in std hard; do
    for seed in 42 43 44; do
      qadd "card_${tag}_${surface}_h25_s${seed}|${seed}|25|50|${surface}|solver=lip solver.actor_path=${actor}"
      qadd "card_${tag}_${surface}_h50_s${seed}|${seed}|50|100|${surface}|solver=lip solver.actor_path=${actor}"
    done
  done
}
card_sum(){ # tag -> sum of 12 cells (FAIL/missing -> -1)
  local tag=$1 total=0 v
  for surface in std hard; do
    for seed in 42 43 44; do
      for h in h25 h50; do
        v=$(sc "card_${tag}_${surface}_${h}_s${seed}")
        { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && { echo "-1"; return; }
        total=$(awk "BEGIN{print $total + $v}")
      done
    done
  done
  echo "$total"
}

# ------------------------------------------------------------------ arms
BASE_COMMON="--cache $CACHE5 --cache-td $CACHE1 --h5 $H5 --wm lewm_tworoom
  --init-value $TD --horizon 5 --iters 8 --steps 8000 --n-step 50
  --expectile 0.1 --critic-lr 1e-3 --actor-lr 3e-4"
DROPS="--drop-z0 --drop-zg"
ARMS="base ng sched md12 ax35 fullin"
MIN0_ARMS="base ng sched md12 ax35"

flags(){ case $1 in
  base)   echo "$DROPS";;
  ng)     echo "$DROPS --no-gate";;
  sched)  echo "$DROPS --critic-lr-final 1e-4 --expectile-final 0.03 --actor-lr-final 3e-5";;
  md12)   echo "$DROPS --max-delta 12";;
  ax35)   echo "$DROPS --amax 3.5";;
  fullin) echo "";;
  *) return 1;;
esac; }

train_arm(){ # gpu arm seed
  local gpu=$1 arm=$2 seed=$3
  local out="$ACT/trm_min0_${arm}_s${seed}.pt" fl
  fl=$(flags "$arm") || { log "unknown arm $arm"; return 1; }
  [ -f "$out" ] && { log "train ${arm}_s${seed}: cached"; return 0; }
  log "train ${arm}_s${seed}: start gpu${gpu} [$fl]"
  echo "CMD: train_lip_ac.py $BASE_COMMON $fl --seed $seed" > "$LOGS/train_trm_min0_${arm}_s${seed}.log"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" \
    $BASE_COMMON $fl --seed "$seed" \
    --out "$out" --out-value "$MET/trm_min0_${arm}_s${seed}_value.pt" \
    >> "$LOGS/train_trm_min0_${arm}_s${seed}.log" 2>&1 \
    || { log "train ${arm}_s${seed}: FAILED (see log)"; return 1; }
  local ef; ef=$(grep -E "^step" "$LOGS/train_trm_min0_${arm}_s${seed}.log" | tail -1 \
    | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  log "train ${arm}_s${seed}: done, E_final ${ef:-NA} (indicator only)"
}

rm -f "$QUEUE"; touch "$QUEUE"

# ------------------------------------------------------------------ phase 1:
# trainings (GPU) concurrent with anchor evals (CPU)
log "=== v2 phase 1: 6 screen trainings (GPU) + anchor evals (CPU queue, NWORK=$NWORK)"
(
  train_arm 0 base 0 & train_arm 1 ng 0 & train_arm 2 sched 0 &
  train_arm 3 md12 0 & train_arm 0 ax35 0 & train_arm 1 fullin 0 &
  wait
  log "screen trainings all done"
) &
TRAIN_PID=$!

for seed in 42 43 44; do
  for surface in std hard; do
    qadd "card_lv1ctrl_${surface}_h25_s${seed}|${seed}|25|50|${surface}|solver=lip solver.actor_path=$ACT/lv1_e01n50_seed1.pt"
    qadd "card_lv1ctrl_${surface}_h50_s${seed}|${seed}|50|100|${surface}|solver=lip solver.actor_path=$ACT/lv1_e01n50_seed1.pt"
  done
done
qadd "latent_std_h50_s42|42|50|100|std|"
qadd "latent_hard_h50_s42|42|50|100|hard|"
qrun
log "anchor queue drained"

anchor_fail=0
for surface in std hard; do
  for seed in 42 43 44; do
    for h in h25 h50; do
      v=$(sc "card_lv1ctrl_${surface}_${h}_s${seed}")
      { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && { log "anchor cell ${surface}_${h}_s${seed}: MISSING/FAIL"; anchor_fail=1; continue; }
      awk "BEGIN{exit !($v < 98)}" && { log "anchor cell ${surface}_${h}_s${seed}: $v < 98"; anchor_fail=1; }
    done
  done
done
[ "$anchor_fail" -eq 0 ] || die "anchor card degraded — rebuilt data does not reproduce the old harness"
log "anchor card OK (all >= 98). latent: std_h50 $(sc latent_std_h50_s42) (ref 68.0), hard_h50 $(sc latent_hard_h50_s42) (ref 56.0)"

wait "$TRAIN_PID" 2>/dev/null || true

# ------------------------------------------------------------------ phase 2: screen
log "=== v2 phase 2: screen evals (18 cells)"
for arm in $ARMS; do
  a="$ACT/trm_min0_${arm}_s0.pt"
  [ -f "$a" ] || { log "screen ${arm}: no ckpt, skip"; continue; }
  qadd "m0_${arm}_hard_h25_s42|42|25|50|hard|solver=lip solver.actor_path=$a"
  qadd "m0_${arm}_hard_h50_s42|42|50|100|hard|solver=lip solver.actor_path=$a"
  qadd "m0_${arm}_std_h50_s42|42|50|100|std|solver=lip solver.actor_path=$a"
done
qrun
log "screen queue drained"

log "=== screen results"
declare -A SSUM
for arm in $ARMS; do
  a=$(sc "m0_${arm}_hard_h25_s42"); b=$(sc "m0_${arm}_hard_h50_s42"); c=$(sc "m0_${arm}_std_h50_s42")
  if [ -z "$a" ] || [ -z "$b" ] || [ -z "$c" ] || [ "$a" = "FAIL" ] || [ "$b" = "FAIL" ] || [ "$c" = "FAIL" ]; then
    SSUM[$arm]=-1; log "screen ${arm}: INVALID (${a:-na}/${b:-na}/${c:-na})"
  else
    SSUM[$arm]=$(awk "BEGIN{print $a + $b + $c}")
    log "screen ${arm}: ${a}/${b}/${c} sum ${SSUM[$arm]}"
  fi
done

# ------------------------------------------------------------------ phase 3: cards
PROMO=""
for arm in $ARMS; do
  [ "${SSUM[$arm]}" = "300" ] && PROMO="$PROMO $arm"
done
if [ -z "$PROMO" ]; then
  log "no arm screened 300/300 — falling back to top-2 by screen sum"
  PROMO=$(for arm in $ARMS; do echo "${SSUM[$arm]} $arm"; done | sort -rn | head -2 | awk '{print $2}' | tr '\n' ' ')
fi
log "=== v2 phase 3: cards for [$PROMO]"
for arm in $PROMO; do
  qcard "m0${arm}-s0" "$ACT/trm_min0_${arm}_s0.pt"
done
qrun
log "cards queue drained"

# best arm for the deliverable: MIN0 ARMS ONLY (fullin = control)
best=""; best_sum=-1
for arm in $PROMO; do
  case " $MIN0_ARMS " in *" $arm "*) ;; *) log "card m0${arm}-s0 sum: $(card_sum "m0${arm}-s0") (control, not eligible)"; continue;; esac
  s=$(card_sum "m0${arm}-s0")
  log "card m0${arm}-s0 sum: $s"
  awk "BEGIN{exit !($s > $best_sum)}" && { best_sum=$s; best=$arm; }
done
if [ -z "$best" ]; then
  log "WARNING: no min0 arm was promoted — carding best min0 by screen sum before seeds"
  best=$(for arm in $MIN0_ARMS; do echo "${SSUM[$arm]} $arm"; done | sort -rn | head -1 | awk '{print $2}')
  qcard "m0${best}-s0" "$ACT/trm_min0_${best}_s0.pt"; qrun
  best_sum=$(card_sum "m0${best}-s0")
fi
log "best min0 arm: $best (card sum $best_sum / 1200)"

# ------------------------------------------------------------------ phase 4: seeds
log "=== v2 phase 4: $best seeds 1,2"
train_arm 0 "$best" 1 & train_arm 1 "$best" 2 &
wait
[ -f "$ACT/trm_min0_${best}_s1.pt" ] && qcard "m0${best}-s1" "$ACT/trm_min0_${best}_s1.pt"
[ -f "$ACT/trm_min0_${best}_s2.pt" ] && qcard "m0${best}-s2" "$ACT/trm_min0_${best}_s2.pt"
qrun
log "seed cards drained"

log "=== FINAL SUMMARY"
sort "$SUM" | tee -a "$DRV"
perfect=""
for arm in $PROMO $best; do
  for s in 0 1 2; do
    [ "$(card_sum "m0${arm}-s${s}")" = "1200" ] && perfect="$perfect m0${arm}-s${s}"
  done
done
perfect=$(echo "$perfect" | tr ' ' '\n' | sort -u | tr '\n' ' ')
log "PERFECT CARDS: ${perfect:-none}"
touch "$RES/tworoom_min0.DONE"
log "ALL DONE"
