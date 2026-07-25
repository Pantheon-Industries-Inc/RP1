#!/bin/bash
# OGBench lewm-cube LIP-AC experiment: TD critic + LIP actor trained in tandem (4×H100).
#
# Sequential full-protocol baselines (pod_migration_20260709/WRITEUP, h25/h50 primitive-step goals):
#   latent+CEM   h25: 80/84/62 (s42/43/44)      h50 mean 40.0 (s42 ~44)
#   TD+CEM       h25: 82/90/66                  h50 mean 54.0   teacher = cf_dE_t003n50 (tau .03, n50)
#   LIP v1       h25: 88/96/78                  h50 mean 74.0   actor  = lip_n50_v1 (K8, lr3e-4, 8k)
# On cube LIP already beats CEM — the AC question is whether tandem training extends the lead
# (vs PushT, where it should close a deficit).
#
# Arms (all reuse the sequential winners' recipes: actor K8/lr3e-4/8k, critic tau0.03/n50):
#   fresh      critic from scratch (2k pretrain), teacher EMA, freeze @80%
#   warm       critic warm-started from cf_dE_t003n50
#   nofreeze   warm, teacher never frozen
#   expand03   warm + value expansion w=0.3 (TD backups on planner rollouts)
# Phases: 0 canary anchors (re-eval sequential winners on this pod) -> 1 train 4 arms
# -> 2 select on s42 h25+h50 + CEM-eval each arm's teacher -> 3 confirm winner on 6 cells.
# Idempotent via results/summary.csv (delete rows to force reruns).
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
LIP_WIN=$ACT/lip_n50_v1.pt
EVAL_TIMEOUT=14400

mkdir -p "$LOGS" "$RES" "$MET" "$ACT"
touch "$RES/summary.csv"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver_ac.log"; }
die() { log "FATAL: $*"; exit 1; }

for f in "$CACHE1" "$CACHE5" "$TD_WIN" "$LIP_WIN" "$H5" "$PLAN/train_lip_ac.py"; do
  [ -e "$f" ] || die "missing prerequisite: $f"
done

# run one eval; positional: name gpu seed offset budget, then extra hydra args
run_eval() {
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5; shift 5
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached ($(grep "^${name}," "$RES/summary.csv" | tail -1 | cut -d, -f2))"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu timeout "$EVAL_TIMEOUT" $PY "$PLAN/eval_wm.py" \
    --config-name cube \
    seed="$seed" eval.dataset_name="$H5" ++bf16=true eval.img_size=224 \
    policy="$WM" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    output.filename="${name}.txt" "$@" \
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

# ---------------------------------------------------------------- phase 0: anchors
# Re-run the sequential winners on THIS pod to validate the rebuilt harness+caches
# before spending on AC. Expected (writeup, s42): LIP 88, TD+CEM 82.
log "phase 0: canary anchors (sequential winners on the fresh pod)"
run_eval "anchor_lip_h25_s42" 0 42 25 50 solver=lip "solver.actor_path=$LIP_WIN" &
run_eval "anchor_tdcem_h25_s42" 1 42 25 50 solver=cem "+metric=$TD_WIN" &
wait
a=$(grep "^anchor_lip_h25_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
b=$(grep "^anchor_tdcem_h25_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
[ "$a" != "FAIL" ] && [ -n "$a" ] || die "LIP anchor failed - harness broken, see $LOGS/eval_anchor_lip_h25_s42.log"
[ "$b" != "FAIL" ] && [ -n "$b" ] || die "TD+CEM anchor failed"
log "anchors: LIP ${a} (writeup 88), TD+CEM ${b} (writeup 82)"
awk "BEGIN{exit !(($a-88)^2 <= 144 && ($b-82)^2 <= 144)}" \
  || log "WARNING: anchor >12pts off writeup - fresh-pod drift, interpret AC deltas within-pod only"

# ---------------------------------------------------------------- phase 1: train
ac_train() { # arm gpu extra-args...
  local arm=$1 gpu=$2; shift 2
  local out="$ACT/lip_ac_${arm}.pt" outv="$MET/lip_ac_${arm}_value.pt"
  [ -f "$out" ] && { log "train ${arm}: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm "$WM" \
    --out "$out" --out-value "$outv" \
    --horizon 5 --iters 8 --steps 8000 --actor-lr 3e-4 \
    --expectile 0.03 --n-step 50 "$@" \
    > "$LOGS/train_lip_ac_${arm}.log" 2>&1 \
    || log "train ${arm} FAILED"
}

log "phase 1: training 4 AC arms"
ac_train fresh    0 &
ac_train warm     1 --init-value "$TD_WIN" &
ac_train nofreeze 2 --init-value "$TD_WIN" --freeze-critic-frac 1.0 &
ac_train expand03 3 --init-value "$TD_WIN" --expand-weight 0.3 &
wait
log "AC arms trained"

# ---------------------------------------------------------------- phase 2: select (s42)
ARMS="fresh warm nofreeze expand03"
for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  gpu=0
  for arm in $ARMS; do
    p="$ACT/lip_ac_${arm}.pt"
    [ -f "$p" ] || continue
    run_eval "lipac_${arm}_${cname}_s42" "$gpu" 42 "$coff" "$cbud" \
      solver=lip "solver.actor_path=$p" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

# diagnostic: CEM on each tandem teacher - did co-training move the value itself?
for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  gpu=0
  for arm in $ARMS; do
    m="$MET/lip_ac_${arm}_value.pt"
    [ -f "$m" ] || continue
    run_eval "accem_${arm}_${cname}_s42" "$gpu" 42 "$coff" "$cbud" solver=cem "+metric=$m" &
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
log "AC winner: ${best} (h25+h50 s42 = ${best_score}; sequential LIP anchor-era = 88+74)"

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

log "DONE. Compare lipacwin_${best}_* vs LIP v1 (88/96/78 h25, 74.0 h50 mean) in $RES/summary.csv"
