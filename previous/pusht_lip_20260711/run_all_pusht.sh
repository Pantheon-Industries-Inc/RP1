#!/bin/bash
# PushT LeWM full LIP protocol, autonomous driver (4 GPUs).
#   Phase 1: collect WeakPolicy play data, build h5 + latent caches
#   Phase 2: latent+CEM baseline + random floor (h25/h50 x seeds 42/43/44)
#   Phase 3: offline-TD sweep (CEM-evaluated on h25+h50 s42), winner confirmed everywhere
#   Phase 4: LIP sweep against the frozen TD winner, winner confirmed everywhere
# Results accumulate in /workspace/results/summary.csv (name,success_rate); reruns skip
# any name already recorded (delete rows to force a rerun).
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
TRM=$CODE/scripts/trm
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
CACHES=/workspace/caches
PY=python3
WM=lewm_pusht_official
PLAY=pusht_expert_train.h5
H5=/workspace/swm_home/datasets/pusht_expert_train.h5
CACHE1=$CACHES/pusht_official_fs1.pt
CACHE5=$CACHES/pusht_official_fs5.pt
EVAL_TIMEOUT=14400

mkdir -p "$LOGS" "$RES" "$MET" "$ACT" "$CACHES"
touch "$RES/summary.csv"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver.log"; }
die() { log "FATAL: $*"; exit 1; }

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

# eval a config on all 6 cells (2 horizons x 3 seeds), 4-way parallel
# args: prefix, then extra hydra args appended to every run
eval_all_cells() {
  local prefix=$1; shift
  run_eval "${prefix}_h25_s42" 0 42 25 50 "$@" &
  run_eval "${prefix}_h25_s43" 1 43 25 50 "$@" &
  run_eval "${prefix}_h25_s44" 2 44 25 50 "$@" &
  run_eval "${prefix}_h50_s42" 3 42 50 100 "$@" &
  wait
  run_eval "${prefix}_h50_s43" 0 43 50 100 "$@" &
  run_eval "${prefix}_h50_s44" 1 44 50 100 "$@" &
  wait
}

# ---------------------------------------------------------------- phase 0
log "phase 0: smoke test (WM load + encode)"
if ! grep -q "^smoke," "$RES/summary.csv"; then
  CUDA_VISIBLE_DEVICES=0 $PY - <<'EOF' >> "$LOGS/driver.log" 2>&1 || exit 1
import torch, stable_worldmodel as swm
wm = swm.wm.utils.load_pretrained('lewm_pusht_official').to('cuda').eval()
x = torch.rand(2, 3, 3, 224, 224, device='cuda')
out = wm.encode({'pixels': x})
assert out['emb'].shape == (2, 3, 192), out['emb'].shape
print('smoke: WM encode OK', flush=True)
EOF
  [ $? -eq 0 ] || die "WM smoke test failed"
  echo "smoke,ok" >> "$RES/summary.csv"
fi

# ---------------------------------------------------------------- phase 1
log "phase 1: caches (official dataset, prefix subset for training)"
[ -f "$H5" ] || die "official dataset missing: $H5"
if [ ! -f "$CACHE1" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY "$TRM/cache_latents.py" --wm $WM --dataset "$PLAY" \
    --out "$CACHE1" --state-key state --batch-size 512 --max-rows 600000 \
    > "$LOGS/cache_fs1.log" 2>&1 || die "fs1 cache failed"
  log "fs1 cache built"
fi
if [ ! -f "$CACHE5" ]; then
  $PY "$TRM/subsample_cache.py" --in "$CACHE1" --out "$CACHE5" --frameskip 5 \
    > "$LOGS/cache_fs5.log" 2>&1 || die "fs5 cache failed"
  log "fs5 cache built"
fi

# ---------------------------------------------------------------- phase 2
log "phase 2: latent+CEM baselines + random floor"
# canary: first eval runs alone; abort on integration bugs
run_eval "latent_h25_s42" 0 42 25 50 || die "canary failed: $LOGS/eval_latent_h25_s42.log"
eval_all_cells "latent"
eval_all_cells "random" policy=random

# ---------------------------------------------------------------- phase 3
log "phase 3: offline TD sweep (train on fs1 cache, CEM-eval on h25+h50 s42)"
td_train() { # tau nstep gpu
  local tau=$1 nstep=$2 gpu=$3
  local out="$MET/td_e${tau}_n${nstep}.pt"
  [ -f "$out" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_metric.py" --cache "$CACHE1" \
    --learner td --head quasimetric --expectile "$tau" --n-step "$nstep" \
    --steps 6000 --out "$out" > "$LOGS/td_e${tau}_n${nstep}.log" 2>&1 \
    || log "TD train e${tau} n${nstep} FAILED"
}
gpu=0
for tau in 0.03 0.1 0.3; do
  for nstep in 5 25 50; do
    td_train "$tau" "$nstep" "$gpu" &
    gpu=$(( (gpu + 1) % 4 ))
    if [ "$gpu" -eq 0 ]; then wait; fi
  done
done
wait
log "TD sweep trained"

for horizon in "h25 25 50" "h50 50 100"; do
  set -- $horizon; hname=$1; hoff=$2; hbud=$3
  gpu=0
  for tau in 0.03 0.1 0.3; do
    for nstep in 5 25 50; do
      m="$MET/td_e${tau}_n${nstep}.pt"
      [ -f "$m" ] || continue
      run_eval "tdcem_e${tau}_n${nstep}_${hname}_s42" "$gpu" 42 "$hoff" "$hbud" "+metric=$m" &
      gpu=$(( (gpu + 1) % 4 ))
      if [ "$gpu" -eq 0 ]; then wait; fi
    done
  done
  wait
done

# winner = max(h25+h50 s42); iteration order breaks ties toward the
# prior-work default (low expectile first, long n-step backups first)
best=""; best_score=-1
for tau in 0.03 0.1 0.3; do
  for nstep in 50 25 5; do
    a=$(grep "^tdcem_e${tau}_n${nstep}_h25_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
    b=$(grep "^tdcem_e${tau}_n${nstep}_h50_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
    { [ -z "$a" ] || [ "$a" = "FAIL" ] || [ -z "$b" ] || [ "$b" = "FAIL" ]; } && continue
    s=$(awk "BEGIN{print $a + $b}")
    if awk "BEGIN{exit !($s > $best_score)}"; then best_score=$s; best="e${tau}_n${nstep}"; fi
  done
done
[ -n "$best" ] || die "no TD config produced two valid selection scores"
TD_WIN="$MET/td_${best}.pt"
[ -f "$TD_WIN" ] || die "TD winner metric not found: $TD_WIN"
log "TD winner: td_${best} (h25+h50 s42 = ${best_score})"

log "phase 3b: confirm TD winner on all cells"
eval_all_cells "tdwin_$(basename "$TD_WIN" .pt)" "+metric=$TD_WIN"

# ---------------------------------------------------------------- phase 4
log "phase 4: LIP sweep against frozen TD winner"
lip_train() { # iters lr steps gpu
  local iters=$1 lr=$2 steps=$3 gpu=$4
  local out="$ACT/lip_k${iters}_lr${lr}_st${steps}.pt"
  [ -f "$out" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip.py" --cache "$CACHE5" --h5 "$H5" \
    --wm $WM --value "$TD_WIN" --horizon 5 --iters "$iters" \
    --steps "$steps" --lr "$lr" --out "$out" \
    > "$LOGS/lip_k${iters}_lr${lr}_st${steps}.log" 2>&1 \
    || log "LIP train k${iters} lr${lr} ${steps} FAILED"
}
lip_train 4 3e-4 4000 0 &
lip_train 4 1e-3 4000 1 &
lip_train 8 3e-4 8000 2 &
lip_train 8 1e-3 4000 3 &
wait
log "LIP sweep trained"

LIP_TAGS="k4_lr3e-4_st4000 k4_lr1e-3_st4000 k8_lr3e-4_st8000 k8_lr1e-3_st4000"
for horizon in "h25 25 50" "h50 50 100"; do
  set -- $horizon; hname=$1; hoff=$2; hbud=$3
  gpu=0
  for tag in $LIP_TAGS; do
    a="$ACT/lip_${tag}.pt"
    [ -f "$a" ] || continue
    run_eval "lip_${tag}_${hname}_s42" "$gpu" 42 "$hoff" "$hbud" solver=lip "solver.actor_path=$a" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

# selection like TD: h25 + h50 on the s42 draw; iteration order breaks ties
# toward the cube-canonical recipe (K4 lr3e-4)
LIP_WIN_TAG=""; lip_best=-1
for tag in $LIP_TAGS; do
  a=$(grep "^lip_${tag}_h25_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
  b=$(grep "^lip_${tag}_h50_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
  { [ -z "$a" ] || [ "$a" = "FAIL" ] || [ -z "$b" ] || [ "$b" = "FAIL" ]; } && continue
  s=$(awk "BEGIN{print $a + $b}")
  if awk "BEGIN{exit !($s > $lip_best)}"; then lip_best=$s; LIP_WIN_TAG=$tag; fi
done
[ -n "$LIP_WIN_TAG" ] || die "no LIP config produced two valid selection scores"
LIP_WIN="$ACT/lip_${LIP_WIN_TAG}.pt"
[ -f "$LIP_WIN" ] || die "LIP winner not found: $LIP_WIN"
log "LIP winner: lip_${LIP_WIN_TAG} (h25+h50 s42 = ${lip_best})"

log "phase 4b: confirm LIP winner on all cells"
eval_all_cells "lipwin_${LIP_WIN_TAG}" solver=lip "solver.actor_path=$LIP_WIN"

log "ALL DONE"
echo "==== FINAL SUMMARY ===="
sort "$RES/summary.csv"
