#!/bin/bash
# TwoRoom LeWM full protocol, autonomous driver (v2: standard + hard cross-wall surfaces).
#   Phase 1: collect play data, build h5 + latent caches
#   Phase 2: latent+CEM baseline + random floor, both surfaces (h25/h50 x seeds 42/43/44)
#   Phase 3: offline-TD sweep (CEM-evaluated on HARD h25 s42), winner confirmed everywhere
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
PLAY=tworoom_play.lance
H5=/workspace/caches/tworoom_play.h5
CACHE1=$CACHES/tworoom_lewm_fs1.pt
CACHE5=$CACHES/tworoom_lewm_fs5.pt

mkdir -p "$LOGS" "$RES" "$MET" "$ACT" "$CACHES"
touch "$RES/summary.csv"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver.log"; }
die() { log "FATAL: $*"; exit 1; }

# run one eval; positional: name gpu seed offset budget surface(std|hard), then extra hydra args
run_eval() {
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5 surface=$6; shift 6
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached ($(grep "^${name}," "$RES/summary.csv" | tail -1 | cut -d, -f2))"
    return 0
  fi
  local hard=()
  [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/eval_wm.py" --config-name tworoom_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" \
    "+video_dir=$RES/videos_${name}" "${hard[@]}" "$@" \
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

# eval a checkpointed planner/value on all 12 cells (2 surfaces x 2 horizons x 3 seeds)
# args: prefix, then extra hydra args appended to every run
eval_all_cells() {
  local prefix=$1; shift
  for surface in hard std; do
    local sfx=""
    [ "$surface" = "hard" ] && sfx="hard"
    run_eval "${prefix}_${surface}_h25_s42" 0 42 25 50 "$surface" "$@" &
    run_eval "${prefix}_${surface}_h25_s43" 1 43 25 50 "$surface" "$@" &
    wait
    run_eval "${prefix}_${surface}_h25_s44" 0 44 25 50 "$surface" "$@" &
    run_eval "${prefix}_${surface}_h50_s42" 1 42 50 100 "$surface" "$@" &
    wait
    run_eval "${prefix}_${surface}_h50_s43" 0 43 50 100 "$surface" "$@" &
    run_eval "${prefix}_${surface}_h50_s44" 1 44 50 100 "$surface" "$@" &
    wait
  done
}

# ---------------------------------------------------------------- phase 0
log "phase 0: smoke test (WM load + tiny eval)"
if ! grep -q "^smoke," "$RES/summary.csv"; then
  CUDA_VISIBLE_DEVICES=0 $PY - <<'EOF' >> "$LOGS/driver.log" 2>&1 || exit 1
import torch, stable_worldmodel as swm
wm = swm.wm.utils.load_pretrained('lewm_tworoom').to('cuda').eval()
x = torch.rand(2, 3, 3, 224, 224, device='cuda')
out = wm.encode({'pixels': x})
assert out['emb'].shape == (2, 3, 192), out['emb'].shape
print('smoke: WM encode OK', flush=True)
EOF
  [ $? -eq 0 ] || die "WM smoke test failed"
  echo "smoke,ok" >> "$RES/summary.csv"
fi

# ---------------------------------------------------------------- phase 1
log "phase 1: data + caches"
if [ ! -d "/workspace/swm_home/datasets/$PLAY" ]; then
  $PY /workspace/code/collect_play.py --episodes 1000 --num-envs 32 --seed 7 \
    --out "/workspace/swm_home/datasets/$PLAY" > "$LOGS/collect.log" 2>&1 \
    || die "data collection failed (see $LOGS/collect.log)"
  log "collected 1000 episodes"
fi
if [ ! -f "$H5" ]; then
  $PY /workspace/code/build_h5.py --dataset "$PLAY" --out "$H5" \
    > "$LOGS/build_h5.log" 2>&1 || die "h5 build failed"
  log "h5 built"
fi
if [ ! -f "$CACHE1" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY "$TRM/cache_latents.py" --wm lewm_tworoom --dataset "$PLAY" \
    --out "$CACHE1" --state-key state --batch-size 512 \
    > "$LOGS/cache_fs1.log" 2>&1 || die "fs1 cache failed"
  log "fs1 cache built"
fi
if [ ! -f "$CACHE5" ]; then
  $PY "$TRM/subsample_cache.py" --in "$CACHE1" --out "$CACHE5" --frameskip 5 \
    > "$LOGS/cache_fs5.log" 2>&1 || die "fs5 cache failed"
  log "fs5 cache built"
fi

# ---------------------------------------------------------------- phase 2
log "phase 2: latent+CEM baselines + random floor (both surfaces)"
# canaries: first std + first hard eval run alone; abort on integration bugs
run_eval "latent_std_h25_s42" 0 42 25 50 std || die "std canary failed: $LOGS/eval_latent_std_h25_s42.log"
run_eval "latent_hard_h25_s42" 0 42 25 50 hard || die "hard canary failed: $LOGS/eval_latent_hard_h25_s42.log"
eval_all_cells "latent"
eval_all_cells "random" policy=random

# ---------------------------------------------------------------- phase 3
log "phase 3: offline TD sweep (train on fs1 cache, CEM-eval on HARD h25 s42)"
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
    gpu=$((1 - gpu))
    if [ "$gpu" -eq 0 ]; then wait; fi
  done
done
wait
log "TD sweep trained"

gpu=0
for tau in 0.03 0.1 0.3; do
  for nstep in 5 25 50; do
    m="$MET/td_e${tau}_n${nstep}.pt"
    [ -f "$m" ] || continue
    run_eval "tdcem_e${tau}_n${nstep}_hard_h25_s42" "$gpu" 42 25 50 hard "+metric=$m" &
    gpu=$((1 - gpu))
    if [ "$gpu" -eq 0 ]; then wait; fi
  done
done
wait

# second selection stage (h25-hard saturates at 100 for every config):
# also score every config on the harder h50-hard draw
gpu=0
for tau in 0.03 0.1 0.3; do
  for nstep in 5 25 50; do
    m="$MET/td_e${tau}_n${nstep}.pt"
    [ -f "$m" ] || continue
    run_eval "tdcem_e${tau}_n${nstep}_hard_h50_s42" "$gpu" 42 50 100 hard "+metric=$m" &
    gpu=$((1 - gpu))
    if [ "$gpu" -eq 0 ]; then wait; fi
  done
done
wait

# winner = max(h25+h50 hard s42); iteration order breaks ties toward the
# prior-work default (low expectile first, long n-step backups first)
best=""; best_score=-1
for tau in 0.03 0.1 0.3; do
  for nstep in 50 25 5; do
    a=$(grep "^tdcem_e${tau}_n${nstep}_hard_h25_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
    b=$(grep "^tdcem_e${tau}_n${nstep}_hard_h50_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
    { [ -z "$a" ] || [ "$a" = "FAIL" ] || [ -z "$b" ] || [ "$b" = "FAIL" ]; } && continue
    s=$(awk "BEGIN{print $a + $b}")
    if awk "BEGIN{exit !($s > $best_score)}"; then best_score=$s; best="e${tau}_n${nstep}"; fi
  done
done
[ -n "$best" ] || die "no TD config produced two valid selection scores"
TD_WIN="$MET/td_${best}.pt"
[ -f "$TD_WIN" ] || die "TD winner metric not found: $TD_WIN"
log "TD winner: td_${best} (h25+h50 hard s42 = ${best_score})"

log "phase 3b: confirm TD winner on all cells"
eval_all_cells "tdwin_$(basename "$TD_WIN" .pt)" "+metric=$TD_WIN"

# ---------------------------------------------------------------- phase 4
log "phase 4: LIP sweep against frozen TD winner"
lip_train() { # iters lr steps gpu
  local iters=$1 lr=$2 steps=$3 gpu=$4
  local out="$ACT/lip_k${iters}_lr${lr}_st${steps}.pt"
  [ -f "$out" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip.py" --cache "$CACHE5" --h5 "$H5" \
    --wm lewm_tworoom --value "$TD_WIN" --horizon 5 --iters "$iters" \
    --steps "$steps" --lr "$lr" --out "$out" \
    > "$LOGS/lip_k${iters}_lr${lr}_st${steps}.log" 2>&1 \
    || log "LIP train k${iters} lr${lr} ${steps} FAILED"
}
lip_train 4 3e-4 4000 0 &
lip_train 4 1e-3 4000 1 &
wait
lip_train 8 3e-4 8000 0 &
lip_train 8 1e-3 4000 1 &
wait
log "LIP sweep trained"

# two-stage selection like TD: hard h25 + hard h50 on the s42 draw;
# iteration order breaks ties toward the cube-canonical recipe (K4 lr3e-4)
gpu=0
for tag in k4_lr3e-4_st4000 k4_lr1e-3_st4000 k8_lr3e-4_st8000 k8_lr1e-3_st4000; do
  a="$ACT/lip_${tag}.pt"
  [ -f "$a" ] || continue
  run_eval "lip_${tag}_hard_h25_s42" "$gpu" 42 25 50 hard solver=lip "solver.actor_path=$a" &
  gpu=$((1 - gpu))
  if [ "$gpu" -eq 0 ]; then wait; fi
done
wait
gpu=0
for tag in k4_lr3e-4_st4000 k4_lr1e-3_st4000 k8_lr3e-4_st8000 k8_lr1e-3_st4000; do
  a="$ACT/lip_${tag}.pt"
  [ -f "$a" ] || continue
  run_eval "lip_${tag}_hard_h50_s42" "$gpu" 42 50 100 hard solver=lip "solver.actor_path=$a" &
  gpu=$((1 - gpu))
  if [ "$gpu" -eq 0 ]; then wait; fi
done
wait

LIP_WIN_TAG=""; lip_best=-1
for tag in k4_lr3e-4_st4000 k4_lr1e-3_st4000 k8_lr3e-4_st8000 k8_lr1e-3_st4000; do
  a=$(grep "^lip_${tag}_hard_h25_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
  b=$(grep "^lip_${tag}_hard_h50_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
  { [ -z "$a" ] || [ "$a" = "FAIL" ] || [ -z "$b" ] || [ "$b" = "FAIL" ]; } && continue
  s=$(awk "BEGIN{print $a + $b}")
  if awk "BEGIN{exit !($s > $lip_best)}"; then lip_best=$s; LIP_WIN_TAG=$tag; fi
done
[ -n "$LIP_WIN_TAG" ] || die "no LIP config produced two valid selection scores"
LIP_WIN="$ACT/lip_${LIP_WIN_TAG}.pt"
[ -f "$LIP_WIN" ] || die "LIP winner not found: $LIP_WIN"
log "LIP winner: lip_${LIP_WIN_TAG} (h25+h50 hard s42 = ${lip_best})"

log "phase 4b: confirm LIP winner on all cells"
eval_all_cells "lipwin_${LIP_WIN_TAG}" solver=lip "solver.actor_path=$LIP_WIN"

log "ALL DONE"
echo "==== FINAL SUMMARY ===="
sort "$RES/summary.csv"
