#!/bin/bash
# IDM-LeWM campaign (2026-07-18): evaluate LIPv4 (winning schedamax-6k, arch v4)
# + latent+CEM on the IDM-augmented LeWM checkpoint (encoder/predictor trained
# with an inverse-dynamics auxiliary head; head stripped for planning).
# /workspace/ckpts/idm_lewm (weights_idm_lewm.pt, step 34480 = ~10 epochs).
#
# Protocol: full lewm-cube h5, 3 draws s42/43/44, h25, budget 50. Only the two
# planners the user asked for (latent+CEM, LIPv4); a TD teacher is trained solely
# as the LIPv4 tandem warm-start (not evaluated as TD+CEM).
# Refs (v2WM): latent+CEM 75.3 | LIPv4-6k 87.6. Compare also to the plain
# 10-epoch retrain (latent+CEM 51.3 | LIPv4 56.9) to isolate the IDM effect.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
CACHES=/workspace/caches
PY=python3
WM=/workspace/ckpts/idm_lewm
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5
CACHE1=$CACHES/cube_idm_fs1.pt
CACHE5=$CACHES/cube_idm_fs5.pt
TD=$MET/idm_dE_t003n50.pt
SUM=$RES/summary_idm.csv
DRV=$LOGS/driver_idm.log
TRAIN_TIMEOUT=28800
EVAL_TIMEOUT=14400

mkdir -p "$LOGS" "$RES" "$MET" "$ACT" "$CACHES" /workspace/swm_home
touch "$SUM"

log(){ echo "[$(date +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }

run_eval(){ # name gpu seed offset budget extra-hydra-args...
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5; shift 5
  local c; c=$(sc "$name"); [ -n "$c" ] && { log "eval ${name}: cached (${c})"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name cube \
    seed="$seed" eval.dataset_name="$H5" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    output.filename="${name}.txt" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED (see log)"; echo "${name},FAIL" >> "$SUM"; return 1; }
  echo "${name},${sr}" >> "$SUM"; log "eval ${name}: ${sr}"
}

log "=== IDM-LeWM campaign start (pid $$) ==="
[ -f "$H5" ] || die "h5 missing"
python3 -c "from stable_worldmodel.wm.utils import load_pretrained; load_pretrained('$WM')" \
  >> "$DRV" 2>&1 || die "IDM-LeWM load failed"
log "IDM-LeWM loads OK"

# -------------------------------------------------- 1. caches (IDM encoder)
if [ ! -f "$CACHES/idm_caches.done" ]; then
  log "phase 1: fs1 cache, 4 shards x 2500 eps (IDM-LeWM encoder)"
  for i in 0 1 2 3; do
    s=$((i * 2500)); e=$(((i + 1) * 2500))
    CUDA_VISIBLE_DEVICES=$i $PY /workspace/scripts/cache_cube_full.py \
      --h5 "$H5" --wm "$WM" --stride 1 --ep-start "$s" --ep-end "$e" \
      --out "$CACHES/idm_fs1_shard$i.pt" > "$LOGS/cache_idm_shard$i.log" 2>&1 &
  done
  wait
  for i in 0 1 2 3; do [ -f "$CACHES/idm_fs1_shard$i.pt" ] || die "shard $i missing"; done
  log "merging shards -> fs1 + fs5"
  $PY /workspace/scripts/merge_caches.py "$CACHE1" "$CACHE5" \
    "$CACHES/idm_fs1_shard0.pt" "$CACHES/idm_fs1_shard1.pt" \
    "$CACHES/idm_fs1_shard2.pt" "$CACHES/idm_fs1_shard3.pt" \
    > "$LOGS/cache_idm_merge.log" 2>&1 || die "merge failed"
  rm -f "$CACHES"/idm_fs1_shard*.pt
  echo done > "$CACHES/idm_caches.done"
fi
log "caches ready"

# -------------------------------------------------- 2. TD teacher (gpu0) || latent+CEM (gpu1-3)
td_train(){
  [ -f "$TD" ] && { log "TD: cached"; return 0; }
  log "TD teacher on IDM fs1 (for LIPv4 warm-start; quasimetric tau0.03 n50 6k)"
  CUDA_VISIBLE_DEVICES=0 timeout $TRAIN_TIMEOUT $PY "$PLAN/train_metric.py" \
    --cache "$CACHE1" --learner td --head quasimetric \
    --expectile 0.03 --n-step 50 --steps 6000 --seed 0 \
    --out "$TD" > "$LOGS/td_idm.log" 2>&1 || die "TD failed"
  log "TD teacher done"
}
td_train &
TD_PID=$!
log "latent+CEM on IDM-LeWM (3 draws, gpu1-3)"
run_eval cemlat_idm_h25_s42 1 42 25 50 policy="$WM" solver=cem &
run_eval cemlat_idm_h25_s43 2 43 25 50 policy="$WM" solver=cem &
run_eval cemlat_idm_h25_s44 3 44 25 50 policy="$WM" solver=cem &
wait $TD_PID || die "TD trainer exited nonzero"
wait
log "latent+CEM done: s42=$(sc cemlat_idm_h25_s42) s43=$(sc cemlat_idm_h25_s43) s44=$(sc cemlat_idm_h25_s44) (v2 refs 80/84/62)"

# -------------------------------------------------- 3. LIPv4 seeds 0-2 (gpu1-3)
lip_line(){ # gpu seed
  local gpu=$1 seed=$2
  local out="$ACT/lip4_idm_s${seed}.pt"
  if [ ! -f "$out" ]; then
    log "train lip4_idm_s${seed}: start gpu${gpu} (schedamax-6k, arch v4)"
    CUDA_VISIBLE_DEVICES=$gpu timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" \
      --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm "$WM" \
      --init-value "$TD" \
      --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5 \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 \
      --seed "$seed" --out "$out" --out-value "$MET/lip4_idm_s${seed}_value.pt" \
      > "$LOGS/train_lip4_idm_s${seed}.log" 2>&1 \
      || { log "train lip4_idm_s${seed}: FAILED"; return 1; }
    local ef; ef=$(grep -E "^step" "$LOGS/train_lip4_idm_s${seed}.log" | tail -1 \
      | grep -oE "E_final [0-9.]+" | awk '{print $2}')
    log "train lip4_idm_s${seed}: done, E_final ${ef:-NA}"
  fi
  local s
  for s in 42 43 44; do
    run_eval "lip4idm_s${seed}_h25_s${s}" "$gpu" "$s" 25 50 \
      policy="$WM" solver=lip "solver.actor_path=$out"
  done
}
log "phase 3: LIPv4 seeds 0/1/2 (gpu1/2/3)"
lip_line 1 0 &
lip_line 2 1 &
lip_line 3 2 &
wait

# -------------------------------------------------- 4. summary
log "=== IDM-LeWM SUMMARY (h25 s42/43/44; refs v2WM latent+CEM 75.3 / LIPv4 87.6; 10ep-retrain 51.3 / 56.9) ==="
mean3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{ if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"} else {printf "%.1f",(a+b+c)/3} }'; }
l42=$(sc cemlat_idm_h25_s42); l43=$(sc cemlat_idm_h25_s43); l44=$(sc cemlat_idm_h25_s44)
log "latent+CEM : ${l42}/${l43}/${l44} -> $(mean3 "$l42" "$l43" "$l44")  (v2 75.3 | 10ep 51.3)"
tot=0; n=0
for sd in 0 1 2; do
  a=$(sc lip4idm_s${sd}_h25_s42); b=$(sc lip4idm_s${sd}_h25_s43); c=$(sc lip4idm_s${sd}_h25_s44)
  m=$(mean3 "$a" "$b" "$c"); log "LIPv4 s${sd}   : ${a}/${b}/${c} -> ${m}"
  [ "$m" != "NA" ] && { tot=$(awk "BEGIN{print $tot+$m}"); n=$((n+1)); }
done
[ "$n" -gt 0 ] && log ">>> LIPv4 3-seed mean = $(awk "BEGIN{printf \"%.1f\",$tot/$n}")  (v2 87.6 | 10ep-retrain 56.9)"
log "IDM CAMPAIGN DONE"
