#!/bin/bash
# Dyna stage-1 driver: mix_plain_roll e22 ladder — export -> caches -> TD ->
# LIPv4 x3 -> evals (CEM / TD+CEM / LIP) + A/B imagined-vs-reached probe.
# Blocks until WM training completes (TRAIN_DDP_EXIT=0 + weights_epoch_22.pt),
# so it can be launched while training runs. Idempotent: stages cached by
# artifact file / summary row — re-run freely.
#
# References to beat (lost-volume mix_plain, NO rollout loss, same protocol):
#   latent+CEM 68.7 (66/76/64) | TD+CEM 71.3 (72/80/62) | LIPv4 3-actor 65.1
# v2WM expert refs: latent+CEM 75.3 | TD+CEM 79.3 | LIPv4-6k 87.6 | div ~20.9
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/runs/mix_plain/wm_home
export TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
TRM=$CODE/scripts/trm
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
CACHES=/workspace/caches
MODELS=/workspace/models
PY=python3
WM=$MODELS/mix_plain_roll_e22
CKD=/workspace/runs/mix_plain/wm_home/checkpoints/mix_plain_roll
MIX=/workspace/datasets/mixture/mix.lance
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/mixture_actions.h5
FS1=$CACHES/roll_mixcache_fs1.pt
FS5=$CACHES/roll_mixcache_fs5.pt
TD=$MET/roll_mixcache_TD.pt
SUM=$RES/summary_dyna1.csv
DRV=$LOGS/driver_dyna1.log
PROBE=/workspace/probe_roll_base
TRAIN_TIMEOUT=28800
EVAL_TIMEOUT=14400

mkdir -p "$LOGS" "$RES" "$MET" "$ACT" "$CACHES" "$MODELS" "$PROBE"
touch "$SUM"

log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }

run_eval(){ # name gpu seed extra-hydra-args...
  local name=$1 gpu=$2 seed=$3
  shift 3
  local c
  c=$(sc "$name"); [ -n "$c" ] && { log "eval ${name}: cached (${c})"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name cube seed="$seed" eval.dataset_name="$EXPERT" ++bf16=true \
    eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
    output.filename="${name}.txt" "$@" > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED (see log)"; echo "${name},FAIL" >> "$SUM"; return 1; }
  echo "${name},${sr}" >> "$SUM"; log "eval ${name}: ${sr}"
}

# ------------------------------------------------------------- 0. wait for e22
log "=== dyna stage-1 driver start (pid $$) ==="
for i in $(seq 1 2000); do
  grep -q "TRAIN_DDP_EXIT=0" "$LOGS/train_ddp.log" 2>/dev/null && [ -f "$CKD/weights_epoch_22.pt" ] && break
  grep -qE "TRAIN_DDP_EXIT=[1-9]" "$LOGS/train_ddp.log" 2>/dev/null && die "training exited nonzero"
  sleep 60
done
[ -f "$CKD/weights_epoch_22.pt" ] || die "e22 never appeared (waited ~33h)"
log "training complete, e22 present"

$PY -c "import imageio_ffmpeg" 2>/dev/null || pip install -q imageio imageio-ffmpeg >> "$DRV" 2>&1
log "env deps ok"

# ------------------------------------------------- 1. export model dir (e22)
if [ ! -f "$WM/config.json" ]; then
  mkdir -p "$WM"
  cp "$CKD/weights_epoch_22.pt" "$WM/"
  $PY - "$CKD" "$WM" <<'PYEOF' || die "arch-config export failed"
import json, sys
ckd, wm = sys.argv[1], sys.argv[2]
c = json.load(open(f"{ckd}/config.json"))
arch = c["model"] if "model" in c else c   # ops gotcha: load_pretrained needs the ARCH subtree
json.dump(arch, open(f"{wm}/config.json", "w"), indent=1)
print("arch config keys:", list(arch.keys()))
PYEOF
fi
$PY -c "
from stable_worldmodel.wm.utils import load_pretrained
m = load_pretrained('$WM')
print('WM OK, params', sum(p.numel() for p in m.parameters()))" >> "$DRV" 2>&1 || die "WM load smoke failed"
log "model exported + load-smoked: $WM"

# --------------------------------------------------------- 2. caches (gpu 0)
if [ ! -f "$FS5" ]; then
  if [ ! -f "$FS1" ]; then
    log "caching fs1 (mixture stride-1, e22 encoder, gpu0)"
    CUDA_VISIBLE_DEVICES=0 timeout $TRAIN_TIMEOUT $PY "$TRM/cache_latents.py" \
      --wm "$WM" --dataset "$MIX" --out "$FS1" --state-key privileged/block_0_pos \
      > "$LOGS/cache_fs1.log" 2>&1 || die "fs1 cache failed (see cache_fs1.log)"
  fi
  log "subsampling fs5"
  $PY "$TRM/subsample_cache.py" --in "$FS1" --out "$FS5" --frameskip 5 \
    > "$LOGS/cache_fs5.log" 2>&1 || die "fs5 subsample failed"
fi
log "caches ready: $(du -sh $FS1 | cut -f1) fs1, $(du -sh $FS5 | cut -f1) fs5"

# ------------------------------- 3. TD teacher (gpu0) || latent+CEM (gpu1-3)
td_train(){
  [ -f "$TD" ] && { log "TD: cached"; return 0; }
  log "TD teacher: td/quasimetric tau.03 n50 6k on mixture fs1 (gpu0)"
  CUDA_VISIBLE_DEVICES=0 timeout $TRAIN_TIMEOUT $PY "$PLAN/train_metric.py" \
    --cache "$FS1" --learner td --head quasimetric \
    --expectile 0.03 --n-step 50 --steps 6000 --seed 0 \
    --out "$TD" > "$LOGS/td_roll.log" 2>&1 || die "TD training failed"
  log "TD teacher done"
}
td_train &
TD_PID=$!
run_eval cem_roll_s42 1 42 policy="$WM" solver=cem &
run_eval cem_roll_s43 2 43 policy="$WM" solver=cem &
run_eval cem_roll_s44 3 44 policy="$WM" solver=cem &
wait $TD_PID || die "TD trainer exited nonzero"
wait
log "latent+CEM: $(sc cem_roll_s42)/$(sc cem_roll_s43)/$(sc cem_roll_s44)  (no-roll ref 66/76/64 = 68.7)"

# --------------------------- 4. LIPv4 seeds 0-2 (gpu1-3) || TD+CEM (gpu0)
lip_line(){ # gpu seed
  local gpu=$1
  local seed=$2
  local out="$ACT/lip4_roll_s${seed}.pt"
  if [ ! -f "$out" ]; then
    log "lip4_roll_s${seed}: train start (gpu${gpu}, schedamax-6k v4, expert action-pin)"
    CUDA_VISIBLE_DEVICES=$gpu timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" \
      --cache "$FS5" --cache-td "$FS1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
      --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5 \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --action-stats-pin expert \
      --seed "$seed" --out "$out" --out-value "$MET/lip4_roll_s${seed}_value.pt" \
      > "$LOGS/train_lip4_roll_s${seed}.log" 2>&1 \
      || { log "lip4_roll_s${seed}: TRAIN FAILED"; return 1; }
    local ef
    ef=$(grep -E "^step" "$LOGS/train_lip4_roll_s${seed}.log" | tail -1 | grep -oE "E_final [0-9.]+" | awk '{print $2}')
    log "lip4_roll_s${seed}: done, E_final ${ef:-NA} (no-roll ref ~1.3-2.5)"
  fi
  local s
  for s in 42 43 44; do
    if [ "$seed" = "0" ] && [ "$s" = "42" ]; then
      LIP_PROBE_DIR=$PROBE run_eval "lip4roll_s${seed}_e${s}" "$gpu" "$s" policy="$WM" solver=lip "solver.actor_path=$out"
    else
      run_eval "lip4roll_s${seed}_e${s}" "$gpu" "$s" policy="$WM" solver=lip "solver.actor_path=$out"
    fi
  done
}
log "phase 4: LIP seeds 0/1/2 on gpu1/2/3 + TD+CEM on gpu0 (probe on s0/e42 -> $PROBE)"
lip_line 1 0 &
lip_line 2 1 &
lip_line 3 2 &
( run_eval cemtd_roll_s42 0 42 policy="$WM" solver=cem "+metric=$TD"
  run_eval cemtd_roll_s43 0 43 policy="$WM" solver=cem "+metric=$TD"
  run_eval cemtd_roll_s44 0 44 policy="$WM" solver=cem "+metric=$TD" ) &
wait

# ----------------------------------------------------------------- 5. summary
mean3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{ if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"} else {printf "%.1f", (a+b+c)/3} }'; }
log "=== STAGE-1 SUMMARY (mix_plain_roll e22; no-roll refs CEM 68.7 / TD+CEM 71.3 / LIP 65.1; v2 LIP 87.6) ==="
l42=$(sc cem_roll_s42);  l43=$(sc cem_roll_s43);  l44=$(sc cem_roll_s44)
t42=$(sc cemtd_roll_s42); t43=$(sc cemtd_roll_s43); t44=$(sc cemtd_roll_s44)
log "latent+CEM : ${l42}/${l43}/${l44} -> $(mean3 "$l42" "$l43" "$l44")   (no-roll 68.7)"
log "TD+CEM     : ${t42}/${t43}/${t44} -> $(mean3 "$t42" "$t43" "$t44")   (no-roll 71.3)"
tot=0; n=0
for sd in 0 1 2; do
  a=$(sc lip4roll_s${sd}_e42); b=$(sc lip4roll_s${sd}_e43); c=$(sc lip4roll_s${sd}_e44)
  m=$(mean3 "$a" "$b" "$c")
  log "LIPv4 s${sd}   : ${a}/${b}/${c} -> ${m}"
  [ "$m" != "NA" ] && { tot=$(awk "BEGIN{print $tot + $m}"); n=$((n+1)); }
done
[ "$n" -gt 0 ] && log "LIPv4 3-seed mean = $(awk "BEGIN{printf \"%.1f\", $tot/$n}")   (no-roll 65.1, v2 87.6)"
log "A/B probe dumps (divergence baseline ~20.9 on v2): $PROBE"
log "STAGE1_DONE"
