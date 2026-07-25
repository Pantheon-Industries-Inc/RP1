#!/bin/bash
# HWM x LIPv4 hierarchical eval campaign. Idempotent (summary rows cache evals).
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
PY=python3
WM=/workspace/ckpts/ogbench_cube_single_v2WM
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5
HWMCK=/workspace/hwm/hwm_s25m8.pt
LL0=/workspace/actors/lipabl_ctrl_s0.pt
CHAMP=/workspace/actors/lip_ac90_schedamax.pt
TDV2=/workspace/metrics/cf_dE_t003n50.pt
SUM=$RES/summary_hwm.csv
DRV=$LOGS/driver_hwm.log
EVAL_TIMEOUT=21600
mkdir -p "$LOGS" "$RES"; touch "$SUM"
log(){ echo "[$(date +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
claim(){ while true; do for g in 0 1 2 3 4 5; do
  if mkdir /workspace/.gpu$g 2>/dev/null; then
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "${m:-9999}" -lt 1500 ] && { echo $g; return; }
    rmdir /workspace/.gpu$g 2>/dev/null
  fi; done; sleep 30; done; }
run_eval(){ # name seed offset budget extra-hydra-args...
  local name=$1 seed=$2 offset=$3 budget=$4; shift 4
  local c; c=$(sc "$name"); [ -n "$c" ] && { log "eval ${name}: cached (${c})"; return 0; }
  local g; g=$(claim)
  log "eval ${name}: start gpu${g}"
  CUDA_VISIBLE_DEVICES=$g timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name cube \
    seed="$seed" eval.dataset_name="$H5" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    output.filename="${name}.txt" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  rmdir /workspace/.gpu$g 2>/dev/null
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED (see log)"; echo "${name},FAIL" >> "$SUM"; return 1; }
  echo "${name},${sr}" >> "$SUM"; log "eval ${name}: ${sr}"
}
hlh(){ case $1 in 25|50) echo 2;; 100) echo 4;; 200) echo 8;; *) echo 8;; esac; }

log "=== HWM campaign start (pid $$) ==="
log "waiting for h5"
for i in $(seq 1 720); do [ -f /workspace/datasets/lewm_cube_full/extract.done ] && break; sleep 20; done
[ -f /workspace/datasets/lewm_cube_full/extract.done ] || die "h5 never finished"
python3 - <<'PY' >> "$DRV" 2>&1 || die "h5 verification failed"
import h5py, hdf5plugin, numpy as np
with h5py.File("/workspace/datasets/lewm_cube_full/cube_single_expert.h5","r") as h:
    ln = h["ep_len"][:]
    assert ln.sum() == 2010000 and (ln == 201).all()
    px = h["pixels"]
    assert px.dtype == np.uint8 and px.shape[1:] == (224, 224, 3)
    print("first/last frame readable:", float(px[0].mean()), float(px[-1].mean()))
print("H5 OK")
PY
log "h5 verified"

# ---------------- phase A: anchors (harness integrity on this pod)
if [ ! -f "$RES/anchors.done" ]; then
  run_eval anchor_champ_h25_s43 43 25 50 policy="$WM" solver=lip "solver.actor_path=$CHAMP" &
  run_eval anchor_tdcem_h25_s42 42 25 50 policy="$WM" solver=cem "+metric=$TDV2" &
  wait
  a=$(sc anchor_champ_h25_s43); b=$(sc anchor_tdcem_h25_s42)
  [ -n "$a" ] && [ "$a" != "FAIL" ] || die "champion anchor failed"
  [ -n "$b" ] && [ "$b" != "FAIL" ] || die "TD+CEM anchor failed"
  log "ANCHORS: champion s43 = $a (ref 96.0) | TD+CEM s42 = $b (ref 82.0)"
  echo done > "$RES/anchors.done"
fi

# ---------------- phase B: flat baselines at long horizons
for h in 50 100 200; do
  b=$((2 * h))
  for s in 42 43 44; do
    run_eval "cemlat_h${h}_s${s}" "$s" "$h" "$b" policy="$WM" solver=cem &
    run_eval "lip4flat_h${h}_s${s}" "$s" "$h" "$b" policy="$WM" solver=lip "solver.actor_path=$LL0" &
  done
done
wait
log "phase B done (flat baselines)"

# ---------------- phase C: hcem (HWM-faithful, no learned planner)
for h in 25 50 100 200; do
  b=$((2 * h))
  for s in 42 43 44; do
    run_eval "hcem_h${h}_s${s}" "$s" "$h" "$b" policy="$WM" solver=hlip \
      solver.hl=cem solver.ll=cem "solver.hwm_path=$HWMCK" \
      "solver.hl_horizon=$(hlh $h)" &
  done
done
wait
log "phase C done (hcem)"

# ---------------- phase D: hlip (LIPv4 at both levels; wait for HL actors)
log "waiting for HL actors"
for i in $(seq 1 720); do [ -f /workspace/actors/hl_s25m8.done ] && break; sleep 30; done
[ -f /workspace/actors/lip_hl_s25m8_s0.pt ] || die "HL actors never appeared"
for h in 25 50 100 200; do
  b=$((2 * h))
  for s in 42 43 44; do
    run_eval "hlip_h${h}_s${s}" "$s" "$h" "$b" policy="$WM" solver=hlip \
      solver.hl=lip solver.ll=lip \
      "solver.hl_actor_path=/workspace/actors/lip_hl_s25m8_s0.pt" \
      "solver.ll_actor_path=$LL0" &
  done
done
wait
log "phase D done (hlip)"

# ---------------- summary
log "=== HWM CAMPAIGN SUMMARY (h25 flat refs: latent+CEM 75.3, LIPv4 87.6) ==="
mean3(){ awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN{ if(a==""||b==""||c==""||a=="FAIL"||b=="FAIL"||c=="FAIL"){print "NA"} else {printf "%.1f",(a+b+c)/3} }'; }
for m in cemlat lip4flat hcem hlip; do
  for h in 25 50 100 200; do
    x=$(sc "${m}_h${h}_s42"); y=$(sc "${m}_h${h}_s43"); z=$(sc "${m}_h${h}_s44")
    [ -n "$x$y$z" ] && log "$m h$h: ${x:-–}/${y:-–}/${z:-–} -> $(mean3 "$x" "$y" "$z")"
  done
done
log "HWM CAMPAIGN DONE"
