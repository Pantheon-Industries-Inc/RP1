#!/bin/bash
# PushT min0 campaign: velocity+imagination TD teacher -> tandem min0 actor.
# Design: pusht_min0_20260714/DESIGN.md. Pod b (4x H100).
#
#   P0   assets (WM convert, fs1/fs5 caches) + baseline repro
#        (old teacher CEM h25 s42 ~ 92, LIP2+R8 h25 s42 ~ 88)
#   PA   teacher sweep: velimag grid tau{0.01,0.03,0.1} x n{1,3,5} p_imag 0.5
#        + imag-only (no-vel) + hist-ablate (p_imag 0) at the old corner.
#        CEM-select on h25+h50 s42 combined (old teacher baseline = 154).
#   PB   tandem arms warm-started from PA winner (schedamax schedules, K8 8k):
#        m0pi (min0 + p-imag 0.5) / m0s (min0) / m0f (min0, frozen teacher) /
#        full (champion input). LIP-eval h25+h50 s42; R8 probe top arms;
#        seed-1 retrain of the winner (single-seed guard).
#   PC   winner -> 3-draw h25+h50 (one-shot + R8) + teacher CEM 3-draw.
#
# Idempotent via /workspace/results/summary.csv (name,success) and file
# existence; delete rows/files to force re-runs. STOP_AFTER=pa|pb to gate.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
TRM=$CODE/scripts/trm
MIN0=/workspace/min0
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
TD_OLD=$MET/td2_e0.01_n1.pt
LIP2=$ACT/lip2_k8_lr3e-4_st8000.pt
EVAL_TIMEOUT=14400
STOP_AFTER=${STOP_AFTER:-}

mkdir -p "$LOGS" "$RES" "$MET" "$ACT" "$CACHES"
touch "$RES/summary.csv"

log() { echo "[$(date +%H:%M:%S)] [min0] $*" | tee -a "$LOGS/min0_driver.log"; }
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
    > "$LOGS/eval_${name}.log" 2>&1
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
combined() { # two row names -> sum (empty if either missing/FAIL)
  local x y; x=$(sr_of "$1"); y=$(sr_of "$2")
  { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && return 1
  awk "BEGIN{print $x + $y}"
}

# ---------------------------------------------------------------- P0: assets
log "P0: assets"
[ -f "$H5" ] || die "dataset missing: $H5 (download/decompress still running?)"
if [ ! -d "$STABLEWM_HOME/checkpoints/lewm_pusht_official" ]; then
  log "converting official ckpt"
  mkdir -p "$STABLEWM_HOME/checkpoints/models--quentinll--lewm-pusht"
  cp /workspace/hf_pusht/weights.pt /workspace/hf_pusht/config.json \
     "$STABLEWM_HOME/checkpoints/models--quentinll--lewm-pusht/"
  CUDA_VISIBLE_DEVICES=0 $PY "$MIN0/convert_official_ckpt.py" \
    > "$LOGS/convert_ckpt.log" 2>&1 || die "ckpt conversion failed"
fi
if ! grep -q "^smoke," "$RES/summary.csv"; then
  CUDA_VISIBLE_DEVICES=0 $PY - <<'EOF' >> "$LOGS/min0_driver.log" 2>&1 || die "WM smoke failed"
import torch, stable_worldmodel as swm
wm = swm.wm.utils.load_pretrained('lewm_pusht_official').to('cuda').eval()
x = torch.rand(2, 3, 3, 224, 224, device='cuda')
out = wm.encode({'pixels': x})
assert out['emb'].shape == (2, 3, 192), out['emb'].shape
print('smoke: WM encode OK', flush=True)
EOF
  echo "smoke,ok" >> "$RES/summary.csv"
fi
if [ ! -f "$CACHE1" ]; then
  log "building fs1 cache (600k rows)"
  CUDA_VISIBLE_DEVICES=0 $PY "$TRM/cache_latents.py" --wm $WM \
    --dataset pusht_expert_train.h5 --out "$CACHE1" --state-key state \
    --batch-size 512 --max-rows 600000 > "$LOGS/cache_fs1.log" 2>&1 \
    || die "fs1 cache failed"
fi
[ -f "$CACHE5" ] || { $PY "$TRM/subsample_cache.py" --in "$CACHE1" --out "$CACHE5" \
  --frameskip 5 > "$LOGS/cache_fs5.log" 2>&1 || die "fs5 cache failed"; }
[ -f "$TD_OLD" ] || die "old teacher missing: $TD_OLD (scp it up)"
[ -f "$LIP2" ] || die "old LIP2 actor missing: $LIP2 (scp it up)"

log "P0b: baseline repro (draw s42: teacher CEM ~92, LIP2+R8 ~88)"
run_eval "repro_tdcem_h25_s42" 0 42 25 50 "+metric=$TD_OLD" &
run_eval "repro_tdcem_h50_s42" 1 42 50 100 "+metric=$TD_OLD" &
run_eval "repro_lip2R8_h25_s42" 2 42 25 50 solver=lip "solver.actor_path=$LIP2" \
  solver.restarts=8 solver.restart_noise=0.5 &
wait
log "P0 done"
[ "$STOP_AFTER" = "p0" ] && { log "STOP_AFTER=p0"; exit 0; }

# ---------------------------------------------------------------- PA: teacher sweep
vi_train() { # tau n gpu extra...
  local tau=$1 n=$2 gpu=$3 out=$4; shift 4
  [ -f "$out" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu $PY "$MIN0/train_metric_velimag.py" \
    --cache "$CACHE1" --h5 "$H5" --wm $WM --expectile "$tau" --n-step "$n" \
    --steps 6000 --out "$out" "$@" \
    > "$LOGS/train_$(basename "$out" .pt).log" 2>&1 \
    || log "train $(basename "$out") FAILED"
}

log "PA: training delta + win3 grids (identical sweeps) + win5 probes + ablations"
G=0
launch() { # tau n out extra...  (round-robin over 4 GPUs, barrier each wave)
  local tau=$1 n=$2 out=$3; shift 3
  vi_train "$tau" "$n" "$G" "$out" "$@" &
  G=$(( (G + 1) % 4 )); [ "$G" -eq 0 ] && wait
}
for tau in 0.01 0.03 0.1; do
  for n in 1 3 5; do
    launch "$tau" "$n" "$MET/vi_e${tau}_n${n}.pt"
    launch "$tau" "$n" "$MET/wn3_e${tau}_n${n}.pt" --win-frames 3
  done
done
# long-backup probes: if the state de-aliases velocity, n>5 should stop hurting
launch 0.03 15 "$MET/vi_e0.03_n15.pt"
launch 0.1  15 "$MET/vi_e0.1_n15.pt"
launch 0.03 15 "$MET/wn3_e0.03_n15.pt" --win-frames 3
launch 0.1  15 "$MET/wn3_e0.1_n15.pt"  --win-frames 3
# win5 probes (full sweep only if it beats win3)
launch 0.01 1 "$MET/wn5_e0.01_n1.pt" --win-frames 5
launch 0.03 3 "$MET/wn5_e0.03_n3.pt" --win-frames 5
# ablation pair: imagination without velocity / velocity without imagination
launch 0.01 1 "$MET/im_e0.01_n1.pt" --no-vel
launch 0.01 1 "$MET/vh_e0.01_n1.pt" --p-imag 0
wait
log "PA trained"

PA_METRICS=""
for m in vi wn3; do
  for tau in 0.01 0.03 0.1; do for n in 1 3 5; do
    PA_METRICS="$PA_METRICS ${m}_e${tau}_n${n}"; done; done
  PA_METRICS="$PA_METRICS ${m}_e0.03_n15 ${m}_e0.1_n15"
done
PA_METRICS="$PA_METRICS wn5_e0.01_n1 wn5_e0.03_n3 im_e0.01_n1 vh_e0.01_n1"

for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  gpu=0
  for m in $PA_METRICS; do
    [ -f "$MET/${m}.pt" ] || continue
    run_eval "cem_${m}_${cname}_s42" "$gpu" 42 "$coff" "$cbud" "+metric=$MET/${m}.pt" &
    gpu=$(( (gpu + 1) % 4 )); [ "$gpu" -eq 0 ] && wait
  done
  wait
done

# select PA winner + runner-up (old teacher competes via its repro rows)
meta_of() { # metric name -> "path tau n"
  if [ "$1" = "OLD" ]; then echo "$TD_OLD 0.01 1"; return; fi
  local tau n
  tau=$(echo "$1" | sed -E 's/.*_e([0-9.]+)_n[0-9]+$/\1/')
  n=$(echo "$1" | sed -E 's/.*_n([0-9]+)$/\1/')
  echo "$MET/$1.pt $tau $n"
}
best=""; best_score=""; second=""; second_score=""
consider() { # name score
  if [ -z "$best_score" ] || awk "BEGIN{exit !($2 > $best_score)}"; then
    second=$best; second_score=$best_score; best=$1; best_score=$2
  elif [ -z "$second_score" ] || awk "BEGIN{exit !($2 > $second_score)}"; then
    second=$1; second_score=$2
  fi
}
old_c=$(combined "repro_tdcem_h25_s42" "repro_tdcem_h50_s42") && consider "OLD" "$old_c"
for m in $PA_METRICS; do
  c=$(combined "cem_${m}_h25_s42" "cem_${m}_h50_s42") || continue
  log "PA ${m}: combined ${c}"
  consider "$m" "$c"
done
[ -n "$best" ] || die "PA: no metric produced two valid cells"
set -- $(meta_of "$best"); PA_WIN=$1; PA_TAU=$2; PA_N=$3
echo "${best},${best_score},${PA_WIN},${PA_TAU},${PA_N}" > "$RES/pa_winner.txt"
log "PA winner: ${best} (combined ${best_score}; old teacher ${old_c:-n/a}) tau=${PA_TAU} n=${PA_N}"
log "PA runner-up: ${second:-none} (${second_score:-n/a})"
[ "$STOP_AFTER" = "pa" ] && { log "STOP_AFTER=pa"; exit 0; }

# ---------------------------------------------------------------- PB: tandem arms
ac_train() { # arm gpu seed init tau n extra...
  local arm=$1 gpu=$2 seed=$3 init=$4 tau=$5 n=$6; shift 6
  local out="$ACT/lipm0_${arm}.pt" outv="$MET/lipm0_${arm}_value.pt"
  [ -f "$out" ] && { log "train ${arm}: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm $WM \
    --out "$out" --out-value "$outv" \
    --horizon 5 --iters 8 --steps 8000 \
    --init-value "$init" --n-step "$n" \
    --expectile 0.1 --expectile-final "$tau" \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --amax 3.5 --seed "$seed" "$@" \
    > "$LOGS/train_lipm0_${arm}.log" 2>&1 \
    || log "train ${arm} FAILED"
}

log "PB: tandem min0 arms (warm from ${best})"
ac_train m0pi 0 0 "$PA_WIN" "$PA_TAU" "$PA_N" --drop-z0 --drop-zg --p-imag 0.5 &
ac_train m0s  1 0 "$PA_WIN" "$PA_TAU" "$PA_N" --drop-z0 --drop-zg &
ac_train m0f  2 0 "$PA_WIN" "$PA_TAU" "$PA_N" --drop-z0 --drop-zg --freeze-critic-frac 0.0 &
ac_train full 3 0 "$PA_WIN" "$PA_TAU" "$PA_N" &
wait
ARMS="m0pi m0s m0f full"
# CEM-rank != gradient-rank hedge: runner-up teacher gets its own min0 arm
# when it is within selection noise of the winner
if [ -n "$second" ] && awk "BEGIN{exit !($second_score > $best_score - 8)}"; then
  set -- $(meta_of "$second"); S_WIN=$1; S_TAU=$2; S_N=$3
  log "PB: runner-up ${second} within noise -> m0pi2 arm (tau=${S_TAU} n=${S_N})"
  ac_train m0pi2 0 0 "$S_WIN" "$S_TAU" "$S_N" --drop-z0 --drop-zg --p-imag 0.5
  ARMS="$ARMS m0pi2"
fi
log "PB trained"
for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  gpu=0
  for arm in $ARMS; do
    [ -f "$ACT/lipm0_${arm}.pt" ] || continue
    run_eval "m0_${arm}_${cname}_s42" "$gpu" 42 "$coff" "$cbud" \
      solver=lip "solver.actor_path=$ACT/lipm0_${arm}.pt" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

# rank arms, R8-probe the top two (PushT multimodality vs cube's no-restart lesson)
rank=$(for arm in $ARMS; do
  c=$(combined "m0_${arm}_h25_s42" "m0_${arm}_h50_s42") || continue
  echo "$c $arm"
done | sort -rn)
log "PB ranking:"; echo "$rank" | while read -r l; do log "  $l"; done
top2=$(echo "$rank" | head -2 | awk '{print $2}')
gpu=0
for arm in $top2; do
  run_eval "m0_${arm}_R8_h25_s42" "$gpu" 42 25 50 solver=lip \
    "solver.actor_path=$ACT/lipm0_${arm}.pt" solver.restarts=8 solver.restart_noise=0.5 &
  gpu=$(( gpu + 1 ))
  run_eval "m0_${arm}_R8_h50_s42" "$gpu" 42 50 100 solver=lip \
    "solver.actor_path=$ACT/lipm0_${arm}.pt" solver.restarts=8 solver.restart_noise=0.5 &
  gpu=$(( gpu + 1 ))
done
wait

# arm winner = best of {one-shot, R8} combined; seed-1 guard retrain
PB_ARM=""; PB_MODE="plain"; PB_SCORE=""
for arm in $ARMS; do
  for mode in plain R8; do
    if [ "$mode" = "plain" ]; then
      c=$(combined "m0_${arm}_h25_s42" "m0_${arm}_h50_s42") || continue
    else
      c=$(combined "m0_${arm}_R8_h25_s42" "m0_${arm}_R8_h50_s42") || continue
    fi
    if [ -z "$PB_SCORE" ] || awk "BEGIN{exit !($c > $PB_SCORE)}"; then
      PB_SCORE=$c; PB_ARM=$arm; PB_MODE=$mode
    fi
  done
done
[ -n "$PB_ARM" ] || die "PB: no arm produced valid scores"
log "PB winner: ${PB_ARM} (${PB_MODE}, combined ${PB_SCORE}); baselines: teacher154 lip2R8 142"

# single-seed guard: retrain winner arm at seed 1, evaluate in winner mode
warm_flags=""; W_INIT=$PA_WIN; W_TAU=$PA_TAU; W_N=$PA_N
case "$PB_ARM" in
  m0pi)  warm_flags="--drop-z0 --drop-zg --p-imag 0.5" ;;
  m0pi2) warm_flags="--drop-z0 --drop-zg --p-imag 0.5"
         W_INIT=$S_WIN; W_TAU=$S_TAU; W_N=$S_N ;;
  m0s)   warm_flags="--drop-z0 --drop-zg" ;;
  m0f)   warm_flags="--drop-z0 --drop-zg --freeze-critic-frac 0.0" ;;
  full)  warm_flags="" ;;
esac
ac_train "${PB_ARM}_s1" 0 1 "$W_INIT" "$W_TAU" "$W_N" $warm_flags
R8FLAGS=""
[ "$PB_MODE" = "R8" ] && R8FLAGS="solver.restarts=8 solver.restart_noise=0.5"
run_eval "m0_${PB_ARM}_s1_h25_s42" 0 42 25 50 solver=lip \
  "solver.actor_path=$ACT/lipm0_${PB_ARM}_s1.pt" $R8FLAGS &
run_eval "m0_${PB_ARM}_s1_h50_s42" 1 42 50 100 solver=lip \
  "solver.actor_path=$ACT/lipm0_${PB_ARM}_s1.pt" $R8FLAGS &
wait
c1=$(combined "m0_${PB_ARM}_s1_h25_s42" "m0_${PB_ARM}_s1_h50_s42" || echo "")
PB_CKPT=$ACT/lipm0_${PB_ARM}.pt
if [ -n "$c1" ] && awk "BEGIN{exit !($c1 > $PB_SCORE)}"; then
  PB_CKPT=$ACT/lipm0_${PB_ARM}_s1.pt
  log "seed-1 beats seed-0 (${c1} vs ${PB_SCORE}); confirming s1"
fi
echo "${PB_ARM},${PB_MODE},${PB_SCORE},${PB_CKPT},seed1=${c1:-n/a}" > "$RES/pb_winner.txt"
[ "$STOP_AFTER" = "pb" ] && { log "STOP_AFTER=pb"; exit 0; }

# ---------------------------------------------------------------- PC: headline
log "PC: 3-draw confirm (winner ${PB_ARM} ${PB_MODE}) + teacher CEM 3-draw"
for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  gpu=0
  for seed in 42 43 44; do
    run_eval "final_${PB_ARM}_${cname}_s${seed}" "$gpu" "$seed" "$coff" "$cbud" \
      solver=lip "solver.actor_path=$PB_CKPT" $R8FLAGS &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done
gpu=0
for cell in "h25 25 50" "h50 50 100"; do
  set -- $cell; cname=$1; coff=$2; cbud=$3
  for seed in 43 44; do
    run_eval "finalcem_${cname}_s${seed}" "$gpu" "$seed" "$coff" "$cbud" \
      "+metric=$PA_WIN" &
    gpu=$(( (gpu + 1) % 4 )); [ "$gpu" -eq 0 ] && wait
  done
done
wait

log "PC done. Headline rows:"
grep -E "^final" "$RES/summary.csv" | while read -r l; do log "  $l"; done
log "ALL DONE"
