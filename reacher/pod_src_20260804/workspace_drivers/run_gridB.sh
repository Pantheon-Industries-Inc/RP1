#!/bin/bash
# GRID SEARCH over the four user-specified LIP factors (2026-07-30), completing
# wave A's OFAT cross into a full factorial.
#
#   amax        {1.8, 2.0, 2.2, 2.6}
#   mean-weight {0.1, 0.3, 0.5}
#   actor-lr    {1e-4, 3e-4, 1e-3}   (final = lr/10, as in the canonical recipe)
#   steps        8000 for the screen; 16000 applied to the winners in stage 2
#   = 4 x 3 x 3 = 36 cells
#
# Wave A already covers 8 of the 36 cells AT 3 SEEDS (centre + the OFAT arms);
# those are symlinked into the grid naming rather than retrained. The remaining
# 28 cells run at seeds {0,1}; stage 2 then takes the top cells to a 3rd seed
# and a steps-16000 twin, so no winner is ever declared on thin evidence.
#
# Setting throughout: h25, HELD@0.05, window3 critic, pad-context, 1-frame
# policy conditioning, plain LIP deploy. Bar to beat: Latent+CEM-window 44.7
# (re-measured on the rebuilt artifacts); a real win is ~51+.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
W3=/workspace/metrics/window3_lejepa.pt
SUM=/workspace/results/summary_grid_lejepa.csv; touch "$SUM"
L=/workspace/logs/grid; mkdir -p "$L"
NGPU=6
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][grid] $*"; }

tag(){ echo "a${1/./}m${2/./}l${3}"; }   # e.g. a22m01l3e-4

BASE="--cache /workspace/caches/canon_lejepa_fs5.pt \
 --cache-td /workspace/caches/canon_lejepa_fs1.pt \
 --h5 /workspace/reacher_slim.h5 --wm $WM --pad-context --init-value $W3 \
 --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
 --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4"

# ---------------------------------------------------------------- adopt wave A
log "adopting wave-A cells into the grid naming (no retraining)"
adopt(){ # waveA-tag amax mw lr
  local wa=$1 t; t=$(tag "$2" "$3" "$4")
  for s in 0 1 2; do
    local src=/workspace/actors/lip4_wa_${wa}_s${s}.pt
    local dst=/workspace/actors/lip4_gr_${t}_s${s}.pt
    [ -f "$src" ] && [ ! -e "$dst" ] && ln -s "$src" "$dst" && log "  adopted ${wa} s${s} -> ${t}"
  done
}
wait_for_waveA(){
  while pgrep -f "[r]un_waveA.sh" >/dev/null; do sleep 120; done
}
log "adopting immediately (wave A trainings are complete)"
true
adopt centre  2.2 0.1 3e-4
adopt amax18  1.8 0.1 3e-4
adopt amax20  2.0 0.1 3e-4
adopt amax26  2.6 0.1 3e-4
adopt mw03    2.2 0.3 3e-4
adopt mw05    2.2 0.5 3e-4
adopt alr1e4  2.2 0.1 1e-4
adopt alr1e3  2.2 0.1 1e-3

# ---------------------------------------------------------------- train the rest
train_cell(){ # amax mw lr seed gpu
  local amax=$1 mw=$2 lr=$3 seed=$4 gpu=$5
  local t; t=$(tag "$amax" "$mw" "$lr")
  local A=/workspace/actors/lip4_gr_${t}_s${seed}.pt
  [ -e "$A" ] && return 0
  local lrf
  case "$lr" in
    1e-4) lrf=1e-5 ;;
    3e-4) lrf=3e-5 ;;
    1e-3) lrf=1e-4 ;;
    *) lrf=$(awk "BEGIN{printf \"%.0e\", $lr/10}") ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$PLAN/train_lip_ac.py" \
    $BASE --amax "$amax" --mean-weight "$mw" \
    --actor-lr "$lr" --actor-lr-final "$lrf" --steps 8000 --seed "$seed" \
    --out "$A" --out-value "/workspace/metrics/lip4_gr_${t}_s${seed}_value.pt" \
    > "$L/train_${t}_s${seed}.log" 2>&1 \
    && log "train ${t} s${seed} DONE" || log "train ${t} s${seed} FAILED"
}

log "grid stage 1: 36 cells, seeds {0,1} for the 28 not covered by wave A"
i=0
for amax in 1.8 2.0 2.2 2.6; do
 for mw in 0.1 0.3 0.5; do
  for lr in 1e-4 3e-4 1e-3; do
   for seed in 0 1; do
     train_cell "$amax" "$mw" "$lr" "$seed" $((i % NGPU)) &
     i=$((i + 1))
     [ $((i % 18)) -eq 0 ] && wait
   done
  done
 done
done
wait
log "grid trainings drained"

# ---------------------------------------------------------------- evals
ev(){ local gpu=$1 nm=$2 seed=$3 actor=$4
  grep -q "^${nm}," "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=lip solver.actor_path="$actor" solver.rollout_compat=false \
    solver.batch_size=10 output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h l
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  l=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  echo "${nm},held=${h:-FAIL},latched=${l:-FAIL}" >> "$SUM"
}
mean6(){ local pre=$1; local t=0 n=0 e
  for s in 42 43 44 45 46 47; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "held=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}
cellmean(){ local t=$1; local tt=0 n=0 v
  for s in 0 1 2; do
    v=$(mean6 "gr_${t}_s${s}_e"); [ "$v" = "0.0" ] && continue
    tt=$(awk "BEGIN{print $tt+$v}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $tt/$n : 0)}"
}

log "grid stage 1 evals"
g=0
for amax in 1.8 2.0 2.2 2.6; do
 for mw in 0.1 0.3 0.5; do
  for lr in 1e-4 3e-4 1e-3; do
    t=$(tag "$amax" "$mw" "$lr")
    ( for seed in 0 1 2; do
        A=/workspace/actors/lip4_gr_${t}_s${seed}.pt; [ -e "$A" ] || continue
        for s in 42 43 44 45 46 47; do ev $((g % NGPU)) "gr_${t}_s${seed}_e${s}" $s "$A"; done
      done
      log "CELL ${t} (amax ${amax} mw ${mw} lr ${lr}): HELD $(cellmean "$t")" ) &
    g=$((g + 1))
    [ $((g % NGPU)) -eq 0 ] && wait
  done
 done
done
wait

log "--- GRID (HELD, mean over available training seeds; bar = Latent+CEM-window 44.7) ---"
BEST=""; BESTV=0
for amax in 1.8 2.0 2.2 2.6; do
 for mw in 0.1 0.3 0.5; do
  for lr in 1e-4 3e-4 1e-3; do
    t=$(tag "$amax" "$mw" "$lr"); v=$(cellmean "$t")
    [ "$v" = "0.0" ] && continue
    log "  amax ${amax} mw ${mw} lr ${lr}: ${v}"
    awk "BEGIN{exit !($v > $BESTV)}" && { BESTV=$v; BEST="$amax $mw $lr"; }
  done
 done
done
log "GRID BEST: [${BEST}] HELD ${BESTV}"
echo "$BEST" > /workspace/_GRID_BEST
log "GRIDB_STAGE1_DONE -- stage 2 (3rd seed + steps16k twin for the top cells) is a separate run"
