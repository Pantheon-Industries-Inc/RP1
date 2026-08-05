#!/bin/bash
# GRID @ 1000 STEPS -- find the best hypers for the recipe we actually intend to
# ship (user, 2026-07-31).
#
# Context: held-vs-step curves showed every actor peaks at ~1000 steps and then
# degrades (seed 0: 41 -> 23 -> 18 -> 12 -> 9), because the actor optimises into
# world-model error (corr(E_final, HELD) = +0.583). Averaged over all 9 actors
# the selection-seed mean by snapshot was 1000->49.2 vs 43-45 everywhere else, so
# "train 1000 steps" is a single recipe that is simultaneously best for every
# seed -- not a per-seed compromise.
#
# The optimum for OTHER hypers may move with the budget: at 8000 steps a LOW
# actor-lr won (it had time to converge), but at 1000 steps a HIGHER one may be
# better. So re-sweep rather than inherit.
#
#   actor-lr     {1e-4, 3e-4, 1e-3}
#   lambda-sched {uniform, geom-early}
#   amax         {1.8, 2.2, 2.6}
#   mean-weight  {0.1, 0.3}
#   = 36 configs x training seeds {0,1,2} = 108 trainings (~10 min each)
#
# Budget shape: training is cheap now, EVALUATION is the bottleneck. So screen
# every config on the SELECTION seeds {50,51} only, then card the top 3 on the
# untouched REPORTING seeds {42..47}. Screening and reporting draws stay
# disjoint so the winner is not chosen on its own test set.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
W3=/workspace/metrics/window3_lejepa.pt
SUM=/workspace/results/summary_grid1k_lejepa.csv; touch "$SUM"
L=/workspace/logs/grid1k; mkdir -p "$L"
SEL="50 51"
REP="42 43 44 45 46 47"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][g1k] $*"; }

log "waiting for the fixed-1k evaluation to finish"
while ps -eo args --no-headers | grep -q "[r]un_fixed1k.sh"; do sleep 60; done

BASE="--cache /workspace/caches/canon_lejepa_fs5.pt \
 --cache-td /workspace/caches/canon_lejepa_fs1.pt \
 --h5 /workspace/reacher_slim.h5 --wm $WM --pad-context --init-value $W3 \
 --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
 --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
 --lambda-base 0.5 --steps 1000"

LRS="1e-4 3e-4 1e-3"
SCHEDS="uniform geom-early"
AMAXES="1.8 2.2 2.6"
MWS="0.1 0.3"

mktag(){ # lr sched amax mw -- parameter expansion only (a function named `tr`
         # once shadowed /usr/bin/tr here and silently collapsed 36 tags into 18)
  local lr="$1" sc="$2" am="$3" mw="$4"
  case "$sc" in uniform) sc=u ;; geom-early) sc=ge ;; esac
  echo "l${lr//e-/e}${sc}a${am//./}m${mw//./}"
}
lrfin(){ case "$1" in 1e-4) echo 1e-5 ;; 3e-4) echo 3e-5 ;; 1e-3) echo 1e-4 ;; *) echo 1e-5 ;; esac; }

train_cell(){ # lr sched amax mw seed gpu
  local lr="$1" sc="$2" am="$3" mw="$4" seed="$5" gpu="$6"
  local t; t=$(mktag "$lr" "$sc" "$am" "$mw")
  local A=/workspace/actors/lip4_g1k_${t}_s${seed}.pt
  [ -f "$A" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu timeout 21600 python3 "$PLAN/train_lip_ac.py" \
    $BASE --actor-lr "$lr" --actor-lr-final "$(lrfin $lr)" \
    --lambda-schedule "$sc" --amax "$am" --mean-weight "$mw" --seed "$seed" \
    --out "$A" --out-value "/workspace/metrics/lip4_g1k_${t}_s${seed}_value.pt" \
    > "$L/train_${t}_s${seed}.log" 2>&1 || log "train ${t} s${seed} FAILED"
}
ev(){ local gpu=$1 nm=$2 seed=$3 actor=$4
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=lip solver.actor_path="$actor" solver.rollout_compat=false \
    solver.batch_size=10 output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
meanof(){ local pre=$1 key=$2; shift 2; local t=0 n=0 e
  for s in "$@"; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}

# ---------------------------------------------------------------- trainings
log "108 trainings (36 configs x 3 seeds) at 1000 steps, 3 per GPU"
i=0
for lr in $LRS; do for sc in $SCHEDS; do for am in $AMAXES; do for mw in $MWS; do
  for seed in 0 1 2; do
    train_cell "$lr" "$sc" "$am" "$mw" "$seed" $((i % 6)) &
    i=$((i + 1))
    [ $((i % 18)) -eq 0 ] && wait
  done
done; done; done; done
wait
log "trainings drained: $(ls /workspace/actors/lip4_g1k_*.pt 2>/dev/null | wc -l) actors"

# ---------------------------------------------------------------- screen on SELECTION seeds
log "screening all 36 configs on selection seeds {${SEL}} -- 6 parallel streams"
screen_one(){ local gpu=$1 t=$2
  for seed in 0 1 2; do
    A=/workspace/actors/lip4_g1k_${t}_s${seed}.pt; [ -f "$A" ] || continue
    for s in $SEL; do ev $gpu "g1k_${t}_s${seed}_sel${s}" $s "$A"; done
  done
}
g=0
for lr in $LRS; do for sc in $SCHEDS; do for am in $AMAXES; do for mw in $MWS; do
  t=$(mktag "$lr" "$sc" "$am" "$mw")
  screen_one $((g % 6)) "$t" &
  g=$((g + 1)); [ $((g % 6)) -eq 0 ] && wait
done; done; done; done
wait

log "--- screen (mean held@0.05 over 3 training seeds x selection seeds) ---"
: > /workspace/results/grid1k_screen.txt
for lr in $LRS; do for sc in $SCHEDS; do for am in $AMAXES; do for mw in $MWS; do
  t=$(mktag "$lr" "$sc" "$am" "$mw")
  tot=0; k=0
  for seed in 0 1 2; do
    v=$(meanof "g1k_${t}_s${seed}_sel" held $SEL); [ "$v" = "0.0" ] && continue
    tot=$(awk "BEGIN{print $tot+$v}"); k=$((k+1))
  done
  [ "$k" -eq 0 ] && continue
  m=$(awk "BEGIN{printf \"%.1f\", $tot/$k}")
  echo "$m $t lr=$lr sched=$sc amax=$am mw=$mw" >> /workspace/results/grid1k_screen.txt
  log "  $t (lr $lr, $sc, amax $am, mw $mw): $m"
done; done; done; done
sort -rn /workspace/results/grid1k_screen.txt | head -6 | while read -r line; do log "TOP $line"; done

# ---------------------------------------------------------------- card the top 3
log "carding the top 3 configs on reporting seeds {${REP}}"
g=0
sort -rn /workspace/results/grid1k_screen.txt | head -3 | awk '{print $2}' | while read -r t; do
  ta=0; tb=0; k=0
  for seed in 0 1 2; do
    A=/workspace/actors/lip4_g1k_${t}_s${seed}.pt; [ -f "$A" ] || continue
    for s in $REP; do ev $(( (g % 5) + 1 )) "g1k_${t}_s${seed}_e${s}" $s "$A"; g=$((g+1)); done
    v=$(meanof "g1k_${t}_s${seed}_e" held $REP); w=$(meanof "g1k_${t}_s${seed}_e" held10 $REP)
    log "  CARD ${t} s${seed}: HELD ${v} | @0.1 ${w}"
    ta=$(awk "BEGIN{print $ta+$v}"); tb=$(awk "BEGIN{print $tb+$w}"); k=$((k+1))
  done
  [ "$k" -gt 0 ] && log "POOLED ${t}: HELD $(awk "BEGIN{printf \"%.1f\", $ta/$k}") | @0.1 $(awk "BEGIN{printf \"%.1f\", $tb/$k}") over ${k} seeds"
done
log "GRID1K_DONE -- bar: Latent+CEM-window 44.7 @0.05 / 84.3 @0.1"
