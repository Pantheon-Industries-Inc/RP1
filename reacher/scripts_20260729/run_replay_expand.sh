#!/bin/bash
# REPLAY-PROB x EXPAND-WEIGHT, the two knobs that helped on OGBench (user,
# 2026-08-01). Both are standard LIPv4 flags -- no task-specific prior -- so
# they are admissible for the headline recipe.
#
#   --replay-prob 0.5   half the training contexts come from the actor's OWN
#                       previously-imagined windows instead of dataset starts,
#                       i.e. the actor practises from the states it actually
#                       reaches.
#   --expand-weight 1.0 value expansion: the critic is additionally trained on
#                       the states the ACTOR visits, not only on dataset
#                       transitions.
#
# Why expand-weight is especially interesting here: the diagnosed failure mode
# on this env is the actor exploiting world-model error in regions the critic
# was never fit on (corr(E_final, HELD) = +0.583). Training the critic where the
# actor actually goes attacks that directly.
#
# Why it might instead BACKFIRE: train_lip_ac's own docstring warns the actor
# and critic can CO-EXPLOIT WM error under expansion -- the critic learns to
# agree with the actor's fantasies. So the 2x2 includes the (0,0) corner as the
# control rather than relying on the earlier 48.9 number.
#
# 2x2 x 3 training seeds = 12 trainings at the winning recipe (1000 steps,
# lr 1e-3, uniform lambda, amax 1.8, mw 0.1, batch 128).
# Screen on SELECTION seeds {50,51}; card the top 2 on REPORTING seeds {42..47}.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
W3=/workspace/metrics/window3_lejepa.pt
SUM=/workspace/results/summary_rxe_lejepa.csv; touch "$SUM"
L=/workspace/logs/rxe; mkdir -p "$L"
GPUS="3 4 5"            # Dyna owns 0-2
SEL="50 51"
REP="42 43 44 45 46 47"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][rxe] $*"; }

BASE="--cache /workspace/caches/canon_lejepa_fs5.pt \
 --cache-td /workspace/caches/canon_lejepa_fs1.pt \
 --h5 /workspace/reacher_slim.h5 --wm $WM --pad-context --init-value $W3 \
 --arch v4 --iters 8 --horizon 5 --max-delta 12 --n-step 50 --batch 128 \
 --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
 --lambda-schedule uniform --mean-weight 0.1 --amax 1.8 \
 --actor-lr 1e-3 --actor-lr-final 1e-4 --steps 1000"

tagof(){ echo "r${1//./}e${2//./}"; }   # r0e0, r05e0, r0e10, r05e10

train_one(){ local rp=$1 ew=$2 seed=$3 gpu=$4
  local t; t=$(tagof "$rp" "$ew")
  local A=/workspace/actors/lip4_rxe_${t}_s${seed}.pt
  [ -f "$A" ] && { log "train ${t} s${seed}: exists"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 21600 python3 "$PLAN/train_lip_ac.py" \
    $BASE --replay-prob "$rp" --expand-weight "$ew" --seed "$seed" \
    --out "$A" --out-value "/workspace/metrics/lip4_rxe_${t}_s${seed}_value.pt" \
    > "$L/train_${t}_s${seed}.log" 2>&1 \
    && log "train ${t} s${seed} DONE" || log "train ${t} s${seed} FAILED"
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

log "12 trainings (2x2 x 3 seeds) on GPUs ${GPUS}"
i=0
for rp in 0 0.5; do for ew in 0 1.0; do for seed in 0 1 2; do
  gpu=$(echo $GPUS | cut -d" " -f$(( (i % 3) + 1 )))
  train_one "$rp" "$ew" "$seed" "$gpu" &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done; done
wait
log "trainings drained"

log "screening on selection seeds {${SEL}}"
i=0
for rp in 0 0.5; do for ew in 0 1.0; do
  t=$(tagof "$rp" "$ew")
  gpu=$(echo $GPUS | cut -d" " -f$(( (i % 3) + 1 )))
  ( for seed in 0 1 2; do
      A=/workspace/actors/lip4_rxe_${t}_s${seed}.pt; [ -f "$A" ] || continue
      for s in $SEL; do ev $gpu "rxe_${t}_s${seed}_sel${s}" $s "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % 3)) -eq 0 ] && wait
done
wait

: > /workspace/results/rxe_screen.txt
for rp in 0 0.5; do for ew in 0 1.0; do
  t=$(tagof "$rp" "$ew"); tot=0; k=0
  for seed in 0 1 2; do
    v=$(meanof "rxe_${t}_s${seed}_sel" held $SEL); [ "$v" = "0.0" ] && continue
    tot=$(awk "BEGIN{print $tot+$v}"); k=$((k+1))
  done
  [ "$k" -eq 0 ] && continue
  m=$(awk "BEGIN{printf \"%.1f\", $tot/$k}")
  echo "$m $t replay=$rp expand=$ew" >> /workspace/results/rxe_screen.txt
  log "  replay ${rp} / expand ${ew}: selection held ${m} (n=${k})"
done; done
log "--- screen ranking (r0e0 is the control = the 48.9 recipe) ---"
sort -rn /workspace/results/rxe_screen.txt | while read -r line; do log "  $line"; done

log "carding the top 2 on reporting seeds"
i=0
sort -rn /workspace/results/rxe_screen.txt | head -2 | awk '{print $2}' | while read -r t; do
  gpu=$(echo $GPUS | cut -d" " -f$(( (i % 3) + 1 )))
  ta=0; tb=0; k=0
  for seed in 0 1 2; do
    A=/workspace/actors/lip4_rxe_${t}_s${seed}.pt; [ -f "$A" ] || continue
    for s in $REP; do ev $gpu "rxe_${t}_s${seed}_e${s}" $s "$A"; done
    v=$(meanof "rxe_${t}_s${seed}_e" held $REP); w=$(meanof "rxe_${t}_s${seed}_e" held10 $REP)
    log "  CARD ${t} s${seed}: HELD ${v} | @0.1 ${w}"
    ta=$(awk "BEGIN{print $ta+$v}"); tb=$(awk "BEGIN{print $tb+$w}"); k=$((k+1))
  done
  [ "$k" -gt 0 ] && log "POOLED ${t}: HELD $(awk "BEGIN{printf \"%.1f\", $ta/$k}") | @0.1 $(awk "BEGIN{printf \"%.1f\", $tb/$k}") over ${k} seeds"
  i=$((i + 1))
done
log "RXE_DONE -- control r0e0 = 48.9 | bar Latent+CEM-window 44.7"
