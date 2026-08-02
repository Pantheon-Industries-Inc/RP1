#!/bin/bash
# gamma 0.98 combined with the per-base LIP knobs. Never tested together.
#
# Two independent findings have landed and neither has met the other:
#
#   DISCOUNTING (A/B at the OLD incumbent recipe, lr 3e-4 / mw 0.1):
#       lejepa  g0.98 95.8 (+11.5)   g0.995 81.1 (-3.2)   g1.0 89.0 (+4.7)
#       pldm    g0.98 84.6 (+6.3)    g0.995 79.2 (+0.9)   g1.0 90.4 (+12.1)
#     Non-monotone, and the prediction that 0.98 < 0.995 < 1.0 was wrong.
#     As a SHARED knob scored by min-margin, 0.98 wins: +6.3 against +4.7.
#
#   PER-BASE LIP KNOBS (swept at gamma 1.0):
#       lejepa  lr 1e-4 mw 0.3 -> 91.9   (up from 89.0)
#       pldm    mw 0.3 was selection NOISE: it tied mw 0.1 at 92.0 on the
#               selection seeds, won the tie-break, and then carded 86.1
#               against the incumbent's 90.4. pldm keeps mw 0.1.
#
# So the grid below crosses gamma 0.98 with lr {1e-4, 3e-4} x mw {0.1, 0.3} per
# base. gamma stays SHARED (it is a cost-to-go knob); lr and mw are per-base.
#
# The tie-break lesson is applied here: ties on the selection seeds are reported
# rather than silently resolved by sort order, because that is exactly how pldm
# ended up 4.3 points down.
#
# Bars @0.1: lejepa 84.3, pldm 78.3. Best so far: lejepa 95.8, pldm 90.4.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
SUM=/workspace/results/summary_g098grid.csv; touch "$SUM"
L=/workspace/logs/g098grid; mkdir -p "$L"
SEL="50 51"; REP="42 43 44 45 46 47"; G=0.98
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][g98] $*"; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
amax_of(){ [ "$1" = "lejepa" ] && echo 2.2 || echo 1.8; }
bar_of(){ [ "$1" = "lejepa" ] && echo 84.3 || echo 78.3; }
tagof(){ echo "${1}_l${2//e-/e}m${3//./}"; }

log "24 trainings at gamma ${G}: 2 bases x lr{1e-4,3e-4} x mw{0.1,0.3} x 3 seeds"
i=0
for B in lejepa pldm; do for lr in 1e-4 3e-4; do for mw in 0.1 0.3; do for sd in 0 1 2; do
  t=$(tagof "$B" "$lr" "$mw"); A=/workspace/actors/lip4_g98_${t}_s${sd}.pt
  [ -f "$A" ] && { i=$((i+1)); continue; }
  lrf=1e-5; [ "$lr" = "3e-4" ] && lrf=3e-5
  CUDA_VISIBLE_DEVICES=$(( (i % 5) + 1 )) timeout 28800 python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/canon_${B}_fs5.pt \
    --cache-td /workspace/caches/canon_${B}_fs1.pt \
    --h5 "$SLIM" --wm "$(wm_of $B)" --pad-context \
    --init-value /workspace/metrics/window3_${B}_e005_g098.pt \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --gamma "$G" --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight "$mw" --amax "$(amax_of $B)" \
    --replay-prob 0.5 --expand-weight 0 --steps 1000 \
    --actor-lr "$lr" --actor-lr-final "$lrf" --seed "$sd" \
    --out "$A" --out-value "/workspace/metrics/lip4_g98_${t}_s${sd}_value.pt" \
    > "$L/train_${t}_s${sd}.log" 2>&1 || log "TRAIN FAILED ${t} s${sd}" &
  i=$((i + 1)); [ $((i % 10)) -eq 0 ] && wait
done; done; done; done
wait
log "trainings drained ($(ls /workspace/actors/lip4_g98_*_s?.pt 2>/dev/null | wc -l)/24)"

ev(){ local gpu=$1 B=$2 nm=$3 seed=$4 actor=$5
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$(wm_of $B)" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=lip solver.actor_path="$actor" solver.rollout_compat=false \
    solver.batch_size=10 output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && log "  !! ${nm} no score: $(tail -1 "$L/${nm}.log" | cut -c1-60)"
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
meanof(){ local pre=$1 key=$2; shift 2; local t=0 n=0 e
  for s in "$@"; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"; }
cfgmean(){ local pre=$1 key=$2; local t=0 k=0 v
  for sd in 0 1 2; do
    v=$(meanof "${pre}_s${sd}_sel" "$key" $SEL); [ "$v" = "0.0" ] && continue
    t=$(awk "BEGIN{print $t+$v}"); k=$((k+1)); done
  awk "BEGIN{printf \"%.1f\", ($k ? $t/$k : 0)}"; }

log "screening on selection seeds, scored @0.1"
i=0
for B in lejepa pldm; do for lr in 1e-4 3e-4; do for mw in 0.1 0.3; do
  t=$(tagof "$B" "$lr" "$mw"); g=$(( (i % 5) + 1 ))
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_g98_${t}_s${sd}.pt; [ -f "$A" ] || continue
      for s in $SEL; do ev "$g" "$B" "g98_${t}_s${sd}_sel${s}" "$s" "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % 5)) -eq 0 ] && wait
done; done; done
wait

log "================= SCREEN at gamma ${G} (selection seeds, @0.1) ================="
: > /workspace/results/g098_screen.txt
for B in lejepa pldm; do for lr in 1e-4 3e-4; do for mw in 0.1 0.3; do
  t=$(tagof "$B" "$lr" "$mw"); v=$(cfgmean "g98_${t}" held10)
  [ "$v" = "0.0" ] && continue
  echo "$v $B $lr $mw $t $(cfgmean "g98_${t}" held)" >> /workspace/results/g098_screen.txt
  log "  ${B} lr ${lr} mw ${mw}: @0.1 ${v}  @0.05 $(cfgmean "g98_${t}" held)"
done; done; done

for B in lejepa pldm; do
  top=$(awk -v b="$B" '$2==b {print $1}' /workspace/results/g098_screen.txt | sort -rn | head -1)
  nties=$(awk -v b="$B" -v t="$top" '$2==b && $1==t' /workspace/results/g098_screen.txt | wc -l)
  [ "$nties" -gt 1 ] && log "  NOTE ${B}: ${nties}-way tie at ${top} on the selection seeds -- carding all of them"
  awk -v b="$B" -v t="$top" '$2==b && $1==t' /workspace/results/g098_screen.txt | while read -r v bb lr mw tg v5; do
    log "WINNER ${B}: gamma ${G} lr ${lr} mw ${mw} (screen @0.1 ${v})"
    i=0; ta=0; tb=0; k=0
    for sd in 0 1 2; do
      A=/workspace/actors/lip4_g98_${tg}_s${sd}.pt; [ -f "$A" ] || continue
      for s in $REP; do ev $(( (i % 5) + 1 )) "$B" "g98_${tg}_s${sd}_e${s}" "$s" "$A"; i=$((i+1)); done
      a=$(meanof "g98_${tg}_s${sd}_e" held10 $REP); c=$(meanof "g98_${tg}_s${sd}_e" held $REP)
      log "  CARD ${B} ${tg} s${sd}: @0.1 ${a} | @0.05 ${c}"
      ta=$(awk "BEGIN{print $ta+$a}"); tb=$(awk "BEGIN{print $tb+$c}"); k=$((k+1))
    done
    [ "$k" -gt 0 ] && log "POOLED ${B} ${tg}: @0.1 $(awk "BEGIN{printf \"%.1f\", $ta/$k}") (bar $(bar_of $B)) | @0.05 $(awk "BEGIN{printf \"%.1f\", $tb/$k}")"
  done
done
log "G098GRID_DONE -- best so far: lejepa 95.8 (g0.98 mw0.1), pldm 90.4 (g1.0 mw0.1)"
