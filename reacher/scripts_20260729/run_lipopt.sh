#!/bin/bash
# Follow the learning-rate gradient, scored at @0.1 rad.
#
# WHY THIS AXIS. Re-scoring stage 2 at @0.1 showed actor-LR dominating every
# other knob by an order of magnitude:
#     lejepa   lr 3e-4 -> 90.0     lr 1e-3 -> 74.3
#     pldm     lr 3e-4 -> 92.0     lr 1e-3 -> 81.3
# 3e-4 was the low edge of the swept range, so the optimum may well lie below
# it. Lower LR plausibly needs a longer schedule, hence steps is swept with it.
#
# mean-weight is included because it trades terminal cost against the path mean
# in L = v_K + mw * mean(v_0..v_{K-1}), and its 0.1 setting was tuned at lr 1e-3
# on a 1000-step schedule -- both of which have changed underneath it.
#
# CRITIC PURITY (user, 2026-08-01): the critic stays a pure reward / cost-to-go
# object. Nothing here touches what the critic is a function of -- these are
# optimiser knobs on the actor plus the path/terminal weighting. The earlier
# idea of anchoring the value toward an L2 latent distance is dropped for
# exactly this reason.
#
# SHARED vs PER-BASE, unchanged from the joint sweep: expectile 0.05, replay
# 0.5, expand 0 and n-step 50 are shared; amax is per-base and pinned at each
# base's winner (2.2 lejepa / 1.8 pldm) to keep this sweep to one question.
#
# HEADLINE IS @0.1: selection screens on held10, not held. Bars to beat are
# lejepa 84.3 / pldm 78.3. @0.05 is still recorded for every cell so the old
# headline stays reconstructible.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
SUM=/workspace/results/summary_lipopt.csv; touch "$SUM"
L=/workspace/logs/lipopt; mkdir -p "$L"
SEL="50 51"
REP="42 43 44 45 46 47"
GPUS="1 2 3 4 5"          # GPU 0 is Dyna's fine-tune
NG=5
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][lipopt] $*"; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
amax_of(){ [ "$1" = "lejepa" ] && echo 2.2 || echo 1.8; }
gpu_at(){ echo $GPUS | cut -d" " -f$(( ($1 % NG) + 1 )); }

LRS="1e-4 3e-4"
STEPS="1000 3000"
MWS="0.03 0.1 0.3"

tagof(){ echo "${1}_l${2//e-/e}s${3}m${4//./}"; }

train(){ local B=$1 lr=$2 st=$3 mw=$4 sd=$5 gpu=$6
  local t; t=$(tagof "$B" "$lr" "$st" "$mw")
  local A=/workspace/actors/lip4_opt_${t}_s${sd}.pt
  [ -f "$A" ] && return 0
  local lrf; case "$lr" in 1e-4) lrf=1e-5;; 3e-4) lrf=3e-5;; *) lrf=1e-5;; esac
  CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/canon_${B}_fs5.pt \
    --cache-td /workspace/caches/canon_${B}_fs1.pt \
    --h5 "$SLIM" --wm "$(wm_of $B)" --pad-context \
    --init-value /workspace/metrics/window3_${B}_e005.pt \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight "$mw" --amax "$(amax_of $B)" \
    --replay-prob 0.5 --expand-weight 0 --steps "$st" \
    --actor-lr "$lr" --actor-lr-final "$lrf" --seed "$sd" \
    --out "$A" --out-value "/workspace/metrics/lip4_opt_${t}_s${sd}_value.pt" \
    > "$L/train_${t}_s${sd}.log" 2>&1 || log "TRAIN FAILED ${t} s${sd}"
}

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
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
# cut -d= -f2: a bare [0-9.]+ grep would return the "10" inside the key held10=
meanof(){ local pre=$1 key=$2; shift 2; local t=0 n=0 e
  for s in "$@"; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}
cfgmean(){ local pre=$1 key=$2; local t=0 k=0 v
  for sd in 0 1 2; do
    v=$(meanof "${pre}_s${sd}_sel" "$key" $SEL); [ "$v" = "0.0" ] && continue
    t=$(awk "BEGIN{print $t+$v}"); k=$((k+1))
  done
  awk "BEGIN{printf \"%.1f\", ($k ? $t/$k : 0)}"
}

# ---------------------------------------------------------------- train
N=0; for B in lejepa pldm; do for lr in $LRS; do for st in $STEPS; do for mw in $MWS; do
  N=$((N + 3)); done; done; done; done
log "${N} trainings: 2 bases x lr{${LRS}} x steps{${STEPS}} x mw{${MWS}} x 3 seeds"
i=0
for B in lejepa pldm; do for lr in $LRS; do for st in $STEPS; do for mw in $MWS; do
  for sd in 0 1 2; do
    train "$B" "$lr" "$st" "$mw" "$sd" "$(gpu_at $i)" &
    i=$((i + 1)); [ $((i % 15)) -eq 0 ] && wait
  done
done; done; done; done
wait
log "trainings drained ($(ls /workspace/actors/lip4_opt_*.pt 2>/dev/null | grep -c "_s[0-9]\.pt$") actors)"

# ---------------------------------------------------------------- screen on selection seeds
i=0
for B in lejepa pldm; do for lr in $LRS; do for st in $STEPS; do for mw in $MWS; do
  t=$(tagof "$B" "$lr" "$st" "$mw"); g=$(gpu_at $i)
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_opt_${t}_s${sd}.pt; [ -f "$A" ] || continue
      for s in $SEL; do ev "$g" "$B" "opt_${t}_s${sd}_sel${s}" "$s" "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % NG)) -eq 0 ] && wait
done; done; done; done
wait

# ---------------------------------------------------------------- rank at @0.1
log "=============== SCREEN, ranked by @0.1 (selection seeds) ==============="
: > /workspace/results/lipopt_screen.txt
for B in lejepa pldm; do for lr in $LRS; do for st in $STEPS; do for mw in $MWS; do
  t=$(tagof "$B" "$lr" "$st" "$mw")
  v=$(cfgmean "opt_${t}" held10); v5=$(cfgmean "opt_${t}" held)
  [ "$v" = "0.0" ] && continue
  echo "$v $B $lr $st $mw $t $v5" >> /workspace/results/lipopt_screen.txt
done; done; done; done
for B in lejepa pldm; do
  log "--- ${B} (bar @0.1: $([ "$B" = "lejepa" ] && echo 84.3 || echo 78.3)) ---"
  awk -v b="$B" '$2==b' /workspace/results/lipopt_screen.txt | sort -rn | \
    while read -r v bb lr st mw t v5; do
      log "  lr ${lr} steps ${st} mw ${mw}:  @0.1 ${v}   @0.05 ${v5}"
    done
done

# ---------------------------------------------------------------- card the winner per base
for B in lejepa pldm; do
  bar=$([ "$B" = "lejepa" ] && echo 84.3 || echo 78.3)
  line=$(awk -v b="$B" '$2==b' /workspace/results/lipopt_screen.txt | sort -rn | head -1)
  [ -z "$line" ] && { log "no winner for ${B}"; continue; }
  set -- $line; t=$6
  log "WINNER ${B}: lr $3 steps $4 mw $5 (screen @0.1 $1)"
  i=0; ta=0; tb=0; k=0
  for sd in 0 1 2; do
    A=/workspace/actors/lip4_opt_${t}_s${sd}.pt; [ -f "$A" ] || continue
    for s in $REP; do ev "$(gpu_at $i)" "$B" "opt_${t}_s${sd}_e${s}" "$s" "$A"; i=$((i+1)); done
    v=$(meanof "opt_${t}_s${sd}_e" held10 $REP); w=$(meanof "opt_${t}_s${sd}_e" held $REP)
    log "  CARD ${B} s${sd}: @0.1 ${v} | @0.05 ${w}"
    ta=$(awk "BEGIN{print $ta+$v}"); tb=$(awk "BEGIN{print $tb+$w}"); k=$((k+1))
  done
  [ "$k" -gt 0 ] && log "POOLED ${B}: @0.1 $(awk "BEGIN{printf \"%.1f\", $ta/$k}") (bar ${bar}) | @0.05 $(awk "BEGIN{printf \"%.1f\", $tb/$k}")"
done
log "LIPOPT_DONE -- incumbent: lejepa 89.0 / pldm 90.4 at @0.1"
