#!/bin/bash
# Why does PWM win on PLDM? Decompose budget from objective.
#
# PWM carded 99.0 @0.1 on pldm against LIP's 91.5 -- with ZERO rollouts against
# LIP's ~16. Before that is reported as "the reactive policy beats the planner",
# two confounds have to be removed, and I introduced the first one myself:
#
#   BUDGET. PWM ran 8000 steps; LIP runs 1000. I set that by declaring that
#   "PWM-specific settings stay at the trainer's defaults" -- but --steps is not
#   PWM-specific, it is a shared training-budget knob. LIP is at 1000 because
#   its own sweep found 1000 beat 3000 in all 12 pairings, so PWM received 8x
#   the optimisation budget. My "equal terms" claim was wrong on exactly the
#   axis most likely to matter.
#
#   OBJECTIVE. PWM defaults to --dense, scoring every step of the rollout:
#       J = -sum_{t=1..H} gamma^t d_bar(z_t, z_g)
#   LIP scores the terminal plus mean_weight * path mean. The trainer ships
#   --terminal, documented as "LIPv4's objective, isolating the actor
#   architecture as the single variable" -- which is precisely this ablation.
#
# 2x2: steps {1000, 8000} x objective {dense, terminal}. The 8000/dense cell is
# already carded (lejepa 85.9, pldm 99.0) and is skipped, so three new cells.
#
# WHAT EACH OUTCOME WOULD MEAN
#   1000/dense still ~99 on pldm  -> budget was not the story; the reactive
#                                    actor genuinely wins there
#   1000/dense collapses          -> PWM's win was bought with 8x compute and
#                                    the headline comparison was unfair
#   8000/terminal ~ 8000/dense    -> dense supervision is not the mechanism
#   8000/terminal collapses       -> dense per-step scoring is doing the work,
#                                    and LIP could plausibly adopt it
#
# No selection happens here -- four fixed configurations are measured -- so all
# cards go straight to the reporting seeds. Nothing is chosen on them.
#
# Gated behind run_leak6.sh (the headline) so the GPUs are not shared.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_pwmabl.csv; touch "$SUM"
L=/workspace/logs/pwmabl; mkdir -p "$L"
REP="42 43 44 45 46 47"
G=0.98
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][pwmabl] $*"; }
die(){ log "FATAL: $*"; exit 1; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
amax_of(){ [ "$1" = "lejepa" ] && echo 2.2 || echo 1.8; }

if ! grep -q "LEAK6_DONE" /workspace/logs/leak6.log 2>/dev/null; then
  log "waiting for run_leak6.sh (ceiling 8 h)"
  w=0
  while ! grep -q "LEAK6_DONE" /workspace/logs/leak6.log 2>/dev/null; do
    sleep 120; w=$((w + 120))
    [ $((w % 1800)) -eq 0 ] && log "  still waiting, ${w}s"
    if [ "$w" -ge 28800 ]; then
      log "leak6 did not finish inside the ceiling -- proceeding anyway rather"
      log "than parking idle GPUs (a dead upstream gate already cost 6 h today)"
      break
    fi
  done
fi

# ---------------------------------------------------------------- train
log "18 trainings: 3 new cells x 2 bases x 3 seeds (8000/dense already carded)"
i=0
for st in 1000 8000; do for obj in dense terminal; do
  [ "$st" = "8000" ] && [ "$obj" = "dense" ] && continue      # already have it
  for B in lejepa pldm; do for sd in 0 1 2; do
    t="${B}_s${st}_${obj}"
    A=/workspace/actors/pwmabl_${t}_s${sd}.pt
    [ -f "$A" ] && { i=$((i+1)); continue; }
    OBJFLAG="--dense"; [ "$obj" = "terminal" ] && OBJFLAG="--terminal"
    CUDA_VISIBLE_DEVICES=$(( i % 6 )) timeout 28800 python3 "$PLAN/train_pwm_ac.py" \
      --cache /workspace/caches/canon_${B}_fs5.pt \
      --cache-td /workspace/caches/canon_${B}_fs1.pt \
      --wm "$(wm_of $B)" \
      --init-value /workspace/metrics/window3_${B}_e005_g098.pt \
      --gamma "$G" --amax "$(amax_of $B)" --horizon 5 --max-delta 12 \
      --n-step 50 --batch 128 --steps "$st" $OBJFLAG \
      --expectile 0.1 --expectile-final 0.03 --seed "$sd" \
      --out "$A" --out-value "/workspace/metrics/pwmabl_${t}_s${sd}_value.pt" \
      > "$L/train_${t}_s${sd}.log" 2>&1 || log "TRAIN FAILED ${t} s${sd}" &
    i=$((i + 1)); [ $((i % 12)) -eq 0 ] && wait
  done; done
done; done
wait
log "actors: $(ls /workspace/actors/pwmabl_*_s?.pt 2>/dev/null | wc -l)/18"

# ---------------------------------------------------------------- card
ev(){ local gpu=$1 B=$2 nm=$3 seed=$4 actor=$5
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$(wm_of $B)" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver=pwm solver.actor_path="$actor" solver.batch_size=10 \
    output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && log "  !! ${nm} no score: $(tail -1 "$L/${nm}.log" | cut -c1-60)"
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
meanof(){ local pre=$1 key=$2; local t=0 n=0 e
  for s in $REP; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"; }

log "carding on reporting seeds"
i=0
for st in 1000 8000; do for obj in dense terminal; do
  [ "$st" = "8000" ] && [ "$obj" = "dense" ] && continue
  for B in lejepa pldm; do
    t="${B}_s${st}_${obj}"
    ( for sd in 0 1 2; do
        A=/workspace/actors/pwmabl_${t}_s${sd}.pt; [ -f "$A" ] || continue
        for s in $REP; do ev $(( i % 4 )) "$B" "abl_${t}_s${sd}_e${s}" "$s" "$A"; done
      done ) &
    i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
  done
done; done
wait

log "=============== PWM: steps x objective (reporting seeds, @0.1) ==============="
log "  reference  LIP 1000 steps : lejepa 97.7  pldm 91.5"
log "  reference  Latent+CEM bar : lejepa 84.3  pldm 78.3"
for B in lejepa pldm; do
  log "--- ${B} ---"
  for st in 1000 8000; do for obj in dense terminal; do
    if [ "$st" = "8000" ] && [ "$obj" = "dense" ]; then
      v=$([ "$B" = "lejepa" ] && echo 85.9 || echo 99.0)
      log "  $(printf '%-5s %-9s @0.1 %-6s' "$st" "$obj" "$v")  [already carded]"
      continue
    fi
    t="${B}_s${st}_${obj}"; ta=0; tb=0; k=0; nsc=0
    for sd in 0 1 2; do
      a=$(meanof "abl_${t}_s${sd}_e" held10); c=$(meanof "abl_${t}_s${sd}_e" held)
      n=$(grep -c "^abl_${t}_s${sd}_e.*held=[0-9]" "$SUM"); nsc=$((nsc + n))
      [ "$n" -eq 0 ] && continue
      ta=$(awk "BEGIN{print $ta+$a}"); tb=$(awk "BEGIN{print $tb+$c}"); k=$((k+1))
    done
    [ "$k" -eq 0 ] && { log "  $(printf '%-5s %-9s' "$st" "$obj")  NOTHING SCORED"; continue; }
    log "  $(printf '%-5s %-9s @0.1 %-6s @0.05 %-6s (%s/18)' "$st" "$obj" \
        "$(awk "BEGIN{printf \"%.1f\", $ta/$k}")" "$(awk "BEGIN{printf \"%.1f\", $tb/$k}")" "$nsc")"
  done; done
done
log "PWMABL_DONE"
