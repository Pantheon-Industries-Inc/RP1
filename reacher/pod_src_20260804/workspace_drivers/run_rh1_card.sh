#!/bin/bash
# Stage 3: the rh=1 report card, reporting seeds 42-47.
#
# PROTOCOL. Every arm is carded under the plan-cost readout that is BEST FOR
# THAT ARM, chosen on the selection seeds (50-51) and never on these. From the
# 108-cell RLP screen and the 24-cell sampling screen:
#
#   RLP (randhorizon-aligned)  I3 off       100.0 @0.1 vs 100.0 deadline, but
#                                           96.0 vs 65.0 @0.05 -- I3 hurts an
#                                           actor already trained to control
#                                           every step
#   Latent+CEM                 I3 deadline  87.0 vs 32.0 @0.1  (lejepa)
#   Value+CEM                  I3 deadline  48.0 vs 15.0 @0.1  (lejepa)
#   Latent/Value+Adam          screened in stage A below -- same terminal-cost
#                                           family, but measured not assumed
#
# Giving each arm its own best readout is the choice that is CONSERVATIVE for
# the claim: the baselines gain 55 points from I3 and RLP gains nothing, so
# adopting one global readout on RLP's evidence would have inflated the margin.
#
# FIVE ROWS, two of them RLP:
#   RLP (rh=1 recipe)   randhorizon-aligned, 6 training seeds -- the arm's own
#                       best configuration at this cadence
#   RLP (as reported)   the rh=5-trained artifact the paper's table contains,
#                       run unchanged at rh=1. This row is what "the reported
#                       artifact at the published cadence" actually scores, and
#                       it must be published alongside the first one.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_rh1_card.csv; touch "$SUM"
SEL_SUM=/workspace/results/summary_rh1_bars.csv
L=/workspace/logs/rh1card; mkdir -p "$L"
REP="42 43 44 45 46 47"
L2=/workspace/metrics/l2window3.pt
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][card] $*"; }
val_of(){ echo "/workspace/metrics/window3_${1}_e005_g098.pt"; }

# ev <sum> <gpu> <base> <name> <seed> <mode> <deadline0|1> -- solver args follow
ev(){ local sm=$1 gpu=$2 B=$3 nm=$4 seed=$5 mode=$6 dl=$7; shift 7
  grep -q "^${nm}," "$sm" && return 0
  local DLARG=""; [ "$dl" = "1" ] && DLARG="+plan_config.deadline=50"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu I3_MODE=$mode \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy=/workspace/swm_home/checkpoints/${B}_reacher \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    plan_config.receding_horizon=1 $DLARG \
    "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10 i3
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  i3=$(grep -c "^\[I3\]" "$L/${nm}.log")
  # guard: a deadline cell that never fired I3 silently returns the unaligned
  # baseline, and a core-dumped sampling eval used to average in as a real 0.0
  if [ "$dl" = "1" ] && [ "$i3" = "0" ]; then log "  !! ${nm} I3 NEVER FIRED"; h=NOFIRE; t10=NOFIRE; fi
  if [ -z "$h" ]; then log "  !! ${nm} NO SCORE: $(tail -2 "$L/${nm}.log" | head -1 | cut -c1-60)"; h=FAIL; t10=FAIL; fi
  echo "${nm},held=${h},held10=${t10},i3=${i3}" >> "$sm"
}

run_queue(){ local q=$1 sm=$2 gpus=$3
  local W=0; for _ in $gpus; do W=$((W + 1)); done
  local k=0
  for g in $gpus; do
    ( idx=0
      while IFS='|' read -r B nm s R DL ARGS; do
        if [ $((idx % W)) -eq $k ]; then ev "$sm" "$g" "$B" "$nm" "$s" "$R" "$DL" $ARGS; fi
        idx=$((idx + 1))
      done < "$q" ) &
    k=$((k + 1))
  done
  wait
}

# ================================================== stage A: Adam readout screen
# The CEM arms were screened; Adam was not. Same terminal-cost family, but the
# gradient path reaches the cost differently (GradientSolver needs requires_grad
# through the metric), so measure rather than inherit.
QA=$L/queueA.txt; : > "$QA"
for B in lejepa pldm; do for R in off deadline; do
  DL=1; [ "$R" = "off" ] && DL=0
  for s in 50 51; do
    echo "$B|l2adam_${B}_${R}_e${s}|$s|$R|$DL|solver=adam solver.n_steps=10 solver.num_samples=300 +metric=$L2" >> "$QA"
    echo "$B|tdadam_${B}_${R}_e${s}|$s|$R|$DL|solver=adam solver.n_steps=10 solver.num_samples=300 +metric=$(val_of $B)" >> "$QA"
  done
done; done
log "stage A: $(wc -l < "$QA") Adam readout cells on the SELECTION seeds"
run_queue "$QA" "$SEL_SUM" "0 1 2 3"

selmean(){ grep "^${1}" "$SEL_SUM" | grep -oE "${2}=[0-9.]+" | cut -d= -f2 \
           | awk '{t+=$1;n++}END{print (n?t/n:0)}'; }
# pick per (arm, base) whichever readout scored higher @0.1 on the selection seeds
pick(){ local arm=$1 B=$2
  local o d; o=$(selmean "${arm}_${B}_off_" held10); d=$(selmean "${arm}_${B}_deadline_" held10)
  awk -v o="$o" -v d="$d" 'BEGIN{print (d>o)?"deadline":"off"}'
}
declare -A RD
for B in lejepa pldm; do for A in l2cem tdcem l2adam tdadam; do
  RD[$A,$B]=$(pick "$A" "$B")
  log "  readout for ${A} ${B}: ${RD[$A,$B]} (off $(selmean "${A}_${B}_off_" held10) vs deadline $(selmean "${A}_${B}_deadline_" held10) @0.1)"
done; done

# ==================================================== stage B: the report card
QB=$L/queueB.txt; : > "$QB"
for B in lejepa pldm; do
  for s in $REP; do
    # --- RLP, rh=1 recipe (randhorizon), 6 training seeds, I3 off
    for sd in 0 1 2 3 4 5; do
      A=/workspace/actors/rh1fin_${B}_s${sd}.pt
      [ -f "$A" ] && echo "$B|c_rlpN_${B}_s${sd}_e${s}|$s|off|0|solver=lip solver.actor_path=$A solver.rollout_compat=false" >> "$QB"
    done
    # --- RLP, the reported artifact, unchanged, at rh=1
    for sd in 0 1 2 3 4 5; do
      A=/workspace/actors/lip4_leak6_${B}_s${sd}.pt
      [ -f "$A" ] && echo "$B|c_rlpR_${B}_s${sd}_e${s}|$s|off|0|solver=lip solver.actor_path=$A solver.rollout_compat=false" >> "$QB"
    done
    # --- PWM, reactive, 0 rollouts (cadence-independent: it acts every step)
    for sd in 0 1 2; do
      A=/workspace/actors/pwmabl_${B}_s1000_terminal_s${sd}.pt
      [ -f "$A" ] && echo "$B|c_pwm_${B}_s${sd}_e${s}|$s|off|0|solver=pwm solver.actor_path=$A" >> "$QB"
    done
    # --- sampling arms, each under its screened readout
    for A in l2cem tdcem; do
      R=${RD[$A,$B]}; DL=1; [ "$R" = "off" ] && DL=0
      M=$L2; [ "$A" = tdcem ] && M=$(val_of $B)
      echo "$B|c_${A}_${B}_e${s}|$s|$R|$DL|solver=cem solver.n_steps=10 +metric=$M" >> "$QB"
    done
    for A in l2adam tdadam; do
      R=${RD[$A,$B]}; DL=1; [ "$R" = "off" ] && DL=0
      M=$L2; [ "$A" = tdadam ] && M=$(val_of $B)
      echo "$B|c_${A}_${B}_e${s}|$s|$R|$DL|solver=adam solver.n_steps=10 solver.num_samples=300 +metric=$M" >> "$QB"
    done
  done
done
log "stage B: $(wc -l < "$QB") report cells on seeds $REP"
run_queue "$QB" "$SUM" "0 1 2 3 4 5"

# ==================================================================== report
stat(){ grep "^${1}" "$SUM" | grep -oE "${2}=[0-9.]+" | cut -d= -f2 \
        | awk '{a[n++]=$1;t+=$1}END{if(!n){printf "  --  ";exit}
               m=t/n; for(i=0;i<n;i++)s+=(a[i]-m)^2;
               printf "%5.1f+-%-4.1f(n=%d)", m, (n>1?sqrt(s/(n-1)):0), n}'; }
log "================ RLP paper, reacher, receding_horizon=1 ================"
log "  reported rh=5 for reference: RLP 98.2/94.2 | Latent+CEM 84.3/78.3 @0.1"
for M in held10 held; do
  lbl=$([ "$M" = held10 ] && echo "@0.1" || echo "@0.05")
  log "  ---------------- $lbl ----------------"
  printf "      %-26s %-8s %-24s %s\n" arm rollouts lejepa pldm
  printf "      %-26s %-8s %-24s %s\n" "RLP (rh=1 recipe)"   "~16"   "$(stat c_rlpN_lejepa_ $M)" "$(stat c_rlpN_pldm_ $M)"
  printf "      %-26s %-8s %-24s %s\n" "RLP (reported artifact)" "~16" "$(stat c_rlpR_lejepa_ $M)" "$(stat c_rlpR_pldm_ $M)"
  printf "      %-26s %-8s %-24s %s\n" "Latent + CEM (bar)"  "3000"  "$(stat c_l2cem_lejepa_ $M)" "$(stat c_l2cem_pldm_ $M)"
  printf "      %-26s %-8s %-24s %s\n" "PWM (reactive)"      "0"     "$(stat c_pwm_lejepa_ $M)" "$(stat c_pwm_pldm_ $M)"
  printf "      %-26s %-8s %-24s %s\n" "Value + CEM"         "3000"  "$(stat c_tdcem_lejepa_ $M)" "$(stat c_tdcem_pldm_ $M)"
  printf "      %-26s %-8s %-24s %s\n" "Latent + Adam"       "3000"  "$(stat c_l2adam_lejepa_ $M)" "$(stat c_l2adam_pldm_ $M)"
  printf "      %-26s %-8s %-24s %s\n" "Value + Adam"        "3000"  "$(stat c_tdadam_lejepa_ $M)" "$(stat c_tdadam_pldm_ $M)"
done
log "  readouts used: $(for B in lejepa pldm; do for A in l2cem tdcem l2adam tdadam; do printf "%s/%s=%s " "$A" "$B" "${RD[$A,$B]}"; done; done)"
log "  failures: $(grep -cE "FAIL|NOFIRE" "$SUM") of $(wc -l < "$QB")"
log "CARD_DONE"
