#!/bin/bash
# The planner x cost baseline table, both bases, one protocol.
#
# LIP has been compared against a single bar (Latent+CEM) measured at various
# times. This fills in the grid so the claim "the amortised planner beats
# sampling-based planning" is tested against more than one alternative:
#
#            |  CEM 300x10        Adam 300x10        LIP ~16 rollouts
#   L2 cost  |  the paper's bar   NEW                --
#   TD cost  |  NEW (e0.05)       NEW                measured
#
# Adam is stable_worldmodel.solver.GradientSolver (AdamW, lr 0.1). Its config
# defaults to num_samples 100 / n_steps 30; both are overridden to 300 / 10 so
# it gets EXACTLY the CEM budget (3000 rollouts). Comparing planners at unequal
# budgets would confound planner quality with compute.
#
# WHY TD+CEM IS RE-MEASURED. The existing numbers (17.7/45.3 lejepa,
# 14.3/35.0 pldm) used the expectile-0.1 window value. The shared recipe is now
# expectile 0.05, so those are off-recipe. L2+CEM is re-run too even though the
# bar is known (44.7/84.3, 39.3/78.3): taking every cell of the table from one
# run removes any doubt about protocol drift between campaigns.
#
# PWM IS NOT HERE. train_pwm_ac.py has no vframes/pad-context support (0 matches
# against 23 in the patched LIP trainer), so it cannot consume the 3-frame
# window value -- it would hit the 576-d shape error that the LIP deploy path
# already had. Porting that patch is a separate change, done before PWM can join
# this table on equal terms.
#
# Concurrency is deliberately modest (4 groups): CEM evals core-dumped under
# heavy contention twice in this campaign, and both times succeeded when re-run
# with fewer in flight. A no-score is recorded as FAIL and reported, never
# averaged in as 0.0.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SUM=/workspace/results/summary_baselines.csv; touch "$SUM"
L=/workspace/logs/baselines; mkdir -p "$L"
REP="42 43 44 45 46 47"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][base] $*"; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
L2=/workspace/metrics/l2window3.pt
td_of(){ echo "/workspace/metrics/window3_${1}_e005.pt"; }

ev(){ local gpu=$1 B=$2 nm=$3 seed=$4; shift 4
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$(wm_of $B)" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    solver.batch_size=10 "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
  local h t10
  h=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")
  t10=$(grep -oE "0\.1rad [0-9.]+" "$L/${nm}.log" | tail -1 | sed -E "s/0\.1rad //")
  [ -z "$h" ] && log "  !! ${nm} no score: $(tail -1 "$L/${nm}.log" | cut -c1-64)"
  echo "${nm},held=${h:-FAIL},held10=${t10:-FAIL}" >> "$SUM"
}
meanof(){ local pre=$1 key=$2; local t=0 n=0 e
  for s in $REP; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"; }
nof(){ grep -c "^${1}[0-9]*,held=[0-9]" "$SUM"; }

CEM="solver=cem solver.n_steps=10"
# 300 samples x 10 steps = the CEM budget exactly
ADAM="solver=adam solver.n_steps=10 solver.num_samples=300"

log "4 arms x 2 bases x 6 reporting seeds = 48 evals"
i=0
for B in lejepa pldm; do
  ( for s in $REP; do ev $((i % 6)) "$B" "bl_${B}_l2cem_s${s}"  $s $CEM  "+metric=$L2"; done ) &
  i=$((i + 1))
  ( for s in $REP; do ev $((i % 6)) "$B" "bl_${B}_tdcem_s${s}" $s $CEM  "+metric=$(td_of $B)"; done ) &
  i=$((i + 1))
  ( for s in $REP; do ev $((i % 6)) "$B" "bl_${B}_l2adam_s${s}" $s $ADAM "+metric=$L2"; done ) &
  i=$((i + 1))
  ( for s in $REP; do ev $((i % 6)) "$B" "bl_${B}_tdadam_s${s}" $s $ADAM "+metric=$(td_of $B)"; done ) &
  i=$((i + 1))
  wait      # one base at a time: 4 concurrent evals, not 8
done

log "===================== PLANNER x COST TABLE ====================="
log "               (held-at-end, reporting seeds 42..47, h25, eval budget 50)"
for B in lejepa pldm; do
  inc10=$([ "$B" = lejepa ] && echo 91.9 || echo "see card")
  log "--- ${B} ---"
  log "  $(printf '%-22s %8s %8s %6s' 'arm' '@0.05' '@0.1' 'n')"
  for a in l2cem tdcem l2adam tdadam; do
    case $a in
      l2cem)  nm="L2  + CEM  (bar)";; tdcem) nm="TD  + CEM";;
      l2adam) nm="L2  + Adam";;       tdadam) nm="TD  + Adam";;
    esac
    log "  $(printf '%-22s %8s %8s %6s' "$nm" "$(meanof bl_${B}_${a}_s held)" "$(meanof bl_${B}_${a}_s held10)" "$(nof bl_${B}_${a}_s)")"
  done
  log "  $(printf '%-22s %8s %8s' 'LIP (~16 rollouts)' "$([ "$B" = lejepa ] && echo 49.8 || echo '--')" "$inc10")"
done
log "BASELINES_DONE"
