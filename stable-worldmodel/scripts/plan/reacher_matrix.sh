#!/bin/bash
# Reacher full planner x cost matrix on the CANONICAL LeWM dataset, pre- and
# post-Dyna, for LeWM(lejepa) and PLDM.
#
# Protocol (fixed everywhere): canonical quentinll/lewm-reacher reacher.h5,
# task qpos_match, repo-default plan_config (horizon 5 / receding 5 /
# action_block 5), card = {h25 = offset 25 budget 50, h50 = offset 50 budget
# 100} x eval seeds {42,43,44}, n=50/cell. Reference: LeWM Latent+CEM = 75.3.
#
# Conditions per (base, wm-version):
#   Latent+CEM  Latent+MPPI  Latent+Adam      (native latent cost)
#   TD+CEM      TD+MPPI      TD+Adam          (learned TD value as cost)
#   LIP v4 tandem  (3 actor seeds, reported as the 3-seed mean)
# wm-version: pre = converted author base; post = Dyna round-1 fine-tuned WM
# (80/20 arm). post uses the FRESH TD + FRESH actors trained on that WM.
#
# Usage: reacher_matrix.sh <base:lejepa|pldm> <gpu>
# Idempotent per row via results/summary_matrix_<base>.csv.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export TQDM_DISABLE=1 MUJOCO_GL=osmesa
export OMP_NUM_THREADS=18 MKL_NUM_THREADS=18
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")

BASE=$1
GPU=$2
# STAGES: which blocks to run. "pre" = all 7 conditions on the author base.
# "postlip" = ONLY LIP on the LIP-Dyna WM (the sole valid post row for that WM,
# since its fine-tune consumed LIP rollouts). Per-planner Dyna for the sampling
# baselines lives in reacher_dyna_perplanner.sh — each planner must be
# fine-tuned on ITS OWN rollouts or the comparison is confounded.
STAGES=${3:-"pre postlip"}
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
PLAN=/workspace/stable-worldmodel/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
CK=/workspace/swm_home/checkpoints
SUM=$RES/summary_matrix_${BASE}.csv
DRV=$LOGS/driver_matrix_${BASE}.log
mkdir -p "$LOGS" "$RES"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][mx-${BASE}] $*" | tee -a "$DRV"; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

# ---- per-base asset map: pre-Dyna and post-Dyna (Dyna r1, 80/20 arm)
case "$BASE" in
  lejepa)
    WM_PRE=$CK/lejepa_reacher;               TD_PRE=$MET/td_lejepa_pre    # + _s{0,1,2}.pt
    ACT_PRE=$ACT/lip4_reacher_lejepa_a22     # + _s{0,1,2}.pt
    WM_POST=$CK/dyna_reacher_r1_8020_final;  TD_POST=$MET/td_lejepa_post
    ACT_POST=$ACT/lip4_reacher_r1_8020
    ;;
  pldm)
    WM_PRE=$CK/pldm_reacher;                 TD_PRE=$MET/td_pldm_pre
    ACT_PRE=$ACT/lip4_reacher_pldm_a22
    WM_POST=$CK/dyna_reacher_r1_pldm_8020_final; TD_POST=$MET/td_pldm_post
    ACT_POST=$ACT/lip4_reacher_r1_pldm
    ;;
  *) echo "unknown base $BASE"; exit 1;;
esac

ev(){ # name wm seed offset budget extra...
  local nm=$1 wm=$2 seed=$3 off=$4 bud=$5; shift 5
  grep -q "^${nm}," "$SUM" && { log "eval ${nm}: cached ($(sc "$nm"))"; return 0; }
  CUDA_VISIBLE_DEVICES=$GPU timeout 14400 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$wm" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    seed="$seed" eval.goal_offset_steps="$off" eval.eval_budget="$bud" \
    solver.batch_size=10 output.filename="${BASE}_${nm}.txt" "$@" \
    > "$LOGS/eval_${BASE}_${nm}.log" 2>&1
  # $BASE in the log path is load-bearing: cell names (mx_pre_latent_cem_h25_s42)
  # are base-agnostic and only $SUM was per-base, so running two bases at once
  # had both writing AND parsing the same eval_<cell>.log -- each read whichever
  # process wrote last. Same defect found and fixed in tworoom_matrix.sh, where
  # it made two checkpoints with 4x different fidelity report identical scores.
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${BASE}_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "eval ${nm}: ${sr:-FAIL}"
}

card(){ # tag wm extra...   -> 6 cells + mean
  local tag=$1 wm=$2; shift 2
  for seed in 42 43 44; do
    ev "mx_${tag}_h25_s${seed}" "$wm" "$seed" 25 50  "$@"
    ev "mx_${tag}_h50_s${seed}" "$wm" "$seed" 50 100 "$@"
  done
  local t=0 n=0 v
  for seed in 42 43 44; do for h in h25 h50; do
    v=$(sc "mx_${tag}_${h}_s${seed}"); { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && continue
    t=$(awk "BEGIN{print $t+$v}"); n=$((n+1)); done; done
  [ "$n" -gt 0 ] && log "CARD ${tag}: 6-cell mean $(awk "BEGIN{printf \"%.1f\", $t/$n}") (n=$n)"
}

run_version(){ # ver wm td actprefix
  local ver=$1 wm=$2 td=$3 actp=$4
  log "===== ${ver}: WM=$(basename "$wm")"
  # native latent cost x 3 solvers
  card "${ver}_latent_cem"  "$wm" solver=cem
  card "${ver}_latent_mppi" "$wm" solver=mppi
  card "${ver}_latent_adam" "$wm" solver=adam
  # learned TD value as cost x 3 solvers x 3 TD TRAINING SEEDS
  for slv in cem mppi adam; do
    for ts in 0 1 2; do
      local tdf="${td}_s${ts}.pt"
      [ -f "$tdf" ] || { log "missing TD $tdf"; continue; }
      card "${ver}_td_${slv}_t${ts}" "$wm" solver=$slv "+metric=$tdf"
    done
    # 3-TD-seed aggregate for this solver
    local tt=0 tn=0 tv
    for ts in 0 1 2; do for seed in 42 43 44; do for h in h25 h50; do
      tv=$(sc "mx_${ver}_td_${slv}_t${ts}_${h}_s${seed}")
      { [ -z "$tv" ] || [ "$tv" = "FAIL" ]; } && continue
      tt=$(awk "BEGIN{print $tt+$tv}"); tn=$((tn+1)); done; done; done
    [ "$tn" -gt 0 ] && log "CARD ${ver}_td_${slv}_3seed: mean $(awk "BEGIN{printf \"%.1f\", $tt/$tn}") (n=$tn)"
  done
  # LIP v4 tandem, 3 actor seeds
  for s in 0 1 2; do
    local a="${actp}_s${s}.pt"
    [ -f "$a" ] || { log "missing actor $a"; continue; }
    card "${ver}_lip_s${s}" "$wm" solver=lip "solver.actor_path=$a"
  done
  # LIP 3-seed aggregate
  local t=0 n=0 v
  for s in 0 1 2; do for seed in 42 43 44; do for h in h25 h50; do
    v=$(sc "mx_${ver}_lip_s${s}_${h}_s${seed}"); { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && continue
    t=$(awk "BEGIN{print $t+$v}"); n=$((n+1)); done; done; done
  [ "$n" -gt 0 ] && log "CARD ${ver}_lip_3seed: mean $(awk "BEGIN{printf \"%.1f\", $t/$n}") (n=$n)"
}

for st in $STAGES; do
  case "$st" in
    pre) run_version pre "$WM_PRE" "$TD_PRE" "$ACT_PRE" ;;
    postlip)
      log "===== postlip: LIP only on $(basename "$WM_POST") (its Dyna used LIP rollouts)"
      for s in 0 1 2; do
        a="${ACT_POST}_s${s}.pt"; [ -f "$a" ] || { log "missing actor $a"; continue; }
        card "postlip_lip_s${s}" "$WM_POST" solver=lip "solver.actor_path=$a"
      done
      t=0; n=0
      for s in 0 1 2; do for seed in 42 43 44; do for h in h25 h50; do
        v=$(sc "mx_postlip_lip_s${s}_${h}_s${seed}"); { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && continue
        t=$(awk "BEGIN{print $t+$v}"); n=$((n+1)); done; done; done
      [ "$n" -gt 0 ] && log "CARD postlip_lip_3seed: mean $(awk "BEGIN{printf \"%.1f\", $t/$n}") (n=$n)" ;;
    *) log "unknown stage $st" ;;
  esac
done

log "===== ${BASE} MATRIX SUMMARY"
grep "^mx_" "$SUM" | sort | tee -a "$DRV" > /dev/null
grep -E "^CARD|CARD " "$DRV" | tail -30
log "MATRIX_${BASE}_DONE"
