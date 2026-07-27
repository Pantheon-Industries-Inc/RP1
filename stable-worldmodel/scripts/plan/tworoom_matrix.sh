#!/bin/bash
# TwoRoom full planner x cost matrix on the author-released pretrained bases,
# run under the AUTHORS' protocol so the numbers are comparable to the LeWM
# paper's baseline table (docs/baselines.md).
#
# Scope: the bases exactly as shipped. No Dyna / on-policy fine-tuning arm --
# a post-Dyna column would need one fine-tune per planner (each arm on its own
# rollouts, or the comparison is confounded by whichever planner's state
# distribution the WM was adapted to), which is out of scope here.
#
# PROTOCOL -- what makes this paper-comparable (and what does not):
#   config      scripts/plan/config/tworoom.yaml   <- the AUTHORS' plan config
#                 (state keys pos_agent/goal_pos_agent, keys_to_cache
#                 [action, proprio] so the DINO proprio variant is fed)
#               NOT tworoom_lewm.yaml, which points at our noised
#               tworoom_play.lance and uses state/goal_state.
#   dataset     tworoom.h5 -- both the task draw AND dataset.stats (the action
#               z-score the frozen predictors were trained under).
#   budget      h25 = offset 25 / budget 50. docs/baselines.md states the
#               benchmark uses a fixed 50-step budget, so *** std x h25 is THE
#               paper-comparable cell ***: LeWM 87, PLDM 97, DINO-WM 100.
#   extensions  h50 (offset 50 / budget 100) and the `hard` surface
#               (+eval.cross_wall=true, start and goal on opposite sides of the
#               wall) are OURS -- they have no counterpart in the paper table.
#               Reported, but never compared to 87/97/100.
#
# Card = 12 cells: {std, hard} x eval seed {42,43,44} x {h25, h50}, 50 envs each.
#
# Conditions (7 arms):
#   Latent+CEM   Latent+MPPI   Latent+Adam    (native latent-MSE cost)
#   TD+CEM       TD+MPPI       TD+Adam        (learned MRN quasimetric as cost,
#                                              x3 TD training seeds)
#   LIPv4 tandem                              (x3 actor training seeds)
#
# Usage: tworoom_matrix.sh <base:lejepa|pldm|dinowm> <gpu> [iters] [stages]
#   stages: "anchor" = the 3 paper-comparable Latent+CEM cells only (run this
#           FIRST and check against the reference before spending the matrix);
#           "full" = all 7 arms x 12 cells. Default "anchor full".
#   iters only selects which LIP actors to read (the "k<N>" suffix); train them
#   with tworoom_phase1_matrix.sh first.
# Env overrides: CANON (dataset path).
# Idempotent per cell via results/summary_matrix_tworoom_<base>.csv.
set -u
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/swm_home}
export CODE=${CODE:-/workspace/code/stable-worldmodel}
export PYTHONPATH=$CODE
export HF_HOME=${HF_HOME:-/root/hf}
export TQDM_DISABLE=1
# OMP cap is load-bearing: uncapped threads cost ~40x on TwoRoom evals.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

BASE=$1
GPU=$2
ITERS=${3:-8}
STAGES=${4:-"anchor full"}
SUFF=""; [ "$ITERS" != "8" ] && SUFF="k${ITERS}"

# paper reference for the std x h25 cell (docs/baselines.md TwoRoom column)
# BASE_OVERRIDES: per-base hydra overrides applied to every eval of that base.
#
# dinowm drops `proprio` from keys_to_cache. That is NOT a protocol change --
# it removes a DOUBLE normalization. eval_wm.py fits a StandardScaler for every
# key in keys_to_cache and policy.py applies it to the info dict; DinoWMTokens
# then normalizes proprio AGAIN inside encode() using pro_mu/pro_std, which the
# converter fit from the same dataset. The two transforms are numerically
# identical (mu 111.80/84.99, sigma 36.87/38.19 on both sides), so applying both
# maps every agent position to ~-3: positions 2 px apart arrive identical to 3
# decimals, the goal's position signal is destroyed, and the value sits ~3 sigma
# outside anything the predictor saw in training.
# Measured, same checkpoint/seed/solver: 36.0 double-normalized -> 100.0 single
# (paper reference 100). Only dinowm reads proprio, so the twins are unaffected
# and need no override. This also makes the eval path agree with cache_latents,
# which feeds proprio RAW -- so the TD caches and the eval now see the same
# convention.
BASE_OVERRIDES=()
case "$BASE" in
  lejepa) PAPER=87  ;;   # LeWM
  pldm)   PAPER=97  ;;
  dinowm) PAPER=100      # DINO-WM, proprio variant (the paper's DINO-WM)
          BASE_OVERRIDES=("dataset.keys_to_cache=[action]") ;;
  *) echo "unknown base $BASE (expected lejepa|pldm|dinowm)"; exit 1;;
esac

PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
CKPT="${BASE}_tworoom"
CANON=${CANON:-/workspace/datasets_canon/tworoom/tworoom.h5}
SUM=$RES/summary_matrix_tworoom_${BASE}${SUFF}.csv
DRV=$LOGS/driver_matrix_tworoom_${BASE}${SUFF}.log
EVAL_TIMEOUT=${EVAL_TIMEOUT:-14400}

mkdir -p "$LOGS" "$RES"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][mx-${BASE}] $*" | tee -a "$DRV"; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }
[ -f "$CANON" ] || { log "FATAL: canonical dataset $CANON not found"; exit 1; }

ev(){ # name seed offset budget surface extra...
  local nm=$1 seed=$2 off=$3 bud=$4 surface=$5; shift 5
  grep -q "^${nm}," "$SUM" && { log "eval ${nm}: cached ($(sc "$nm"))"; return 0; }
  local hard=()
  [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  # Log and artifact paths MUST carry $BASE. Cell names are base-agnostic
  # (mx_anchor_..._s42) and only the CSV was per-base, so running bases
  # concurrently had every one of them writing AND parsing the same
  # eval_<cell>.log -- the score was read from whichever process wrote last.
  # That silently returned byte-identical numbers for different checkpoints.
  local elog="$LOGS/eval_${BASE}${SUFF}_${nm}.log"
  CUDA_VISIBLE_DEVICES=$GPU timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name tworoom policy="$CKPT" \
    eval.dataset_name="$CANON" dataset.stats="$CANON" \
    seed="$seed" eval.goal_offset_steps="$off" eval.eval_budget="$bud" \
    solver.batch_size=10 output.filename="${BASE}${SUFF}_${nm}.txt" \
    "${BASE_OVERRIDES[@]}" "${hard[@]}" "$@" \
    > "$elog" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$elog" | tail -1 | grep -oE "[0-9.]+$")
  # Cross-check: the log we just parsed must be the one we just wrote.
  if [ -n "${sr:-}" ] && ! grep -q "policy=${CKPT}\|${CKPT}" "$elog" 2>/dev/null; then
    log "eval ${nm}: WARNING parsed log does not mention ${CKPT}"
  fi
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "eval ${nm}: ${sr:-FAIL}"
}

card(){ # tag extra...  -> the 12 cells
  local tag=$1; shift
  for surface in std hard; do
    for seed in 42 43 44; do
      ev "mx_${tag}_${surface}_h25_s${seed}" "$seed" 25 50  "$surface" "$@"
      ev "mx_${tag}_${surface}_h50_s${seed}" "$seed" 50 100 "$surface" "$@"
    done
  done
  mean_of "mx_${tag}_"          "CARD ${tag} (12-cell)"
  mean_of "mx_${tag}_std_h25_"  "  paper-cell ${tag} (std h25)"
}

mean_of(){ # row-name-prefix label  -> mean over every matching cached cell
  local pre=$1 label=$2 t=0 n=0 v
  while IFS=, read -r k v; do
    case "$k" in "$pre"*) ;; *) continue;; esac
    { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && continue
    t=$(awk "BEGIN{print $t+$v}"); n=$((n+1))
  done < "$SUM"
  [ "$n" -gt 0 ] && log "${label}: mean $(awk "BEGIN{printf \"%.1f\", $t/$n}") over n=${n} cells"
}

log "===== matrix: base ${BASE} (ckpt ${CKPT}), GPU ${GPU}, K=${ITERS}"
log "dataset $CANON (authors' tworoom.h5); config tworoom.yaml; paper std-h25 ref = ${PAPER}"

for st in $STAGES; do
case "$st" in

# ---- ANCHOR: the paper-comparable cell only. 3 evals. Validates that the
# harness reproduces the published number BEFORE 180 more are spent.
anchor)
  log "--- ANCHOR: Latent+CEM, std, h25 (budget 50) -- the docs/baselines.md protocol"
  for seed in 42 43 44; do
    ev "mx_anchor_latent_cem_std_h25_s${seed}" "$seed" 25 50 std solver=cem
  done
  mean_of "mx_anchor_latent_cem_std_h25_" "ANCHOR ${BASE} Latent+CEM std h25"
  log "ANCHOR vs paper: reference ${PAPER} for ${BASE}. A large gap means the"
  log "  protocol still diverges (dataset convention / action stats / state key)"
  log "  -- diagnose before running the full matrix."
  ;;

# ---- FULL: 7 arms x 12 cells
full)
  # native latent-MSE cost x 3 solvers
  for slv in cem mppi adam; do
    card "latent_${slv}" solver=$slv
  done

  # learned TD quasimetric as cost x 3 solvers x 3 TD training seeds
  for slv in cem mppi adam; do
    for ts in 0 1 2; do
      TD=$MET/td_canon_${BASE}_e0.1_n50_s${ts}.pt
      [ -f "$TD" ] || { log "missing TD $TD -- run tworoom_phase1_matrix.sh"; continue; }
      card "td_${slv}_t${ts}" solver=$slv "+metric=$TD"
    done
    mean_of "mx_td_${slv}_t" "CARD td_${slv}_3seed"
  done

  # LIPv4 tandem x 3 actor training seeds
  for s in 0 1 2; do
    A=$ACT/trm_canon_${BASE}_v4${SUFF}_s${s}.pt
    [ -f "$A" ] || { log "missing actor $A -- run tworoom_phase1_matrix.sh"; continue; }
    card "lip_s${s}" solver=lip "solver.actor_path=$A"
  done
  mean_of "mx_lip_s" "CARD lip_3seed"
  ;;

*) log "unknown stage $st" ;;
esac
done

log "===== ${BASE} MATRIX SUMMARY"
sort "$SUM" >> "$DRV"
grep -E "CARD |ANCHOR " "$DRV" | tail -40
log "MATRIX_${BASE}${SUFF}_DONE"
