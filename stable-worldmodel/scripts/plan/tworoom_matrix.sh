#!/bin/bash
# TwoRoom full planner x cost matrix on the author-released pretrained bases,
# PRE-Dyna. Companion: tworoom_dyna_perplanner.sh supplies the post-Dyna half.
#
# Protocol (fixed everywhere): env swm/TwoRoom-v1, dataset tworoom_play.lance,
# repo-default plan_config (horizon 5 / receding 5 / action_block 5), 50 envs
# per cell. Card = 12 cells:
#     surface {std, hard(+eval.cross_wall=true)}
#   x eval seed {42, 43, 44}
#   x horizon  {h25 = offset 25 budget 50, h50 = offset 50 budget 100}
# Reference (docs/baselines.md, authors' TwoRoom): DINO-WM 100, PLDM 97, LeWM 87.
#
# Conditions (7 arms):
#   Latent+CEM   Latent+MPPI   Latent+Adam    (native latent-MSE cost)
#   TD+CEM       TD+MPPI       TD+Adam        (learned TD quasimetric as cost,
#                                              x3 TD training seeds)
#   LIPv4 tandem                              (x3 actor training seeds)
#
# Usage: tworoom_matrix.sh <base:lejepa|pldm|dinowm> <gpu> [iters]
#   iters only picks which LIP actors to read (the "k<N>" suffix), it does not
#   train anything -- run tworoom_phase1_matrix.sh first.
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
SUFF=""; [ "$ITERS" != "8" ] && SUFF="k${ITERS}"

case "$BASE" in
  lejepa|pldm|dinowm) ;;
  *) echo "unknown base $BASE (expected lejepa|pldm|dinowm)"; exit 1;;
esac

PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
CKPT="${BASE}_tworoom"
SUM=$RES/summary_matrix_tworoom_${BASE}${SUFF}.csv
DRV=$LOGS/driver_matrix_tworoom_${BASE}${SUFF}.log
EVAL_TIMEOUT=${EVAL_TIMEOUT:-14400}

mkdir -p "$LOGS" "$RES"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][mx-${BASE}] $*" | tee -a "$DRV"; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

ev(){ # name seed offset budget surface extra...
  local nm=$1 seed=$2 off=$3 bud=$4 surface=$5; shift 5
  grep -q "^${nm}," "$SUM" && { log "eval ${nm}: cached ($(sc "$nm"))"; return 0; }
  local hard=()
  [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  CUDA_VISIBLE_DEVICES=$GPU timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name tworoom_lewm policy="$CKPT" \
    seed="$seed" eval.goal_offset_steps="$off" eval.eval_budget="$bud" \
    solver.batch_size=10 output.filename="${nm}.txt" \
    "${hard[@]}" "$@" \
    > "$LOGS/eval_${nm}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
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
  mean_of "mx_${tag}_" "CARD ${tag}"
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

log "===== PRE-Dyna matrix: base ${BASE} (ckpt ${CKPT}), GPU ${GPU}, K=${ITERS}"

# ---- native latent-MSE cost x 3 solvers
for slv in cem mppi adam; do
  card "pre_latent_${slv}" solver=$slv
done

# ---- learned TD quasimetric as cost x 3 solvers x 3 TD training seeds
for slv in cem mppi adam; do
  for ts in 0 1 2; do
    TD=$MET/td_${BASE}_e0.1_n50_s${ts}.pt
    [ -f "$TD" ] || { log "missing TD $TD -- run tworoom_phase1_matrix.sh"; continue; }
    card "pre_td_${slv}_t${ts}" solver=$slv "+metric=$TD"
  done
  mean_of "mx_pre_td_${slv}_t" "CARD pre_td_${slv}_3seed"
done

# ---- LIPv4 tandem x 3 actor training seeds
for s in 0 1 2; do
  A=$ACT/trm_${BASE}_v4${SUFF}_s${s}.pt
  [ -f "$A" ] || { log "missing actor $A -- run tworoom_phase1_matrix.sh"; continue; }
  card "pre_lip_s${s}" solver=lip "solver.actor_path=$A"
done
mean_of "mx_pre_lip_s" "CARD pre_lip_3seed"

log "===== ${BASE} PRE MATRIX SUMMARY"
sort "$SUM" >> "$DRV"
grep "CARD " "$DRV" | tail -40
log "MATRIX_PRE_${BASE}${SUFF}_DONE"
