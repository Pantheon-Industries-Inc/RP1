#!/usr/bin/env bash
# FULLY UNIFIED recipe ("uniB", 2026-09-21, user directive: "we want fully unified ...
# mean-weight and actor LR should be unified, expansion, freeze fraction and teacher
# steps also"; actor steps chosen by early stopping). ONE configuration for every
# env x base cell; the only per-env value left is the critic's latent window
# (1 frame; Reacher 2 = velocity observability), an environment property.
#
#   actor    a_max 2.5 | K 8 | H 5 | mean-weight 0.3 | actor-lr 3e-4 (cosine -> 1/10)
#            batch 256 | steps cap 12000 (2026-09-22; 8000 bound in 14/38 rows), snapshot every 2000, best on draws 48-51
#            max-delta 20 | p_cross 0.3 | anti-constancy 0.5 | replay 0 | expansion 0
#   co-critic MRN depth 2 | gamma 0.99 | n-step 1 | expectile 0.1->0.03 | lr 1e-3->1e-4
#            freeze 0.8 | EMA 0.005 | TD batch 1024
#   teacher  MRN depth 2 | gamma 0.99 | n-step 1 | expectile 0.03 | 12000 steps | p_cross 0.3
#   eval     held-out 8000:10000 | report 42/43/44 | both horizons (tworoom/cube)
#
# Provenance of the shared values: mw 0.3 / lr 3e-4 = joint winner of the Aug-6 open-loop
# sweep over all 5 base x env combos; expansion 0 = better in every PushT pair, catastrophic
# with thin windows on Reacher, inert-when-frozen was free on Cube; teacher 12k = the Reacher
# 2x2 (12k >> 6k) and TwoRoom frzB12k > frzB6k; freeze 0.8 = TwoRoom's value (Cube/Reacher
# are insensitive to freezing); gamma 0.99 = user decision for h100, applied to both horizons
# so the recipe stays single (the config-B gamma jobs show whether h25 tolerates it).
# Teachers are PER SEED in every env (TD_PER_SEED / RS_TD_PER_SEED; TwoRoom via SPLIT=1):
# the paper protocol is "fresh actor and critic seeds". Untested-before choices flagged: replay 0 on Cube/Reacher, batch 256 on TwoRoom,
# expectile 0.03 teacher on TwoRoom.
#
# JOINT EARLY STOPPING (user 2026-09-21, "the critic should also be able to do early
# stopping ... even more granular"): the teacher trains once to TDS=18000 saving a snapshot
# every 3000 steps; every snapshot (3k..15k) plus the final gets its own actor row
# (unig_ctrl_a2.5_t<N> / rs_x0_r0_t<N>), each with the usual actor ES (every 2000 steps on
# draws 48-51). The (teacher, actor) pair is then chosen per cell on eval/ckv5 h25/h100
# selection scores -- never on the report draws.
#
#   HZ=25 ENVS=cube bash scripts/sky/unigamma/launch_uniB.sh    # h25 stack (gamma_h25), names rlp-cu-uniJ-h25*
#   HZ=100 ENVS=cube bash scripts/sky/unigamma/launch_uniB.sh   # h100 stack (gamma_h100)
#   SEEDS="3 4 5" SFX=-s345 bash scripts/sky/unigamma/launch_uniB.sh
set -euo pipefail
export PATH="$HOME/.sky/bin:$PATH"
RECIPE=${RECIPE:-scripts/sky/unigamma/recipes/uniJ.yaml}   # THE recipe: every shared value lives there
eval "$(python3 scripts/sky/unigamma/recipes/recipe_env.py "$RECIPE")"   # section.key -> UNI_<KEY>
DRY=${DRY:-0}              # 1 = print the resolved launch commands, launch nothing
CUBE_LOCK=${CUBE_LOCK:-gpu}   # cube eval serialization: gpu (default; A/B 2026-09-22 bit-exact, +8% time) | none (2 evals per GPU bit-exact, +12%) | node (old, 1 eval per job)
[ "$DRY" = 1 ] && exec 3>&1   # fd 3 escapes the per-launch "2>&1 | { grep -E "sky jobs logs [0-9]+|Submitted sky.jobs.launch request|[Ee]rror|rror:" || true; } | tail -1" filters
sky(){ if [ "$DRY" = 1 ]; then printf "%s\n" "$@" | grep -vx -- --env | sed "1,4d" >&3; else command sky "$@"; fi; }
DATE=${DATE:-20260921}
SEEDS=${SEEDS:-$UNI_SEEDS}
SFX=${SFX:-}
TDS=${TDS:-$UNI_TD_STEPS}  # teacher steps CAP; with TD_SNAPS every snapshot below it is also an actor row
TSNAP=${TSNAP:-$UNI_TD_SNAP}  # teacher snapshot interval (joint teacher x actor early stopping)
TSNAPS=${TSNAPS:-$(seq -s " " $TSNAP $TSNAP $((TDS-TSNAP)))}   # "3000 6000 9000 12000 15000"
TSFX=${TSFX:-}
# per-horizon critic (user 2026-09-21): HZ=25 -> gamma_h25, evaluate+select at h25 only;
# HZ=100 -> gamma_h100, h100 only. Reacher is h25 by construction (gamma_h25). Job names get -h25/-h100.
HZ=${HZ:-100}
case "$HZ" in
  25)  GAMMA=${GAMMA:-$UNI_GAMMA_H25};  OFFS=25;  HSFX=-h25 ;;
  100) GAMMA=${GAMMA:-$UNI_GAMMA_H100}; OFFS=100; HSFX=-h100 ;;
  both) GAMMA=${GAMMA:-$UNI_GAMMA_H100}; OFFS=$UNI_OFFSETS; HSFX="" ;;   # legacy single-stack layout
  *) echo "HZ must be 25, 100 or both"; exit 2 ;;
esac
CVSFX=${CVSFX:--uniJ-g${GAMMA#0.}-e${UNI_TD_EXPECTILE#0.}-t$((TDS/1000))k}   # teacher lives in the cache dir => the key MUST carry the teacher signature
PRIO=${PRIO:-p1}
# WORKSPACE (optional): submit into a specific SkyPilot workspace / quota rather than the client default.
WSARG=""; [ -n "${WORKSPACE:-}" ] && WSARG="--workspace $WORKSPACE"
GPUS=${GPUS:-4}            # GPUs per job (8 = one whole node)
SLOTS=${SLOTS:-}           # parallel actor slots (default = GPUS; up to 2*GPUS on pods = 2 actors/GPU); tworoom only
ENVS=${ENVS:-"tworoom cube reacher"}
USERV=armin@pantheon.inc
COMMON=(--env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer --env PANTHEON_USER=$USERV
        --env TRAIN_SEEDS="$SEEDS" --env MAXPAR=$GPUS)
UNI=(--env UNIG_MW=$UNI_MW --env UNIG_LR=$UNI_LR --env UNIG_SCHED=uni --env UNIG_MD=$UNI_MD --env UNIG_K=$UNI_K
     --env UNI_FREEZE=$UNI_FREEZE --env UNI_EXPECTILE=$UNI_EXPECTILE --env UNI_EXPECTILE_FINAL=$UNI_EXPECTILE_FINAL
     --env UNI_CRITIC_LR=$UNI_CRITIC_LR --env UNI_CRITIC_LR_FINAL=$UNI_CRITIC_LR_FINAL --env UNI_EXPAND=$UNI_EXPAND --env UNI_REPLAY=$UNI_REPLAY
     --env CRIT_DEPTH=$UNI_CRIT_DEPTH --env UNIG_WIDTH=$UNI_REFINER_WIDTH --env UNIG_LAYERS=$UNI_REFINER_LAYERS --env ACR_LAM=$UNI_ACR --env UNIG_PC=$UNI_PC --env TR_NSTEP=$UNI_NSTEP
     --env TDB=$UNI_TD_BATCH --env EMA_TAU=$UNI_EMA_TAU
     --env STEPS=$UNI_STEPS --env BATCH=$UNI_BATCH --env TD_STEPS=$TDS --env TD_SAVE_EVERY=$TSNAP --env TD_SNAPS="$TSNAPS" --env RH=$UNI_RH
     --env CKPT_SELECT=1 --env CKPT_EVERY=${CKPT_EVERY:-$UNI_CKPT_EVERY} --env CKPT_VAL_SEEDS="$UNI_VAL_DRAWS"
     --env EVAL_SEEDS="$UNI_REPORT_DRAWS" --env UNIG_OFF="$OFFS" --env EVAL_RAWDIR=volume --env REPORT_EVAL=${REPORT_EVAL:-0})

for ENVN in $ENVS; do
  # reacher_gamma.yaml names the LeWM base "lejepa" (its wm_base artifact is reacher_base_lewm,
  # selected by BASE == lejepa); cube uses "lewm". BASE=lewm on reacher fails at the tarball.
  case $ENVN in cube) BASES="lewm pldm";; pusht) BASES="lewm${PT_PLDM_WM:+ pldm}";; *) BASES="lejepa pldm";; esac
  BASES=${ONLY_BASES:-$BASES}
  for BASE in $BASES; do
    P=""; [ "$BASE" = pldm ] && P="-p"
    case $ENVN in
      tworoom)
        NAME="rlp-tw-uniJ${HSFX}${TSFX}${P}${SFX}"; TAG="${NAME}-${DATE}"
        TW_ENVS=(ENVNAME=tworoom BASE=$BASE GRID=unig ONLYCFG=unig_ctrl_a2.5
          SPLIT=1 SMOKE=${SMOKE:-0} INCLUDE_WINNERS=0 REUSE_ONLY=0 ACTOR_ONLY=0
          STAGED=0 ACTOR_IMPORT_TAG= FULLCACHE=0 SHARD_INDEX=0 SHARD_COUNT=1
          TR_GAMMA=$GAMMA TW_TD_EXPECTILE=$UNI_TD_EXPECTILE TW_REPLAY=$UNI_REPLAY TD_PER_SEED=$UNI_TD_PER_SEED
          CACHE_VERSION="tw-v2${P}-n1s2-v1${CVSFX}" EXPERIMENT_TAG="$TAG"
          UNIG_MW=$UNI_MW UNIG_LR=$UNI_LR UNIG_SCHED=uni UNIG_MD=$UNI_MD UNIG_K=$UNI_K
          UNI_FREEZE=$UNI_FREEZE UNI_EXPECTILE=$UNI_EXPECTILE UNI_EXPECTILE_FINAL=$UNI_EXPECTILE_FINAL
          UNI_CRITIC_LR=$UNI_CRITIC_LR UNI_CRITIC_LR_FINAL=$UNI_CRITIC_LR_FINAL UNI_EXPAND=$UNI_EXPAND UNI_REPLAY=$UNI_REPLAY
          CRIT_DEPTH=$UNI_CRIT_DEPTH UNIG_WIDTH=$UNI_REFINER_WIDTH UNIG_LAYERS=$UNI_REFINER_LAYERS ACR_LAM=$UNI_ACR UNIG_PC=$UNI_PC TR_NSTEP=$UNI_NSTEP
          TDB=$UNI_TD_BATCH EMA_TAU=$UNI_EMA_TAU
          STEPS=$UNI_STEPS BATCH=$UNI_BATCH TD_STEPS=$TDS TD_SAVE_EVERY=$TSNAP TD_SNAPS="$TSNAPS" RH=$UNI_RH
          CKPT_SELECT=1 CKPT_EVERY=${CKPT_EVERY:-$UNI_CKPT_EVERY} CKPT_VAL_SEEDS="$UNI_VAL_DRAWS"
          EVAL_SEEDS="$UNI_REPORT_DRAWS" UNIG_OFF="$OFFS" EVAL_RAWDIR=volume
          WANDB_PROJECT=RLP WANDB_ENTITY=armin-sommer PANTHEON_USER=$USERV
          TRAIN_SEEDS="$SEEDS" MAXPAR=${SLOTS:-$GPUS} PAR_CKV=${PAR_CKV:-1} REPORT_EVAL=${REPORT_EVAL:-0})
        if [ -n "${POD:-}" ]; then
          # POD=host:port -> run the same yaml on a plain SSH host (RunPod) via the pod runner
          echo "==> tworoom / $BASE  ($NAME) on POD $POD"
          ENVARGS=(); for kv in "${TW_ENVS[@]}"; do ENVARGS+=(--env "$kv"); done
          $([ "$DRY" = 1 ] && echo echo) python3 scripts/sky/pod/run_yaml_on_pod.py --host "${POD%%:*}" --port "${POD##*:}" --key "${POD_KEY:-$HOME/.ssh/id_ed25519}" \
            --yaml scripts/sky/unigamma/tworoom_g98_rescue.yaml --name "$NAME" --pre "${POD_PRE:-}" ${POD_HOME:+--home "$POD_HOME"} ${POD_NOSTART:+--no-start} "${ENVARGS[@]}"
        else
          echo "==> tworoom / $BASE  ($NAME)"
          ENVARGS=(); for kv in "${TW_ENVS[@]}"; do ENVARGS+=(--env "$kv"); done
          sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$NAME" --priority $PRIO $WSARG --gpus H200:$GPUS -y --async \
            "${ENVARGS[@]}" 2>&1 | { grep -E "sky jobs logs [0-9]+|Submitted sky.jobs.launch request|[Ee]rror|rror:" || true; } | tail -1
        fi ;;
      cube)
        # DYNA_WM=<volume dir with weights.pt+config.json> DYNA_IT=<k>: run the same grid on a Dyna-finetuned WM
        # (name/tag/cache get a -dyna<k> suffix; the base's own caches are never reused -- latents are WM-specific)
        DSFX=""; [ -n "${DYNA_WM:-}" ] && DSFX="-dyna${DYNA_IT:?DYNA_WM needs DYNA_IT}"
        NAME="rlp-cu-uniJ${HSFX}${TSFX}${P}${DSFX}${SFX}"; TAG="${NAME}-${DATE}"
        CU_ENVS=(--env ENVNAME=cube --env BASE=$BASE --env GRID=unig --env ONLYCFG=unig_ctrl_a2.5
          --env SPLIT=0 --env SMOKE=${SMOKE:-0} --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0
          --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 --env SHARD_INDEX=0 --env SHARD_COUNT=1
          --env CU_DYNA=0 --env CU_DYNA_WM="${DYNA_WM:-}" --env CU_GAMMA=$GAMMA --env CU_TD_EXPECTILE=$UNI_TD_EXPECTILE
          --env CU_EXPAND=$UNI_EXPAND --env CU_REPLAY=$UNI_REPLAY --env TD_PER_SEED=$UNI_TD_PER_SEED
          --env CACHE_VERSION="cu-v2${P}-n1s2-v1${CVSFX}${DSFX}" --env EXPERIMENT_TAG="$TAG"
          --env CUBE_LOCK=${CUBE_LOCK:-node}
          --env JANITOR_PRUNE_PREFIXES="${JANITOR_PRUNE_PREFIXES-rlp-re-uniJ rlp-re-unicrit rlp-tw-uniJ rlp-tw-unicrit}" --env JANITOR_PRUNE_CACHE_PREFIXES="${JANITOR_PRUNE_CACHE_PREFIXES-tw-}"
          "${UNI[@]}" "${COMMON[@]}")
        # DYNA_ROW=<row> DYNA_STEP=<N|final> (with DYNA_WM): retrain RLP with the base cell's WINNING config only --
        # teacher 18k with the single snapshot the row needs, one actor row, the step-N snapshot deployed without
        # selection; evaluated on validation (48-51) AND test (42-44) draws so the iteration count can be chosen on val
        if [ -n "${DYNA_ROW:-}" ]; then
          DT=${DYNA_ROW#unig_ctrl_a2.5}; DT=${DT#_t}
          CU_ENVS+=(--env ONLYCFG="$DYNA_ROW" --env TD_SNAPS="$DT" --env CKPT_SELECT=0 --env DEPLOY_STEP="${DYNA_STEP:?}"
                    --env REPORT_EVAL=1 --env EVAL_SEEDS="$UNI_VAL_DRAWS $UNI_REPORT_DRAWS")
        fi
        # FIXED_ROW=<row> FIXED_IMPORT=<import tag> FIXED_SEEDS="<seeds>": test-draw evaluation of the staged
        # shared-pair snapshots (scripts/sky/tools/fixed_stage.yaml) -- no training, EVAL_TAG=rh5fixed, one row.
        # TD_SNAPS is trimmed so ONLYCFG's prefix match selects exactly that row (bare row = 18k teacher).
        if [ -n "${FIXED_ROW:-}" ]; then
          FT=${FIXED_ROW#unig_ctrl_a2.5}; FT=${FT#_t}
          NAME="${NAME}-fixed"; TAG="${NAME}-${DATE}"
          CU_ENVS+=(--env EXPERIMENT_TAG="$TAG" --env REUSE_ONLY=1 --env ACTOR_IMPORT_TAG="${FIXED_IMPORT:?}" --env REPORT_EVAL=1
                    --env EVAL_TAG=rh5fixed --env ONLYCFG="$FIXED_ROW" --env TD_SNAPS="$FT" --env TRAIN_SEEDS="${FIXED_SEEDS:-$SEEDS}"
                    --env CKPT_SELECT=0 --env MAXPAR=$(echo ${FIXED_SEEDS:-$SEEDS} | wc -w | tr -d ' '))
        fi
        if [ -n "${POD:-}" ]; then
          # POD=host:port -> same yaml on a RunPod SSH host; cube = 1 actor per GPU (evals are GPU-bound), MAXPAR from SLOTS
          echo "==> cube / $BASE  ($NAME) on POD $POD"
          $([ "$DRY" = 1 ] && echo echo) python3 scripts/sky/pod/run_yaml_on_pod.py --host "${POD%%:*}" --port "${POD##*:}" --key "${POD_KEY:-$HOME/.ssh/id_ed25519}" \
            --yaml scripts/sky/unigamma/tworoom_g98_rescue.yaml --name "$NAME" --pre "${POD_PRE:-}" ${POD_HOME:+--home "$POD_HOME"} ${POD_NOSTART:+--no-start} \
            "${CU_ENVS[@]}" --env MAXPAR=${SLOTS:-$GPUS} --env PAR_CKV=${PAR_CKV:-0}
        else
          echo "==> cube / $BASE  ($NAME)"
          # SLOTS > GPUS packs rows 2 per GPU (cube actor training ~50% util / 5% mem, snapshot evals CPU-bound; measured 2026-09-22)
          sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$NAME" --priority $PRIO $WSARG --gpus H200:$GPUS -y --async \
            "${CU_ENVS[@]}" --env MAXPAR=${SLOTS:-$GPUS} 2>&1 | { grep -E "sky jobs logs [0-9]+|Submitted sky.jobs.launch request|[Ee]rror|rror:" || true; } | tail -1
        fi ;;
      pusht)
        # PushT on the model=rlp pixi stack (scripts/sky/pusht_uniJ.yaml); h25 stack only (gamma = UNI_GAMMA_H25).
        # Pod mode ships ONE prepare job + per-seed teacher jobs + per-(seed, teacher-snapshot) row jobs, each with a
        # static GPU (round-robin over PT_GPUS); scripts/sky/pod/gpu_chains.sh runs them as per-GPU sequential chains.
        # Cluster mode: one single-GPU job per seed (MODE=all).
        [ "$HZ" = both ] && { echo "pusht: use HZ=25 or HZ=100 (one stack per launch)" >&2; continue; }
        NAME="rlp-pt-uniJ${HSFX}${TSFX}${P}${SFX}"; TAG="${NAME}-${DATE}"
        # PLDM base: the in-house PushT PLDM WM (epoch 19); on pods the W&B artifact copy at /mnt/raid0/rlp-pusht/wm/pldm_epoch19
        [ "$BASE" = pldm ] && [ -z "${PT_PLDM_WM:-}" ] && [ -n "${POD:-}" ] && PT_PLDM_WM=/mnt/raid0/rlp-pusht/wm/pldm_epoch19
        PT_ENVS=(--env ENVNAME=pusht --env BASE=$BASE --env WM_DIR="${PT_PLDM_WM:-}" --env EXPERIMENT_TAG="$TAG" --env HZ=$HZ
          --env CACHE_TAG="${PT_CACHE_TAG:-rlp-pt-uniJ-cache-$BASE}" --env SMOKE=${SMOKE:-0}
          --env UNI_MW=$UNI_MW --env UNI_LR=$UNI_LR --env UNI_BATCH=$UNI_BATCH --env UNI_STEPS=$UNI_STEPS
          --env UNI_CKPT_EVERY=${CKPT_EVERY:-$UNI_CKPT_EVERY} --env UNI_MD=$UNI_MD --env UNI_PC=$UNI_PC --env UNI_ACR=$UNI_ACR
          --env UNI_REPLAY=$UNI_REPLAY --env UNI_EXPAND=$UNI_EXPAND --env UNI_K=$UNI_K --env UNI_AMAX=$UNI_AMAX
          --env UNI_REFINER_WIDTH=$UNI_REFINER_WIDTH --env UNI_REFINER_LAYERS=$UNI_REFINER_LAYERS
          --env UNI_GAMMA=$GAMMA --env UNI_NSTEP=$UNI_NSTEP --env UNI_EXPECTILE=$UNI_EXPECTILE --env UNI_EXPECTILE_FINAL=$UNI_EXPECTILE_FINAL
          --env UNI_CRITIC_LR=$UNI_CRITIC_LR --env UNI_CRITIC_LR_FINAL=$UNI_CRITIC_LR_FINAL --env UNI_FREEZE=$UNI_FREEZE
          --env UNI_EMA_TAU=$UNI_EMA_TAU --env UNI_TD_BATCH=$UNI_TD_BATCH --env UNI_CRIT_DEPTH=$UNI_CRIT_DEPTH
          --env TD_STEPS=$TDS --env TD_SAVE_EVERY=$TSNAP --env UNI_TD_EXPECTILE=$UNI_TD_EXPECTILE
          --env CKPT_VAL_SEEDS="$UNI_VAL_DRAWS" --env REPORT_DRAWS="$UNI_REPORT_DRAWS" --env EVAL_PAR=${PT_EVAL_PAR:-4}
          --env PANTHEON_USER=$USERV)
        [ "$BASE" = pldm ] && [ -z "${PT_PLDM_WM:-}" ] && { echo "pusht/pldm needs PT_PLDM_WM=<wm dir>" >&2; continue; }
        if [ -n "${POD:-}" ]; then
          PODARGS=(--host "${POD%%:*}" --port "${POD##*:}" --key "${POD_KEY:-$HOME/.ssh/id_ed25519}" --yaml scripts/sky/pusht_uniJ.yaml
                   --home "${POD_HOME:-/root/hpusht}" --no-start --pre "${POD_PRE:-}"
                   --env OUT_ROOT=/mnt/raid0/rlp-pusht --env CACHE_ROOT=/mnt/raid0/rlp-pusht --env RLP_DATA_HOME=/mnt/raid0/rlp-pusht/rlp-data)
          NG=${PT_GPUS:-8}; g=0; JOBLIST=""
          # PT_BASELINE=t<T> -> ship+START one MODE=baseline job per seed (CEM etc. on latent + this cell's locked teacher)
          if [ -n "${PT_BASELINE:-}" ]; then
            RARGS=(); for a in "${PODARGS[@]}"; do [ "$a" = --no-start ] || RARGS+=("$a"); done
            echo "==> pusht / $BASE  ($NAME) on POD $POD: BASELINE ${PT_SOLVERS:-cem} x ${PT_COSTS:-latent value}, teacher $PT_BASELINE, seeds $SEEDS"
            for S in $SEEDS; do
              $([ "$DRY" = 1 ] && echo echo) python3 scripts/sky/pod/run_yaml_on_pod.py "${RARGS[@]}" --name "${NAME}-s${S}-base" --env MODE=baseline --env FIXED_ROW=$PT_BASELINE --env SEED=$S --env GPU=$((S % ${PT_GPUS:-8})) --env BASELINE_SOLVERS="${PT_SOLVERS:-cem}" --env BASELINE_COSTS="${PT_COSTS:-latent value}" "${PT_ENVS[@]}" 2>&1 | grep -E "pod-runner|rror"
            done
            continue
          fi
          # PT_REPORT=t<T>:<STEP> -> ship+START one MODE=report job per seed (test draws of the locked pair), nothing else
          if [ -n "${PT_REPORT:-}" ]; then
            RROW=${PT_REPORT%%:*}; RSTEP=${PT_REPORT##*:}
            RARGS=(); for a in "${PODARGS[@]}"; do [ "$a" = --no-start ] || RARGS+=("$a"); done   # start immediately
            echo "==> pusht / $BASE  ($NAME) on POD $POD: REPORT jobs for $RROW step $RSTEP, seeds $SEEDS"
            for S in $SEEDS; do
              $([ "$DRY" = 1 ] && echo echo) python3 scripts/sky/pod/run_yaml_on_pod.py "${RARGS[@]}" --name "${NAME}-s${S}-report" --env MODE=report --env FIXED_ROW=$RROW --env FIXED_STEP=$RSTEP --env SEED=$S --env GPU=$((S % NG)) "${PT_ENVS[@]}" 2>&1 | grep -E "pod-runner|rror"
            done
            continue
          fi
          echo "==> pusht / $BASE  ($NAME) on POD $POD: prepare + $(echo $SEEDS | wc -w | tr -d ' ') teachers + rows, GPUs 0..$((NG-1))"
          # the code tree is shipped ONCE (with the prepare job); teacher/row jobs share $POD_HOME/sky_workdir
          # PT_ONLY_PREPARE=1 -> ship AND START just the prepare job (pixi env, dataset, WM, caches), nothing else
          if [ "${PT_ONLY_PREPARE:-0}" = 1 ]; then
            RARGS=(); for a in "${PODARGS[@]}"; do [ "$a" = --no-start ] || RARGS+=("$a"); done
            $([ "$DRY" = 1 ] && echo echo) python3 scripts/sky/pod/run_yaml_on_pod.py "${RARGS[@]}" --workdir . --name "${NAME}-prepare" --env MODE=prepare --env SEED=0 --env GPU=${PT_PREP_GPU:-0} "${PT_ENVS[@]}" 2>&1 | grep -E "pod-runner|rror"
            continue
          fi
          # PT_SKIP_PREPARE=1 -> the prepare job was started separately (PT_ONLY_PREPARE); do not overwrite its files
          [ "${PT_SKIP_PREPARE:-0}" = 1 ] || $([ "$DRY" = 1 ] && echo echo) python3 scripts/sky/pod/run_yaml_on_pod.py "${PODARGS[@]}" --workdir . --name "${NAME}-prepare" --env MODE=prepare --env SEED=0 --env GPU=0 "${PT_ENVS[@]}"
          for S in $SEEDS; do
            $([ "$DRY" = 1 ] && echo echo) python3 scripts/sky/pod/run_yaml_on_pod.py "${PODARGS[@]}" --name "${NAME}-s${S}-teacher" --env MODE=teacher --env SEED=$S --env GPU=$((g % NG)) "${PT_ENVS[@]}"
            JOBLIST="$JOBLIST ${NAME}-s${S}-teacher:$((g % NG))"; g=$((g+1))
          done
          for S in $SEEDS; do for T in $TSNAPS; do
            $([ "$DRY" = 1 ] && echo echo) python3 scripts/sky/pod/run_yaml_on_pod.py "${PODARGS[@]}" --name "${NAME}-s${S}-t${T}" --env MODE=row --env ROW=t$T --env SEED=$S --env GPU=$((g % NG)) "${PT_ENVS[@]}"
            JOBLIST="$JOBLIST ${NAME}-s${S}-t${T}:$((g % NG))"; g=$((g+1))
          done; done
          echo "pod job list (name:gpu) for scripts/sky/pod/gpu_chains.sh:$JOBLIST" | tee "/tmp/${NAME}-podjobs.txt"
        else
          # cluster: one job per seed; GPUS>1 runs the rows in parallel (MODE=all_par), 1 GPU = sequential (MODE=all);
          # PT_REPORT=t<T>:<STEP> -> MODE=report jobs (test draws of the locked pair) instead, 1 GPU each
          PM=all; [ "${GPUS:-1}" -gt 1 ] && PM=all_par
          if [ -n "${PT_REPORT:-}" ]; then
            RROW=${PT_REPORT%%:*}; RSTEP=${PT_REPORT##*:}
            for S in $SEEDS; do
              echo "==> pusht / $BASE  ($NAME-s$S-report, $RROW step $RSTEP)"
              sky jobs launch scripts/sky/pusht_uniJ.yaml -n "$NAME-s$S-report" --priority $PRIO $WSARG --gpus H200:1 -y --async \
                --env MODE=report --env FIXED_ROW=$RROW --env FIXED_STEP=$RSTEP --env SEED=$S --env GPU=0 \
                --env OUT_ROOT=/newcheckpoints --env CACHE_ROOT=/newcheckpoints --env RLP_DATA_HOME=/tmp/rlp-data "${PT_ENVS[@]}" 2>&1 | { grep -E "Submitted|[Ee]rror|rror:" || true; } | tail -1
            done
            continue
          fi
          for S in $SEEDS; do
            echo "==> pusht / $BASE  ($NAME-s$S, MODE=$PM, ${GPUS:-1} GPUs, wm ${PT_PLDM_WM:-official})"
            sky jobs launch scripts/sky/pusht_uniJ.yaml -n "$NAME-s$S" --priority $PRIO $WSARG --gpus H200:${GPUS:-1} --cpus 40+ -y --async \
              --env MODE=$PM --env SEED=$S --env GPU=0 --env EVAL_PAR=${PT_EVAL_PAR:-4} \
              --env OUT_ROOT=/newcheckpoints --env CACHE_ROOT=/newcheckpoints --env RLP_DATA_HOME=/tmp/rlp-data "${PT_ENVS[@]}" 2>&1 | { grep -E "sky jobs logs [0-9]+|Submitted sky.jobs.launch request|[Ee]rror|rror:" || true; } | tail -1
          done
        fi ;;
      reacher)
        NAME="rlp-re-uniJ${HSFX}${TSFX}${P}${SFX}"; TAG="${NAME}-${DATE}"   # -h25: reacher is h25 by construction
        echo "==> reacher / $BASE  ($NAME)"
        sky jobs launch scripts/sky/unigamma/reacher_gamma.yaml -n "$NAME" --priority $PRIO $WSARG --gpus H200:$GPUS -y --async \
          --env BASE=$BASE --env GRID=cross --env AMFIX=$UNI_AMAX --env CROSS_MW=$UNI_MW --env CROSS_LR=$UNI_LR \
          --env CROSS_EXPANDS="$UNI_EXPAND" --env CROSS_REPLAYS="$UNI_REPLAY" --env RS_FREEZE=$UNI_FREEZE --env RS_TD_PER_SEED=$UNI_TD_PER_SEED \
          --env UNI_EXPECTILE=$UNI_EXPECTILE --env UNI_EXPECTILE_FINAL=$UNI_EXPECTILE_FINAL --env UNI_CRITIC_LR=$UNI_CRITIC_LR --env UNI_CRITIC_LR_FINAL=$UNI_CRITIC_LR_FINAL \
          --env SMOKE=${SMOKE:-0} --env REUSE_ONLY=0 --env ACTOR_ONLY=0 --env ACTOR_IMPORT_TAG="" \
          --env RS_GAMMA=$UNI_GAMMA_H25 --env RS_NSTEP=$UNI_NSTEP --env CRIT_DEPTH=$UNI_CRIT_DEPTH --env RS_WIDTH=$UNI_REFINER_WIDTH --env RS_LAYERS=$UNI_REFINER_LAYERS --env RS_MD=$UNI_MD --env RS_K=$UNI_K \
          --env RS_EMA_TAU=$UNI_EMA_TAU --env ACR_LAM=$UNI_ACR --env VALUE_FRAMES=$UNI_WINDOW_REACHER --env VALUE_EXPECTILE=$UNI_TD_EXPECTILE \
          --env RS_TD_STEPS=$TDS --env RS_TD_SAVE_EVERY=$TSNAP --env RS_TD_SNAPS="$TSNAPS" --env RS_TD_PC=$UNI_PC \
          --env STEPS=$UNI_STEPS --env BATCH=$UNI_BATCH --env CKPT_EVERY=${CKPT_EVERY:-$UNI_CKPT_EVERY} --env RS_CKPT_SELECT=1 --env RS_CKPT_VAL_SEEDS="$UNI_VAL_DRAWS" \
          --env RS_LATCHED=1 --env FINAL_ONLY=1 --env HELD_AT_END=0 --env REPORT_DRAWS="$UNI_REPORT_DRAWS" \
          --env SUCCESS_THRESHOLDS="0.1 0.05" --env PLANNER_RH=$UNI_RH \
          --env EXPERIMENT_TAG="$TAG" \
          "${COMMON[@]}" 2>&1 | { grep -E "sky jobs logs [0-9]+|Submitted sky.jobs.launch request|[Ee]rror|rror:" || true; } | tail -1 ;;
    esac
  done
done
echo "collect: python3 scripts/wandb_collect_rlp.py 'uniJ' --reacher"
