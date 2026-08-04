#!/bin/bash
# JOINT LIPv4 SWEEP over BOTH bases (user, 2026-08-01), then Dyna on the winner.
#
# Constraint from the user: TD / dataset knobs (replay-prob, expand-weight,
# expectile) must be SHARED by both bases; LIP knobs (amax, actor-lr) may differ.
#
# Stage 0  PLDM artifacts: fs1/fs5 caches + window values at both expectiles.
# Stage 1  SHARED replay/expand probe on PLDM. On lejepa these were already
#          measured and both hurt badly (selection held: control 56.7,
#          replay0.5 43.3, expand1.0 25.0, both 20.0) -- expand-weight teaches
#          the critic to agree with the actor's world-model exploitation, the
#          failure mode this env is dominated by. The probe checks whether PLDM
#          agrees before the shared setting is fixed for both.
# Stage 2  Per-base LIP sweep at the surviving shared setting:
#            shared   expectile {0.05, 0.1}          (window value retrained per value)
#            per-base amax {1.8, 2.2} x lr {3e-4, 1e-3}
#          = 8 configs x 2 bases x 3 seeds = 48 trainings at 1000 steps.
# Stage 3  Shared expectile chosen by the CROSS-BASE mean (it must serve both);
#          amax/lr chosen per base. Winners carded on reporting seeds.
# Stage 4  Writes the winning recipe per base to /workspace/_BEST_<base>, which
#          the Dyna driver consumes.
#
# Screening on SELECTION seeds {50,51}; cards on REPORTING seeds {42..47}.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
TRM=/workspace/swm_cem/scripts/trm
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
SUM=/workspace/results/summary_joint.csv; touch "$SUM"
L=/workspace/logs/joint; mkdir -p "$L"
SEL="50 51"; REP="42 43 44 45 46 47"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][joint] $*"; }

wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }

# ---------------------------------------------------------------- stage 0
for B in lejepa pldm; do
  C1=/workspace/caches/canon_${B}_fs1.pt
  C5=/workspace/caches/canon_${B}_fs5.pt
  if [ ! -f "$C1" ]; then
    log "stage0: fs1 cache for ${B}"
    CUDA_VISIBLE_DEVICES=0 python3 "$TRM/cache_latents.py" --wm "$(wm_of $B)" \
      --dataset "$CANON" --out "$C1" --state-key qpos --batch-size 512 \
      > "$L/cache1_${B}.log" 2>&1 || { log "FATAL fs1 ${B}"; exit 1; }
  fi
  [ -f "$C5" ] || CUDA_VISIBLE_DEVICES=0 python3 "$TRM/subsample_cache.py" \
    --in "$C1" --out "$C5" --frameskip 5 > "$L/cache5_${B}.log" 2>&1
  for EX in 005 01; do
    exv=0.05; [ "$EX" = "01" ] && exv=0.1
    W=/workspace/metrics/window3_${B}_e${EX}.pt
    [ -f "$W" ] || { log "stage0: window value ${B} expectile ${exv}"
      CUDA_VISIBLE_DEVICES=0 python3 /workspace/train_window.py --cache "$C1" \
        --lag 5 --frames 3 --expectile "$exv" --n-step 50 --steps 6000 --seed 0 \
        --out "$W" > "$L/w3_${B}_e${EX}.log" 2>&1 || log "window ${B} e${exv} FAILED"; }
  done
done
log "stage0 done: $(ls /workspace/metrics/window3_*_e*.pt 2>/dev/null | wc -l)/4 window values"

# ---------------------------------------------------------------- helpers
train(){ # base wm w3 amax lr replay expand seed out gpu
  local B=$1 W3=$2 am=$3 lr=$4 rp=$5 ew=$6 seed=$7 out=$8 gpu=$9
  [ -f "$out" ] && return 0
  local lrf=1e-4; [ "$lr" = "3e-4" ] && lrf=3e-5
  CUDA_VISIBLE_DEVICES=$gpu timeout 21600 python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/canon_${B}_fs5.pt \
    --cache-td /workspace/caches/canon_${B}_fs1.pt \
    --h5 "$SLIM" --wm "$(wm_of $B)" --pad-context --init-value "$W3" \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight 0.1 --steps 1000 \
    --amax "$am" --actor-lr "$lr" --actor-lr-final "$lrf" \
    --replay-prob "$rp" --expand-weight "$ew" --seed "$seed" \
    --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/$(basename ${out%.pt}).log" 2>&1 || log "TRAIN FAILED $(basename $out)"
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
meanof(){ local pre=$1 key=$2; shift 2; local t=0 n=0 e
  for s in "$@"; do
    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$e" ] && continue; t=$(awk "BEGIN{print $t+$e}"); n=$((n+1))
  done
  awk "BEGIN{printf \"%.1f\", ($n ? $t/$n : 0)}"
}
cfgmean(){ local pre=$1; shift; local t=0 k=0 v
  for sd in 0 1 2; do
    v=$(meanof "${pre}_s${sd}_sel" held $SEL); [ "$v" = "0.0" ] && continue
    t=$(awk "BEGIN{print $t+$v}"); k=$((k+1))
  done
  awk "BEGIN{printf \"%.1f\", ($k ? $t/$k : 0)}"
}

# ---------------------------------------------------------------- stage 1: shared probe on PLDM
log "stage1: replay/expand probe on PLDM (lejepa already says both hurt)"
i=0
for rp in 0 0.5; do for ew in 0 1.0; do for sd in 0 1 2; do
  t="p_r${rp//./}e${ew//./}"
  train pldm /workspace/metrics/window3_pldm_e01.pt 2.2 1e-3 "$rp" "$ew" "$sd" \
        /workspace/actors/lip4_jp_${t}_s${sd}.pt $((i % 6)) &
  i=$((i + 1)); [ $((i % 12)) -eq 0 ] && wait
done; done; done
wait
i=0
for rp in 0 0.5; do for ew in 0 1.0; do
  t="p_r${rp//./}e${ew//./}"
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_jp_${t}_s${sd}.pt; [ -f "$A" ] || continue
      for s in $SEL; do ev $((i % 6)) pldm "jp_${t}_s${sd}_sel${s}" $s "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
done; done
wait
# lejepa measured the identical 2x2 under the identical protocol
# (results/summary_rxe_lejepa.csv). The knob is SHARED, so it must be scored by
# the CROSS-BASE MEAN -- PLDM's local best is not admissible evidence on its own.
lejepa_rxe(){ case "$1_$2" in
  0_0)     echo 56.7;; 0_1.0)   echo 25.0;;
  0.5_0)   echo 43.3;; 0.5_1.0) echo 20.0;; *) echo 0;; esac; }

log "--- replay x expand, scored CROSS-BASE MEAN (the knob is shared) ---"
BEST_EW=0; BEST_EWV=0
for ew in 0 1.0; do
  tot=0; k=0
  for rp in 0 0.5; do
    t="p_r${rp//./}e${ew//./}"; vp=$(cfgmean "jp_${t}"); vl=$(lejepa_rxe "$rp" "$ew")
    m=$(awk "BEGIN{printf \"%.1f\", ($vp+$vl)/2}")
    log "  replay ${rp} / expand ${ew}: pldm ${vp} | lejepa ${vl} | cross-base ${m}"
    tot=$(awk "BEGIN{print $tot+$m}"); k=$((k+1))
  done
  em=$(awk "BEGIN{printf \"%.1f\", ($k ? $tot/$k : 0)}")
  awk "BEGIN{exit !($em > $BEST_EWV)}" && { BEST_EWV=$em; BEST_EW=$ew; }
done
log "SHARED expand-weight = ${BEST_EW} (cross-base ${BEST_EWV})"
log "  NOTE: PLDM alone prefers expand 1.0 at BOTH replay levels; lejepa is"
log "  destroyed by it. The shared constraint costs PLDM here -- report it."

# replay is a tie cross-base (37.7 vs 37.5 at n=3x2), so it is NOT decided by
# this probe: both values enter stage 2 as a shared axis and stage 3 picks the
# (expectile, replay) pair by the same cross-base rule.
RP_LIST="0 0.5"
log "SHARED replay = undecided by probe (tie); carrying {${RP_LIST}} into stage 2"

# ---------------------------------------------------------------- stage 2: per-base LIP sweep
log "stage2: per-base sweep, shared expectile {0.05,0.1} x per-base amax/lr"
i=0
for B in lejepa pldm; do for EX in 005 01; do for rp in $RP_LIST; do
for am in 1.8 2.2; do for lr in 3e-4 1e-3; do
  for sd in 0 1 2; do
    t="${B}_e${EX}a${am//./}l${lr//e-/e}r${rp//./}"
    train "$B" /workspace/metrics/window3_${B}_e${EX}.pt "$am" "$lr" "$rp" "$BEST_EW" "$sd" \
          /workspace/actors/lip4_js_${t}_s${sd}.pt $((i % 6)) &
    i=$((i + 1)); [ $((i % 18)) -eq 0 ] && wait
  done
done; done; done; done; done
wait
log "stage2 trainings drained"

i=0
for B in lejepa pldm; do for EX in 005 01; do for rp in $RP_LIST; do
for am in 1.8 2.2; do for lr in 3e-4 1e-3; do
  t="${B}_e${EX}a${am//./}l${lr//e-/e}r${rp//./}"
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_js_${t}_s${sd}.pt; [ -f "$A" ] || continue
      for s in $SEL; do ev $((i % 6)) "$B" "js_${t}_s${sd}_sel${s}" $s "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done; done; done; done
wait

# ---------------------------------------------------------------- stage 3: choose
log "--- stage2 screen ---"
: > /workspace/results/joint_screen.txt
for B in lejepa pldm; do for EX in 005 01; do for rp in $RP_LIST; do
for am in 1.8 2.2; do for lr in 3e-4 1e-3; do
  t="${B}_e${EX}a${am//./}l${lr//e-/e}r${rp//./}"; v=$(cfgmean "js_${t}")
  [ "$v" = "0.0" ] && continue
  echo "$v $B $EX $am $lr $rp $t" >> /workspace/results/joint_screen.txt
  log "  ${B} expectile${EX} replay${rp} amax${am} lr${lr}: ${v}"
done; done; done; done; done

# shared expectile: best CROSS-BASE mean of each base's best config at that expectile
# Both expectile and replay are SHARED, so they are chosen together as a pair,
# by the cross-base mean of each base's best (amax, lr) at that pair. amax/lr
# stay free per base -- that is the only asymmetry the user allowed.
BEST_EX=""; BEST_RP=""; BEST_EXV=0
for EX in 005 01; do for rp in $RP_LIST; do
  tot=0; k=0
  for B in lejepa pldm; do
    b=$(awk -v b="$B" -v e="$EX" -v r="$rp" '$2==b && $3==e && $6==r {print $1}' \
        /workspace/results/joint_screen.txt | sort -rn | head -1)
    [ -z "$b" ] && continue
    tot=$(awk "BEGIN{print $tot+$b}"); k=$((k+1))
  done
  [ "$k" -lt 2 ] && { log "  pair expectile${EX}/replay${rp}: incomplete (${k}/2 bases)"; continue; }
  m=$(awk "BEGIN{printf \"%.1f\", $tot/$k}")
  log "  shared pair expectile${EX} replay${rp}: cross-base mean of per-base bests = ${m}"
  awk "BEGIN{exit !($m > $BEST_EXV)}" && { BEST_EXV=$m; BEST_EX=$EX; BEST_RP=$rp; }
done; done
log "SHARED expectile = ${BEST_EX}, replay = ${BEST_RP}, expand = ${BEST_EW} (cross-base ${BEST_EXV})"

for B in lejepa pldm; do
  line=$(awk -v b="$B" -v e="$BEST_EX" -v r="$BEST_RP" '$2==b && $3==e && $6==r' \
         /workspace/results/joint_screen.txt | sort -rn | head -1)
  [ -z "$line" ] && { log "no winner for ${B}"; continue; }
  set -- $line; sv=$1; am=$4; lr=$5; t=$7
  log "WINNER ${B}: amax ${am}, lr ${lr}, expectile ${BEST_EX}, replay ${BEST_RP}, expand ${BEST_EW} (screen ${sv})"
  echo "${B} ${am} ${lr} ${BEST_EX} ${BEST_RP} ${BEST_EW}" > /workspace/_BEST_${B}
  ta=0; k=0
  for sd in 0 1 2; do
    A=/workspace/actors/lip4_js_${t}_s${sd}.pt; [ -f "$A" ] || continue
    for s in $REP; do ev $((sd % 6)) "$B" "js_${t}_s${sd}_e${s}" $s "$A"; done
    v=$(meanof "js_${t}_s${sd}_e" held $REP)
    log "  CARD ${B} s${sd}: HELD ${v} | @0.1 $(meanof "js_${t}_s${sd}_e" held10 $REP)"
    ta=$(awk "BEGIN{print $ta+$v}"); k=$((k+1))
  done
  [ "$k" -gt 0 ] && log "POOLED ${B}: HELD $(awk "BEGIN{printf \"%.1f\", $ta/$k}") over ${k} seeds"
done
log "JOINT_DONE -- bars: lejepa Latent+CEM-window 44.7 | pldm TBD"
