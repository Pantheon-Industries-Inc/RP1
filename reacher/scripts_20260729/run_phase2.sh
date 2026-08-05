#!/bin/bash
# Phase 2, chained behind run_g098_grid.sh: refine gamma, then push mean-weight.
#
# A. GAMMA FINE SWEEP {0.95, 0.97, 0.99}
#    Gamma produced the largest single swing in the campaign (lejepa 89.0 ->
#    95.8 @0.1) and the response is NOT monotone:
#        lejepa   g0.98 95.8   g0.995 81.1   g1.0 89.0
#        pldm     g0.98 84.6   g0.995 79.2   g1.0 90.4
#    0.98 is simply the value that was named, not a located optimum, and the
#    sharp collapse at 0.995 says the useful region is narrow and below it.
#    Gamma is a cost-to-go knob, so it stays SHARED and is chosen by min-margin
#    over each base's own bar.
#
# B. MEAN-WEIGHT {0.5, 1.0} at the winning gamma
#    mw 0.3 won at the EDGE of {0.03, 0.1, 0.3} -- the identical setup that made
#    the LR sweep necessary after 3e-4 won at the edge of {3e-4, 1e-3}. Twice
#    burned, so the range is extended rather than the edge being reported as an
#    optimum. mw is a LIP knob, so it may differ per base.
#
# Both phases inherit each base's (lr, mw) winner from the gamma-0.98 grid,
# read from results/g098_screen.txt rather than hardcoded.
#
# Caveat carried forward: screens are on selection seeds {50,51} while the bars
# (84.3 / 78.3) are reporting-seed numbers, so min-margin on screens is an
# approximation used only to RANK. Winners are always carded on {42..47}.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
SUM=/workspace/results/summary_phase2.csv; touch "$SUM"
L=/workspace/logs/phase2; mkdir -p "$L"
SEL="50 51"; REP="42 43 44 45 46 47"
GAMMAS="0.95 0.97 0.99"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][p2] $*"; }
die(){ log "FATAL: $*"; exit 1; }
wm_of(){ echo "/workspace/swm_home/checkpoints/${1}_reacher"; }
amax_of(){ [ "$1" = "lejepa" ] && echo 2.2 || echo 1.8; }
bar_of(){ [ "$1" = "lejepa" ] && echo 84.3 || echo 78.3; }
gt(){ echo "g${1//./}"; }
lrf_of(){ case "$1" in 1e-4) echo 1e-5;; 3e-4) echo 3e-5;; *) echo 1e-5;; esac; }

# ---------------------------------------------------------------- gate
if ! grep -q "G098GRID_DONE" /workspace/logs/g098grid.log 2>/dev/null; then
  log "waiting for the gamma-0.98 grid (ceiling 8 h)"
  w=0
  while ! grep -q "G098GRID_DONE" /workspace/logs/g098grid.log 2>/dev/null; do
    sleep 120; w=$((w + 120))
    [ $((w % 1800)) -eq 0 ] && log "  still waiting, ${w}s"
    [ "$w" -ge 28800 ] && die "g098 grid never finished"
  done
fi
[ -s /workspace/results/g098_screen.txt ] || die "no g098 screen to inherit from"

declare -A BLR BMW
for B in lejepa pldm; do
  line=$(awk -v b="$B" '$2==b' /workspace/results/g098_screen.txt | sort -rn | head -1)
  [ -z "$line" ] && die "no g098 winner for ${B}"
  set -- $line; BLR[$B]=$3; BMW[$B]=$4
  log "inherited ${B}: lr ${BLR[$B]} mw ${BMW[$B]} (g0.98 screen @0.1 $1)"
done

# ---------------------------------------------------------------- helpers
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
trainlip(){ local B=$1 g=$2 lr=$3 mw=$4 sd=$5 out=$6 gpu=$7 val=$8
  [ -f "$out" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/canon_${B}_fs5.pt \
    --cache-td /workspace/caches/canon_${B}_fs1.pt \
    --h5 "$SLIM" --wm "$(wm_of $B)" --pad-context --init-value "$val" \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --gamma "$g" --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight "$mw" --amax "$(amax_of $B)" \
    --replay-prob 0.5 --expand-weight 0 --steps 1000 \
    --actor-lr "$lr" --actor-lr-final "$(lrf_of $lr)" --seed "$sd" \
    --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/$(basename ${out%.pt}).log" 2>&1 || log "TRAIN FAILED $(basename $out)"; }

card(){ local B=$1 pre=$2 label=$3 tagbase=$4
  local i=0 ta=0 tb=0 k=0 a c
  for sd in 0 1 2; do
    local A=${tagbase}_s${sd}.pt; [ -f "$A" ] || continue
    for s in $REP; do ev $(( (i % 5) + 1 )) "$B" "${pre}_s${sd}_e${s}" "$s" "$A"; i=$((i+1)); done
    a=$(meanof "${pre}_s${sd}_e" held10 $REP); c=$(meanof "${pre}_s${sd}_e" held $REP)
    log "  CARD ${label} s${sd}: @0.1 ${a} | @0.05 ${c}"
    ta=$(awk "BEGIN{print $ta+$a}"); tb=$(awk "BEGIN{print $tb+$c}"); k=$((k+1))
  done
  [ "$k" -gt 0 ] && log "POOLED ${label}: @0.1 $(awk "BEGIN{printf \"%.1f\", $ta/$k}") (bar $(bar_of $B)) | @0.05 $(awk "BEGIN{printf \"%.1f\", $tb/$k}")"; }

# ================================================================ A. gamma fine sweep
log "=== A: gamma fine sweep {${GAMMAS}} at each base's inherited (lr, mw) ==="
i=0
for g in $GAMMAS; do for B in lejepa pldm; do
  W=/workspace/metrics/window3_${B}_e005_$(gt $g).pt
  [ -f "$W" ] && { i=$((i+1)); continue; }
  CUDA_VISIBLE_DEVICES=$(( (i % 5) + 1 )) python3 /workspace/train_window.py \
    --cache /workspace/caches/canon_${B}_fs1.pt --lag 5 --frames 3 \
    --expectile 0.05 --n-step 50 --gamma "$g" --steps 6000 --seed 0 --out "$W" \
    > "$L/w3_${B}_$(gt $g).log" 2>&1 &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done
wait
log "values built"

i=0
for g in $GAMMAS; do for B in lejepa pldm; do for sd in 0 1 2; do
  trainlip "$B" "$g" "${BLR[$B]}" "${BMW[$B]}" "$sd" \
    "/workspace/actors/lip4_p2g_${B}_$(gt $g)_s${sd}.pt" $(( (i % 5) + 1 )) \
    "/workspace/metrics/window3_${B}_e005_$(gt $g).pt" &
  i=$((i + 1)); [ $((i % 10)) -eq 0 ] && wait
done; done; done
wait
log "gamma trainings drained"

i=0
for g in $GAMMAS; do for B in lejepa pldm; do
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_p2g_${B}_$(gt $g)_s${sd}.pt; [ -f "$A" ] || continue
      for s in $SEL; do ev $(( (i % 5) + 1 )) "$B" "p2g_${B}_$(gt $g)_s${sd}_sel${s}" "$s" "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % 5)) -eq 0 ] && wait
done; done
wait

log "--- gamma screen (selection seeds, @0.1; 0.98 row is the g098 grid winner) ---"
BEST_G=""; BEST_MM=-999
for g in $GAMMAS; do
  ml=$(cfgmean "p2g_lejepa_$(gt $g)" held10); mp=$(cfgmean "p2g_pldm_$(gt $g)" held10)
  [ "$ml" = "0.0" ] || [ "$mp" = "0.0" ] && { log "  gamma ${g}: incomplete (lejepa ${ml}, pldm ${mp})"; continue; }
  mm=$(awk "BEGIN{a=$ml-84.3; b=$mp-78.3; print (a<b?a:b)}")
  log "  gamma ${g}: lejepa ${ml} | pldm ${mp} | min-margin ${mm}"
  awk "BEGIN{exit !($mm > $BEST_MM)}" && { BEST_MM=$mm; BEST_G=$g; }
done
g98l=$(awk -v b=lejepa '$2==b {print $1}' /workspace/results/g098_screen.txt | sort -rn | head -1)
g98p=$(awk -v b=pldm   '$2==b {print $1}' /workspace/results/g098_screen.txt | sort -rn | head -1)
mm98=$(awk "BEGIN{a=$g98l-84.3; b=$g98p-78.3; print (a<b?a:b)}")
log "  gamma 0.98: lejepa ${g98l} | pldm ${g98p} | min-margin ${mm98}   [incumbent]"
awk "BEGIN{exit !($mm98 > $BEST_MM)}" && { BEST_MM=$mm98; BEST_G=0.98; }
log "SHARED gamma = ${BEST_G} (min-margin ${BEST_MM})"

if [ "$BEST_G" != "0.98" ]; then
  for B in lejepa pldm; do
    card "$B" "p2g_${B}_$(gt $BEST_G)" "${B} gamma ${BEST_G}" \
         "/workspace/actors/lip4_p2g_${B}_$(gt $BEST_G)"
  done
fi

# ================================================================ B. mean-weight extension
log "=== B: mean-weight {0.5, 1.0} at gamma ${BEST_G} ==="
VAL_OF(){ [ "$1" = "0.98" ] && echo "/workspace/metrics/window3_${2}_e005_g098.pt" \
                            || echo "/workspace/metrics/window3_${2}_e005_$(gt $1).pt"; }
i=0
for mw in 0.5 1.0; do for B in lejepa pldm; do for sd in 0 1 2; do
  trainlip "$B" "$BEST_G" "${BLR[$B]}" "$mw" "$sd" \
    "/workspace/actors/lip4_p2m_${B}_m${mw//./}_s${sd}.pt" $(( (i % 5) + 1 )) \
    "$(VAL_OF $BEST_G $B)" &
  i=$((i + 1)); [ $((i % 10)) -eq 0 ] && wait
done; done; done
wait
log "mean-weight trainings drained"

i=0
for mw in 0.5 1.0; do for B in lejepa pldm; do
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_p2m_${B}_m${mw//./}_s${sd}.pt; [ -f "$A" ] || continue
      for s in $SEL; do ev $(( (i % 5) + 1 )) "$B" "p2m_${B}_m${mw//./}_s${sd}_sel${s}" "$s" "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
done; done
wait

log "--- mean-weight screen at gamma ${BEST_G} (selection seeds, @0.1) ---"
for B in lejepa pldm; do
  best_mw=""; best_v=-1
  for mw in 0.5 1.0; do
    v=$(cfgmean "p2m_${B}_m${mw//./}" held10)
    log "  ${B} mw ${mw}: @0.1 ${v}   (inherited mw ${BMW[$B]} is the reference)"
    awk "BEGIN{exit !($v > $best_v)}" && { best_v=$v; best_mw=$mw; }
  done
  [ -n "$best_mw" ] && card "$B" "p2m_${B}_m${best_mw//./}" "${B} gamma ${BEST_G} mw ${best_mw}" \
        "/workspace/actors/lip4_p2m_${B}_m${best_mw//./}"
done
log "PHASE2_DONE -- reference: lejepa 95.8 (g0.98), pldm 90.4 (g1.0); bars 84.3 / 78.3"
