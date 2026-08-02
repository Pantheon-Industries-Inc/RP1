#!/bin/bash
# DINO-WM (dinowmnp) as a third base: cache -> value -> its own bar -> LIP.
#
# The checkpoint is reacher-specific (reacher/dinowm_noprop.tar.zst in git,
# converted by convert_reacher_bases.py --only dinowmnp). Its parity gate:
#     backbone vs stock dinov2-small: 0 differing tensors  (frozen DINOv2)
#     1step  pred 0.328 | copy-last 0.304  (ratio 1.077)
#     4step  pred 0.358 | copy-last 0.614  (ratio 0.583)
#     shuffled-act ratios 0.541 / 0.361
# So it is genuinely action-conditioned and beats copy-last over 4 steps, but
# NOT at 1 step. Prior campaign: Latent+CEM 65.3 at the paper-anchor config
# against the paper's quoted 79, and it is on record as a "documented dead end".
#
# THAT IS WHY IT NEEDS ITS OWN BAR. A weaker world model lowers every arm that
# plans through it, so comparing LIP-on-dinowmnp against lejepa's 84.3 would be
# meaningless. Stage 2 measures Latent+CEM and TD+CEM on this base, exactly as
# was done for pldm (39.3 / 78.3), before any LIP number is interpreted.
#
# LEAKED configuration (user, 2026-08-02): full caches, episodes 0..9999, which
# is what the reported lejepa/pldm numbers use. The separate clean-split control
# measured the leak at +0.2 / +3.6 points.
#
# The L2 window cost is parameter-free but its declared width must match
# 3 x latent_dim. lejepa/pldm are 192-d, giving the existing 576-d l2window3.pt;
# DINOv2-small is 384-d, so if this base has a different latent width the stub is
# regenerated at the right size rather than silently reused at the wrong one.
#
# Gated behind PWM so the three jobs do not overlap on the GPUs.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
TRM=/workspace/swm_cem/scripts/trm
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
SLIM=/workspace/reacher_slim.h5
B=dinowmnp
WM=/workspace/swm_home/checkpoints/${B}_reacher
C1=/workspace/caches/canon_${B}_fs1.pt
C5=/workspace/caches/canon_${B}_fs5.pt
W3=/workspace/metrics/window3_${B}_e005_g098.pt
SUM=/workspace/results/summary_dinowm.csv; touch "$SUM"
L=/workspace/logs/dinowm; mkdir -p "$L"
SEL="50 51"; REP="42 43 44 45 46 47"; G=0.98
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][dino] $*"; }
die(){ log "FATAL: $*"; exit 1; }

[ -f "$WM/weights.pt" ] || die "no converted checkpoint at $WM"

if ! grep -q "PWM_DONE" /workspace/logs/pwm.log 2>/dev/null; then
  log "waiting for run_pwm.sh (ceiling 8 h)"
  w=0
  while ! grep -q "PWM_DONE" /workspace/logs/pwm.log 2>/dev/null; do
    sleep 120; w=$((w + 120))
    [ $((w % 1800)) -eq 0 ] && log "  still waiting, ${w}s"
    [ "$w" -ge 28800 ] && die "pwm never finished"
  done
fi

# ---------------------------------------------------------------- 1. caches
if [ ! -f "$C1" ]; then
  log "encoding fs1 latents (~30 min)"
  CUDA_VISIBLE_DEVICES=0 python3 "$TRM/cache_latents.py" --wm "$WM" \
    --dataset "$CANON" --out "$C1" --state-key qpos --batch-size 512 \
    > "$L/cache1.log" 2>&1 || die "fs1 cache failed (see cache1.log)"
fi
[ -f "$C5" ] || CUDA_VISIBLE_DEVICES=0 python3 "$TRM/subsample_cache.py" \
  --in "$C1" --out "$C5" --frameskip 5 > "$L/cache5.log" 2>&1 || die "fs5 failed"
DIM=$(python3 -c "
import torch; c=torch.load('$C1',map_location='cpu',weights_only=False); print(c['z'].shape[1])")
log "caches ready, latent_dim=${DIM}"

# ---------------------------------------------------------------- 2. L2 stub at the right width
L2=/workspace/metrics/l2window3.pt
if [ "$DIM" != "192" ]; then
  L2=/workspace/metrics/l2window3_${DIM}.pt
  [ -f "$L2" ] || python3 -c "
import torch, collections
torch.save({'learner':'l2','latent_dim':3*$DIM,
            'arch':{'window_frames':3,'window_lag':5,'unlearned':True},
            'state_dict':collections.OrderedDict()}, '$L2')
print('wrote $L2 at 3x$DIM')"
  log "using ${L2} (this base is ${DIM}-d, not 192-d)"
fi

# ---------------------------------------------------------------- 3. window value
if [ ! -f "$W3" ]; then
  log "window value: gamma ${G}, expectile 0.05, n-step 50"
  CUDA_VISIBLE_DEVICES=0 python3 /workspace/train_window.py \
    --cache "$C1" --lag 5 --frames 3 --expectile 0.05 --n-step 50 \
    --gamma "$G" --steps 6000 --seed 0 --out "$W3" > "$L/w3.log" 2>&1 || die "value failed"
fi
log "value loss $(grep -oE 'final loss=[0-9.]+' "$L/w3.log" | tail -1 | cut -d= -f2)"
grep -oE "\[tiled-goal diag\].*" "$L/w3.log" | tail -1

# ---------------------------------------------------------------- helpers
ev(){ local gpu=$1 nm=$2 seed=$3; shift 3
  grep -q "^${nm},held=[0-9]" "$SUM" && return 0
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  timeout 9000 python3 "$PLAN/eval_wm.py" --config-name reacher \
    policy="$WM" eval.dataset_name="$CANON" dataset.stats="$CANON" \
    +eval.ep_range=8000:10000 seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 solver.batch_size=10 \
    "$@" output.filename=${nm}.txt > "$L/${nm}.log" 2>&1
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

# ---------------------------------------------------------------- 4. this base's own bars
log "stage2: Latent+CEM and TD+CEM on ${B} -- LIP is meaningless without them"
( for s in $REP; do ev 0 "dw_l2cem_s${s}" $s solver=cem solver.n_steps=10 "+metric=$L2"; done ) &
( for s in $REP; do ev 1 "dw_tdcem_s${s}" $s solver=cem solver.n_steps=10 "+metric=$W3"; done ) &
wait
BAR10=$(meanof dw_l2cem_s held10 $REP); BAR05=$(meanof dw_l2cem_s held $REP)
log "  Latent+CEM (bar): @0.05 ${BAR05} | @0.1 ${BAR10}  ($(grep -c '^dw_l2cem_s.*held=[0-9]' "$SUM")/6)"
log "  TD+CEM         : @0.05 $(meanof dw_tdcem_s held $REP) | @0.1 $(meanof dw_tdcem_s held10 $REP)  ($(grep -c '^dw_tdcem_s.*held=[0-9]' "$SUM")/6)"
log "  reference: lejepa bar 84.3, pldm bar 78.3 (@0.1); prior dinowmnp anchor 65.3 latched"

# ---------------------------------------------------------------- 5. LIP grid
log "stage3: LIP grid -- amax{1.8,2.2} x lr{1e-4,3e-4} x mw{0.1,0.3} x 3 seeds"
tagof(){ echo "a${1//./}l${2//e-/e}m${3//./}"; }
i=0
for am in 1.8 2.2; do for lr in 1e-4 3e-4; do for mw in 0.1 0.3; do for sd in 0 1 2; do
  t=$(tagof "$am" "$lr" "$mw"); A=/workspace/actors/lip4_dw_${t}_s${sd}.pt
  [ -f "$A" ] && { i=$((i+1)); continue; }
  lrf=1e-5; [ "$lr" = "3e-4" ] && lrf=3e-5
  CUDA_VISIBLE_DEVICES=$(( i % 6 )) timeout 28800 python3 "$PLAN/train_lip_ac.py" \
    --cache "$C5" --cache-td "$C1" --h5 "$SLIM" --wm "$WM" --pad-context \
    --init-value "$W3" \
    --arch v4 --iters 8 --horizon 5 --max-delta 12 --batch 128 --n-step 50 \
    --gamma "$G" --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --lambda-schedule uniform --mean-weight "$mw" --amax "$am" \
    --replay-prob 0.5 --expand-weight 0 --steps 1000 \
    --actor-lr "$lr" --actor-lr-final "$lrf" --seed "$sd" \
    --out "$A" --out-value "/workspace/metrics/lip4_dw_${t}_s${sd}_value.pt" \
    > "$L/train_${t}_s${sd}.log" 2>&1 || log "TRAIN FAILED ${t} s${sd}" &
  i=$((i + 1)); [ $((i % 12)) -eq 0 ] && wait
done; done; done; done
wait
log "actors: $(ls /workspace/actors/lip4_dw_*_s?.pt 2>/dev/null | wc -l)/24"

i=0
for am in 1.8 2.2; do for lr in 1e-4 3e-4; do for mw in 0.1 0.3; do
  t=$(tagof "$am" "$lr" "$mw")
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_dw_${t}_s${sd}.pt; [ -f "$A" ] || continue
      for s in $SEL; do ev $(( i % 5 )) "dw_${t}_s${sd}_sel${s}" $s solver=lip solver.actor_path="$A" solver.rollout_compat=false; done
    done ) &
  i=$((i + 1)); [ $((i % 5)) -eq 0 ] && wait
done; done; done
wait

log "--- LIP screen on ${B} (selection seeds, @0.1) ---"
: > /workspace/results/dinowm_screen.txt
for am in 1.8 2.2; do for lr in 1e-4 3e-4; do for mw in 0.1 0.3; do
  t=$(tagof "$am" "$lr" "$mw"); v=$(cfgmean "dw_${t}" held10)
  [ "$v" = "0.0" ] && continue
  echo "$v $am $lr $mw $t" >> /workspace/results/dinowm_screen.txt
  log "  amax ${am} lr ${lr} mw ${mw}: @0.1 ${v}  @0.05 $(cfgmean "dw_${t}" held)"
done; done; done

line=$(sort -rn /workspace/results/dinowm_screen.txt | head -1)
[ -z "$line" ] && { log "no LIP winner -- every cell failed"; log "DINOWM_DONE"; exit 0; }
set -- $line; t=$5
log "WINNER ${B}: amax $2 lr $3 mw $4 (screen @0.1 $1)"
i=0; ta=0; tb=0; k=0; nsc=0
for sd in 0 1 2; do
  A=/workspace/actors/lip4_dw_${t}_s${sd}.pt; [ -f "$A" ] || continue
  for s in $REP; do ev $(( i % 5 )) "dw_${t}_s${sd}_e${s}" $s solver=lip solver.actor_path="$A" solver.rollout_compat=false; i=$((i+1)); done
  a=$(meanof "dw_${t}_s${sd}_e" held10 $REP); c=$(meanof "dw_${t}_s${sd}_e" held $REP)
  n=$(grep -c "^dw_${t}_s${sd}_e.*held=[0-9]" "$SUM"); nsc=$((nsc + n))
  log "  CARD ${B} s${sd}: @0.1 ${a} | @0.05 ${c}  (${n}/6)"
  ta=$(awk "BEGIN{print $ta+$a}"); tb=$(awk "BEGIN{print $tb+$c}"); k=$((k+1))
done
if [ "$k" -gt 0 ]; then
  M10=$(awk "BEGIN{printf \"%.1f\", $ta/$k}")
  log "POOLED ${B} LIP: @0.1 ${M10} | @0.05 $(awk "BEGIN{printf \"%.1f\", $tb/$k}")  (${nsc}/18 evals)"
  log "  vs its OWN Latent+CEM bar ${BAR10} -> margin $(awk "BEGIN{printf \"%+.1f\", $M10-$BAR10}")"
fi
log "DINOWM_DONE"
