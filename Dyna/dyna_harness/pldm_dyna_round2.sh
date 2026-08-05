#!/bin/bash
# DYNA ROUND 2 ON PLDM -- iterate the loop on WM1's own actors.
#
# Round 1 gave PRE 75.1 -> POST 83.1 (+8.0, n=3, t=2.11, not yet significant;
# the 20-seed replication is running separately). Round 2 asks whether the loop
# compounds: collect with the ROUND-1 actors rolled out on the ROUND-1 WM, then
# fine-tune WM1 -> WM2 on that fresh data.
#
# Protocol is byte-identical to round 1 so r0->r1 and r1->r2 are the same
# operation and the deltas are directly comparable: episode-disjoint 0-7999,
# terminate_at_goal=False, 50/50 expert:on-policy uniform-K, lr 1e-5 x 2 epochs
# with epoch 1 pre-registered, one lance per collection call, then fresh
# caches -> TD -> 3 LIP actors at the winning config (mw0.1/amax4.5/lr3e-4).
#
# INIT is WM1, not the original PLDM -- that is what makes this an iteration
# rather than a re-run with more data. (The LeWM campaign's round 2 instead
# used a 60/30/10 expert/r1/r2 no-duplication mix and reproduced round 1
# without exceeding it; keeping the operation identical here isolates
# 'does the loop compound' from 'does the mixture matter', and the LeWM null
# trilogy already said mixture does not.)
#
# Expect an honest possibility of ~0: round 1's gain may have been a one-off
# distribution correction. LeWM's round 2 did not exceed round 1.
#
# Queued behind the POST seed replication AND the CEM probe (collection
# renders, so it must not overlap training or evals).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_split
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_r2.log
WM1=/workspace/models/dyna_pldm_5050          # round-1 WM (the init AND the collector)
WM2=/workspace/models/dyna_pldm_r2
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
MIX=$D/mix_pldm_r2.lance
R2F1F=/workspace/caches/r2pldm_full_fs1.pt
R2F1=/workspace/caches/r2pldm_tr8000_fs1.pt
R2F5=/workspace/caches/r2pldm_tr8000_fs5.pt
R2TD=/workspace/metrics/r2pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; COLLECT_RANGE="0:${EPHI}"
AMAX=4.5; MW=0.1; ALR=3e-4; ALRF=3e-5
SEEDS="0 1 2"; DRAWS="42 43 44"; NCALL=12; NGPU=8
mkdir -p "$D" "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== DYNA ROUND 2 ON PLDM (pid $$) ==="
for s in $SEEDS; do [ -f "/workspace/actors/lip4_dynapldm_s${s}.pt" ] || die "round-1 actor s$s missing"; done
[ -f "$WM1/weights_epoch_1.pt" ] || die "round-1 WM missing"
for f in "$AH5" /workspace/build_dyna_mix.py /workspace/filter_cache_eprange.py; do
  [ -e "$f" ] || die "missing $f"; done
log "P0: preflight OK (init+collector = WM1, config mw$MW/a$AMAX/lr$ALR)"
[ "${1:-}" = check ] && { log "check mode"; exit 0; }

# ---------------- wait for the queue ahead (collection renders)
log "P1: waiting for the POST seed replication and the CEM probe"
T0=$(date +%s)
until grep -q "PLDM_CEMPROBE_DONE" "$L/driver_cemprobe.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && { log "WARN: queue not drained in 12h; proceeding once idle"; break; }
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 60; done
log "P1: box quiet"

# ------------------------------------------------- P2 collection with WM1 actors
if [ ! -f "$D/R2_COLLECT_DONE" ]; then
  log "P2: collection -- WM1 + round-1 actors, ${NCALL} calls x 3 actors, 8-wide"
  g=0
  for i in $(seq 0 $((NCALL-1))); do for a in $SEEDS; do
    rec="$D/onpolicy_r2_a${a}_c${i}.lance"; lg="$L/r2col_a${a}_c${i}.log"
    grep -q "kept=" "$lg" 2>/dev/null && continue
    (
      SWM_RECORD_PATH=$rec SWM_RECORD_OUTCOME=1 \
      CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 10800 \
      python3 "$P/eval_wm.py" --config-name cube seed=$((5000 + a*100 + i)) \
        eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
        eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=50 \
        "+eval.ep_range=$COLLECT_RANGE" world.terminate_at_goal=False \
        policy="$WM1" solver=lip \
        "solver.actor_path=/workspace/actors/lip4_dynapldm_s${a}.pt" \
        output.filename="r2col_a${a}_c${i}.txt" > "$lg" 2>&1
      rl=$(grep -h "\[record\]" "$lg" | tail -1)
      echo "$rl" | grep -q "kept=50 dropped=0" || echo "  WARN a$a/c$i: $rl" >> "$DRV"
    ) &
    g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
  done; done; wait
  ok=$(grep -l "kept=50 dropped=0" "$L"/r2col_a*_c*.log 2>/dev/null | wc -l)
  [ "$ok" -ge 30 ] || die "only $ok/36 collection calls clean"
  log "P2: $ok/36 clean; on-policy success $(grep -ho 'kept_success=[0-9]*' "$L"/r2col_a*_c*.log | cut -d= -f2 | awk '{s+=$1;n++}END{printf "%.1f%%", 100*s/(n*50)}') (round 1 collected at 68.8%)"
  touch "$D/R2_COLLECT_DONE"
fi

# ---------------------------------------------------------------- P3 mix
if [ ! -f "$MIX/.done" ]; then
  log "P3: 50/50 uniform-K mix, expert < $EPHI"
  python3 /workspace/build_dyna_mix.py --expert "$EXPERT" \
    --onpolicy $D/onpolicy_r2_a*_c*.lance \
    --out "$MIX" --onpolicy-frac 0.5 --expert-ep-hi "$EPHI" \
    > "$L/r2_mix.log" 2>&1 || { tail -5 "$L/r2_mix.log"; die "mix failed"; }
  grep -q BUILD_MIX_DONE "$L/r2_mix.log" || die "mix incomplete"
  grep -q "\[split\] expert restricted" "$L/r2_mix.log" || die "expert slice NOT restricted"
  touch "$MIX/.done"
fi
log "P3: $(grep -h 'dup K=' "$L/r2_mix.log" | tail -1)"

# ------------------------------------------- P4 fine-tune WM1 -> WM2
if [ ! -f "$WM2/weights_epoch_1.pt" ]; then
  log "P4: fine-tune WM1 -> dyna_pldm_r2 (lr 1e-5, 2 epochs, epoch 1; ~3.5h)"
  INIT_WEIGHTS=$WM1/weights_epoch_1.pt CUDA_VISIBLE_DEVICES=0 timeout 86400 python3 \
    scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
    "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
    "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
    trainer.devices=1 output_model_name=dyna_pldm_r2 subdir=dyna_pldm_r2 \
    +action_stats_pin=expert wandb.enabled=false > "$L/r2_ft.log" 2>&1 \
    || { tail -15 "$L/r2_ft.log"; die "fine-tune failed"; }
  CK=/workspace/swm_home/checkpoints/dyna_pldm_r2
  mkdir -p "$WM2"; cp "$CK/weights_epoch_1.pt" "$WM2/" || die "no weights_epoch_1.pt"
  cp "$WM1/config.json" "$WM2/config.json" || die "no arch config"
fi
cmp -s "$WM2/config.json" "$WM1/config.json" || die "packaged config differs"
grep -h "lr-probe" "$L/r2_ft.log" | head -2 | sed 's/^/  /' | tee -a "$DRV"
log "P4: WM2 ready"

# ------------------------------------------------------- P5 caches + TD
if [ ! -f "$R2F1" ]; then
  log "P5: caching fs1 under WM2"
  [ -f "$R2F1F" ] || CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$TRM/cache_latents.py" \
    --wm "$WM2" --dataset "$EXPERT" --out "$R2F1F" --state-key privileged_block_0_pos \
    > "$L/r2_cache.log" 2>&1 || { tail -5 "$L/r2_cache.log"; die "cache failed"; }
  python3 /workspace/filter_cache_eprange.py "$R2F1F" "$R2F1" --lo 0 --hi "$EPHI" \
    > "$L/r2_filter.log" 2>&1 || die "filter failed"
  grep -q FILTER_CACHE_DONE "$L/r2_filter.log" || die "filter incomplete"
fi
[ -f "$R2F5" ] || python3 "$TRM/subsample_cache.py" --in "$R2F1" --out "$R2F5" --frameskip 5 \
  > "$L/r2_fs5.log" 2>&1 || die "fs5 failed"
[ -f "$R2TD" ] || CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_metric.py" \
  --cache "$R2F1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
  --steps 6000 --seed 0 --out "$R2TD" > "$L/r2_td.log" 2>&1 || die "TD failed"
log "P5: caches + TD ready"

# ---------------------------------------------------- P6 actors + P7 evals
log "P6: round-2 LIP actors, 3 seeds"
g=0; for s in $SEEDS; do
  out=/workspace/actors/lip4_r2pldm_s${s}.pt
  [ -f "$out" ] || CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$R2F5" --cache-td "$R2F1" --h5 "$AH5" --wm "$WM2" --init-value "$R2TD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
    --mean-weight "$MW" --actor-lr "$ALR" --actor-lr-final "$ALRF" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --arch v4 --seed "$s" --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/r2_lip_s${s}.log" 2>&1 &
  g=$((g+1))
done; wait
log "P6: actors done"

ev(){ # gpu name wm extra...
  local gpu=$1 nm=$2 wm=$3 d; shift 3
  d=$(echo "$nm" | grep -oE "e4[234]$" | tr -d e)
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$wm" "$@" output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
}
log "P7: evals -- LIP 9 cells + CEM/TD+CEM on WM2 for the planner ladder"
while pgrep -f "train_lip_a[c]|train_metri[c]" >/dev/null; do sleep 30; done
g=0
for s in $SEEDS; do for d in $DRAWS; do
  ev "$g" "r2pldm_s${s}_e${d}" "$WM2" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_r2pldm_s${s}.pt" &
  g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done
for d in $DRAWS; do ev "$g" "r2wm_cem_e${d}" "$WM2" solver=cem & g=$((g+1)); done
for d in $DRAWS; do ev "$g" "r2wm_cemtd_e${d}" "$WM2" solver=cem "+metric=$R2TD" & g=$((g+1)); done
wait

# ------------------------------------------------------------------ P8 card
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
def m3(p):
    vs = [rows.get(f"{p}_e{d}") for d in (42,43,44)]
    return None if any(v is None for v in vs) else mean(vs)
def seeds(p, ns=range(20)):
    out = []
    for s in ns:
        vs = [rows.get(f"{p}_s{s}_e{d}") for d in (42,43,44)]
        if all(v is not None for v in vs): out.append(mean(vs))
    return out
print("=== DYNA ROUND 2 ON PLDM (held-out, EGL, amax 4.5) ===")
print(f"  {'stage':16s} {'LIP':>18s} {'latent+CEM':>11s} {'TD+CEM':>8s}")
lip0 = seeds("grid_mw01_a45_lr3e-4"); lip1 = seeds("dynapldm"); lip2 = seeds("r2pldm")
rows_out = [
    ("frozen PLDM", lip0, m3("ref_cem"),    m3("ref_cemtd")),
    ("round 1 WM",  lip1, m3("dynawm_cem"), m3("dynawm_cemtd")),
    ("round 2 WM",  lip2, m3("r2wm_cem"),   m3("r2wm_cemtd")),
]
for nm, lip, c, t in rows_out:
    ls = f"{mean(lip):.1f} (n={len(lip)})" if lip else "--"
    print(f"  {nm:16s} {ls:>18s} {('%.1f'%c) if c else '--':>11s} {('%.1f'%t) if t else '--':>8s}")
if lip1 and lip2:
    # paired on the 3 shared seeds
    a = seeds("dynapldm", range(3)); b = seeds("r2pldm", range(3))
    if len(a) == 3 and len(b) == 3:
        d = [x-y for x, y in zip(b, a)]; m = mean(d)
        sd = math.sqrt(sum((x-m)**2 for x in d)/2)
        t = m/(sd/math.sqrt(3)) if sd > 0 else float('inf')
        print(f"  round2 - round1 (paired, seeds 0-2): {m:+.2f} sd {sd:.2f} t {t:.2f} (df=2)")
        print("  reading: >0 => the Dyna loop COMPOUNDS on PLDM (LeWM's round 2 did not);")
        print("  ~0 => round 1 was a one-off distribution correction, loop saturated.")
print("PLDM_R2_DONE")
PY
log "done"
