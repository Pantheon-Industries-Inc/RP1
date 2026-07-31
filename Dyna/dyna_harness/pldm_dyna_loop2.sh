#!/bin/bash
# DYNA LOOP ON PLDM -- the best-results protocol, on the rebuilt EGL pod.
#
# Fills the last empty cell of the cross-base table. Protocol is exactly the
# one that produced LeWM's +3.6: episode-disjoint (collect 0-7999, eval
# 8000-9999), terminate_at_goal=False collection, 50/50 uniform-K mix, WM
# fine-tune at lr 1e-5 for 2 epochs with epoch 1 pre-registered, then fresh
# caches -> TD -> LIP actors on the fine-tuned WM.
#
# Base compatibility verified before writing this: PLDM_OgBench_lewm and v2WM
# are architecturally IDENTICAL (ViT-tiny 192/12/3, predictor 192-192-192
# depth 6 heads 16 mlp 2048; only emb_dropout None vs 0.0 differs, which is
# not shape-affecting), so lewm_expert.py's model accepts PLDM via
# INIT_WEIGHTS. Caveat to state in any writeup: the fine-tune objective is
# LeWM's (MSE + SIGReg), not the one PLDM was pretrained with.
#
# PRE arm = the grid winner mw0.1/amax4.5/lr3e-4, which is also the config
# the earlier amax sweep landed on. Its draws 43/44 exist from stage B and its
# s0/e42 from the screen; P1 fills the two missing cells so PRE is a full
# 3 seeds x 3 draws before POST is measured against it.
#
# PARALLELISM: collection and evals run 8-wide using the validated per-GPU
# pinning (CUDA_VISIBLE_DEVICES=i AND MUJOCO_EGL_DEVICE_ID=i). Each collection
# CALL writes its OWN lance -- concurrent appends to one lance would race, and
# build_dyna_mix.py takes an arbitrary number of --onpolicy paths anyway.
#
# ~5h: collection ~20m, mix ~3m, fine-tune ~3.5h (dominates), caches ~25m,
# TD 2m, LIP wave ~50m, evals ~5m.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_split
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_pldmdyna2.log
PLDM=/workspace/models/PLDM_OgBench_lewm
WMD=/workspace/models/dyna_pldm_5050
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
DF1F=/workspace/caches/dynapldm_full_fs1.pt
DF1=/workspace/caches/dynapldm_tr8000_fs1.pt
DF5=/workspace/caches/dynapldm_tr8000_fs5.pt
DTD=/workspace/metrics/dynapldm_TD.pt
MIX=$D/mix_pldm_5050.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; COLLECT_RANGE="0:${EPHI}"
AMAX=4.5; MW=0.1; ALR=3e-4; ALRF=3e-5
WIN=mw01_a45_lr3e-4          # PRE actors: lip4_pldm_${WIN}_s{0,1,2}.pt
SEEDS="0 1 2"; DRAWS="42 43 44"; NCALL=12; NGPU=8
mkdir -p "$D" "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

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

log "=== DYNA ON PLDM (pid $$) ==="
for f in "$QF1" "$QF5" "$QTD" "$AH5" "$PLDM/weights.pt" "$PLDM/config.json" \
         /workspace/build_dyna_mix.py /workspace/filter_cache_eprange.py; do
  [ -e "$f" ] || die "missing $f"
done
for s in $SEEDS; do [ -f "/workspace/actors/lip4_pldm_${WIN}_s${s}.pt" ] || die "PRE actor s$s missing"; done
grep -q "SWM_RECORD_OUTCOME" stable_worldmodel/world/world.py || die "world.py lacks the outcome recorder"
log "P0: preflight OK (PRE = $WIN, amax $AMAX)"
[ "${1:-}" = check ] && { log "check mode"; exit 0; }

# ------------------------------------------------ P1 complete the PRE arm
log "P1: completing PRE to 3 seeds x 3 draws"
g=0
for s in $SEEDS; do for d in $DRAWS; do
  ev "$g" "grid_${WIN}_s${s}_e${d}" "$PLDM" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_pldm_${WIN}_s${s}.pt" &
  g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done; wait
log "P1: PRE complete"

# ------------------------------------------------------- P2 collection 8-wide
if [ ! -f "$D/PLDM_COLLECT_DONE" ]; then
  log "P2: collection, ${NCALL} calls x 3 actors, terminate_at_goal=False, 8-wide"
  g=0
  for i in $(seq 0 $((NCALL-1))); do for a in $SEEDS; do
    rec="$D/onpolicy_pldm_a${a}_c${i}.lance"     # one lance PER CALL: concurrent
    lg="$L/pldmcol_a${a}_c${i}.log"              # appends to one lance would race
    grep -q "kept=" "$lg" 2>/dev/null && continue
    (
      SWM_RECORD_PATH=$rec SWM_RECORD_OUTCOME=1 \
      CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 10800 \
      python3 "$P/eval_wm.py" --config-name cube seed=$((4000 + a*100 + i)) \
        eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
        eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=50 \
        "+eval.ep_range=$COLLECT_RANGE" world.terminate_at_goal=False \
        policy="$PLDM" solver=lip \
        "solver.actor_path=/workspace/actors/lip4_pldm_${WIN}_s${a}.pt" \
        output.filename="pldmcol_a${a}_c${i}.txt" > "$lg" 2>&1
      rl=$(grep -h "\[record\]" "$lg" | tail -1)
      echo "$rl" | grep -q "kept=50 dropped=0" || echo "  WARN a$a/c$i: $rl" >> "$DRV"
    ) &
    g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
  done; done; wait
  ok=$(grep -l "kept=50 dropped=0" "$L"/pldmcol_a*_c*.log 2>/dev/null | wc -l)
  [ "$ok" -ge 30 ] || die "only $ok/36 collection calls clean"
  log "P2: $ok/36 calls clean; success rates $(grep -ho 'kept_success=[0-9]*' "$L"/pldmcol_a*_c*.log | cut -d= -f2 | awk '{s+=$1;n++}END{printf "%.1f%% mean", 100*s/(n*50)}')"
  touch "$D/PLDM_COLLECT_DONE"
fi

# ------------------------------------------------------------- P3 mix
if [ ! -f "$MIX/.done" ]; then
  log "P3: 50/50 uniform-K mix, expert < $EPHI"
  python3 /workspace/build_dyna_mix.py --expert "$EXPERT" \
    --onpolicy $D/onpolicy_pldm_a*_c*.lance \
    --out "$MIX" --onpolicy-frac 0.5 --expert-ep-hi "$EPHI" \
    > "$L/pldm_mix.log" 2>&1 || { tail -5 "$L/pldm_mix.log"; die "mix failed"; }
  grep -q BUILD_MIX_DONE "$L/pldm_mix.log" || die "mix incomplete"
  grep -q "\[split\] expert restricted" "$L/pldm_mix.log" || die "expert slice NOT restricted"
  touch "$MIX/.done"
fi
log "P3: $(grep -h 'dup K=' "$L/pldm_mix.log" | tail -1)"

# --------------------------------------------------------- P4 WM fine-tune
if [ ! -f "$WMD/weights_epoch_1.pt" ]; then
  log "P4: fine-tune PLDM -> dyna_pldm_5050 (lr 1e-5, 2 epochs, epoch 1; ~3.5h)"
  INIT_WEIGHTS=$PLDM/weights.pt CUDA_VISIBLE_DEVICES=0 timeout 86400 python3 \
    scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
    "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
    "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
    trainer.devices=1 output_model_name=dyna_pldm_5050 subdir=dyna_pldm_5050 \
    +action_stats_pin=expert wandb.enabled=false > "$L/pldm_ft.log" 2>&1 \
    || { tail -15 "$L/pldm_ft.log"; die "fine-tune failed"; }
  CK=/workspace/swm_home/checkpoints/dyna_pldm_5050
  mkdir -p "$WMD"; cp "$CK/weights_epoch_1.pt" "$WMD/" || die "no weights_epoch_1.pt"
  cp "$PLDM/config.json" "$WMD/config.json" || die "no arch config"
fi
cmp -s "$WMD/config.json" "$PLDM/config.json" || die "packaged config differs from the PLDM arch config"
log "P4: WM_dyna_pldm ready"

# ------------------------------------------------------- P5 caches + TD
if [ ! -f "$DF1" ]; then
  log "P5: caching fs1 under the fine-tuned WM"
  [ -f "$DF1F" ] || CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$TRM/cache_latents.py" \
    --wm "$WMD" --dataset "$EXPERT" --out "$DF1F" --state-key privileged_block_0_pos \
    > "$L/pldm_cache.log" 2>&1 || { tail -5 "$L/pldm_cache.log"; die "cache failed"; }
  python3 /workspace/filter_cache_eprange.py "$DF1F" "$DF1" --lo 0 --hi "$EPHI" \
    > "$L/pldm_filter.log" 2>&1 || die "filter failed"
  grep -q FILTER_CACHE_DONE "$L/pldm_filter.log" || die "filter incomplete"
fi
[ -f "$DF5" ] || python3 "$TRM/subsample_cache.py" --in "$DF1" --out "$DF5" --frameskip 5 \
  > "$L/pldm_fs5.log" 2>&1 || die "fs5 failed"
[ -f "$DTD" ] || CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_metric.py" \
  --cache "$DF1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
  --steps 6000 --seed 0 --out "$DTD" > "$L/pldm_td.log" 2>&1 || die "TD failed"
log "P5: caches + TD ready"

# ----------------------------------------------------- P6 POST LIP actors
log "P6: POST actors (mw $MW, amax $AMAX, lr $ALR) on GPUs 0-2"
g=0; for s in $SEEDS; do
  out=/workspace/actors/lip4_dynapldm_s${s}.pt
  [ -f "$out" ] || CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$DF5" --cache-td "$DF1" --h5 "$AH5" --wm "$WMD" --init-value "$DTD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
    --mean-weight "$MW" --actor-lr "$ALR" --actor-lr-final "$ALRF" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --arch v4 --seed "$s" --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/pldm_lip_post_s${s}.log" 2>&1 &
  g=$((g+1))
done; wait
log "P6: POST actors done"

# ------------------------------------------------------------ P7 POST evals
log "P7: POST evals, 9 cells, 8-wide"
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" >/dev/null; do sleep 30; done
g=0
for s in $SEEDS; do for d in $DRAWS; do
  ev "$g" "dynapldm_s${s}_e${d}" "$WMD" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_dynapldm_s${s}.pt" &
  g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done; wait

# ------------------------------------------------------------------ P8 card
WIN="$WIN" python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import os, sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
win = os.environ["WIN"]
def seed_mean(pfx, s):
    vs = [rows.get(f"{pfx}_s{s}_e{d}") for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else mean(vs)
pre = [seed_mean(f"grid_{win}", s) for s in (0, 1, 2)]
post = [seed_mean("dynapldm", s) for s in (0, 1, 2)]
print("=== DYNA ON PLDM (held-out 8000:10000, EGL, amax 4.5, 3 seeds x 3 draws) ===")
for nm, ms in (("PRE  (frozen PLDM)", pre), ("POST (Dyna fine-tune)", post)):
    if all(m is not None for m in ms):
        print(f"  {nm:22s} {' '.join(f'{m:5.1f}' for m in ms)}   mean {mean(ms):.1f}")
if all(m is not None for m in pre + post):
    d = [a - b for a, b in zip(post, pre)]
    m = mean(d); sd = math.sqrt(sum((x-m)**2 for x in d)/2)
    t = m/(sd/math.sqrt(3)) if sd > 0 else float("inf")
    print(f"  Dyna on PLDM: delta {m:+.2f} sd {sd:.2f} t {t:.2f} (df=2) "
          f"{'significant' if t > 2.92 else 'NOT significant'} (one-sided 5%)")
    print(f"  reference -- Dyna on LeWM was +3.6 at 6 seeds (p~0.002).")
    print("  reading: comparable delta => Dyna transfers across bases; ~0 or negative")
    print("  => Dyna repairs distribution shift on a good base, not base capability.")
td = [rows.get(f"ref_cemtd_e{d}") for d in (42,43,44)]
if all(v is not None for v in td):
    print(f"  bar: TD+CEM {mean(td):.1f} | latent+CEM {mean([rows[f'ref_cem_e{d}'] for d in (42,43,44)]):.1f}")
print("PLDM_DYNA2_DONE")
PY
log "done"
