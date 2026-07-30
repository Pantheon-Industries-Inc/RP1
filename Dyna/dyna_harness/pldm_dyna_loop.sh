#!/bin/bash
# DYNA LOOP ON PLDM -- fills the "LIP + Dyna fine-tune / PLDM" cell with the
# exact best-results protocol (episode-disjoint, terminate_at_goal=False
# collection by the base's own tuned actors, uniform-K 50/50, lr 1e-5 x 2
# epochs, epoch 1 pre-registered), 3 seeds throughout.
#
# Collection actors = lip4_pldm_a45_s{0,1,2} (PLDM's tuned operating point,
# amax 4.5, 3-seed 75.3). Seed band 3000+ (dsp used 1000+, s12k 2000+).
#
# PRE-REGISTERED caveat: the fine-tune runs lewm_expert.py's (LeWM) training
# objective on PLDM-origin weights -- arch is shape-identical (config diff:
# predictor.emb_dropout None vs 0.0 only, checked 07-30) but the pretraining
# objective differs. Expectation +0..+4 with a real chance of ~0/negative;
# either outcome completes the cross-base table honestly.
#
# ~1.4h collect + ~3.5-4.5h FT + ~25min caches/TD + ~50min wave + ~20min
# evals => card ~7h after launch.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_split
SUM=$R/summary_dynasplit.csv; DRV=$L/driver_pldmdyna.log
PLDM=/workspace/models/PLDM_OgBench_lewm; PW=$PLDM/weights.pt
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; COLLECT_RANGE="0:${EPHI}"
AMAX=4.5; SEEDS="0 1 2"; DRAWS="42 43 44"; NCALL=12
MIX=$D/mix_pldm_5050.lance
WMP=/workspace/models/dyna_pldm_5050
PF1F=/workspace/caches/pldmdyna_full_fs1.pt
PF1=/workspace/caches/pldmdyna_tr${EPHI}_fs1.pt
PF5=/workspace/caches/pldmdyna_tr${EPHI}_fs5.pt
PTD=/workspace/metrics/pldmdyna_TD.pt
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

# ------------------------------------------------------------------ P0 preflight
log "=== DYNA LOOP, PLDM BASE (pid $$) ==="
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 120; done
for s in $SEEDS; do
  [ -f "/workspace/actors/lip4_pldm_a45_s${s}.pt" ] || die "a45 actor s$s missing"
done
for f in "$PW" "$PLDM/config.json" "$AH5" /workspace/build_dyna_mix.py; do
  [ -e "$f" ] || die "missing $f"
done
[ -d "$EXPERT" ] || die "missing expert"
grep -q "SWM_RECORD_OUTCOME" stable_worldmodel/world/world.py || die "world.py lacks the outcome recorder"
log "P0: preflight OK"
[ "${1:-}" = check ] && { log "check mode: exiting"; exit 0; }

# ------------------------------------------------- P1 collection (a45 actors)
if [ ! -f "$D/PLDMDYNA_COLLECT_DONE" ]; then
  log "P1: collection, ${NCALL} calls x 3 a45 actors, terminate_at_goal=False"
  for i in $(seq 0 $((NCALL-1))); do for a in $SEEDS; do
    rec="$D/onpolicy_pldm_a${a}.lance"; lg="$L/pldmcol_a${a}_c${i}.log"
    grep -q "kept=" "$lg" 2>/dev/null && continue
    SWM_RECORD_PATH=$rec SWM_RECORD_OUTCOME=1 CUDA_VISIBLE_DEVICES=0 timeout 10800 \
    python3 "$P/eval_wm.py" --config-name cube seed=$((3000 + a*100 + i)) \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=50 \
      "+eval.ep_range=$COLLECT_RANGE" world.terminate_at_goal=False \
      policy="$PLDM" solver=lip \
      "solver.actor_path=/workspace/actors/lip4_pldm_a45_s${a}.pt" \
      output.filename="pldmcol_a${a}_c${i}.txt" > "$lg" 2>&1
    grep -q "ep_range 0:${EPHI}" "$lg" || die "call a$a/c$i: ep_range not applied"
    rec_line=$(grep -h "\[record\]" "$lg" | tail -1)
    echo "$rec_line" | grep -q "kept=50 dropped=0" || die "call a$a/c$i: keep/drop -- $rec_line"
    echo "$rec_line" | grep -q "kept_success=" || die "call a$a/c$i: label missing -- $rec_line"
    log "  a$a/c$i: $(echo "$rec_line" | grep -oE 'kept_success=[0-9]+')"
  done; done
  touch "$D/PLDMDYNA_COLLECT_DONE"
fi
log "P1: collection done"

# --------------------------------------------------------------- P2 mix (uniform K)
if [ ! -f "$MIX/.done" ]; then
  log "P2: 50/50 mix, uniform K, expert < $EPHI"
  python3 /workspace/build_dyna_mix.py --expert "$EXPERT" \
    --onpolicy $D/onpolicy_pldm_a0.lance $D/onpolicy_pldm_a1.lance $D/onpolicy_pldm_a2.lance \
    --out "$MIX" --onpolicy-frac 0.5 --expert-ep-hi "$EPHI" > "$L/pldm_mix.log" 2>&1 || die "mix failed"
  grep -q BUILD_MIX_DONE "$L/pldm_mix.log" || die "mix incomplete"
  grep -q "\[split\] expert restricted" "$L/pldm_mix.log" || die "expert slice NOT restricted"
  touch "$MIX/.done"
fi
log "P2: $(grep -h 'dup K=' "$L/pldm_mix.log" | tail -1)"

# ------------------------------------------------------------ P3 WM fine-tune
if [ ! -f "$WMP/weights_epoch_1.pt" ]; then
  log "P3: fine-tune PLDM -> dyna_pldm_5050 (LeWM objective on PLDM weights; pre-registered caveat)"
  INIT_WEIGHTS=$PW CUDA_VISIBLE_DEVICES=0 timeout 86400 python3 \
    scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
    "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
    "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
    trainer.devices=1 output_model_name=dyna_pldm_5050 subdir=dyna_pldm_5050 \
    +action_stats_pin=expert wandb.enabled=false > "$L/pldm_ft.log" 2>&1 || die "fine-tune failed"
  CK=/workspace/swm_home/checkpoints/dyna_pldm_5050
  mkdir -p "$WMP"
  cp "$CK/weights_epoch_1.pt" "$WMP/" || die "no weights_epoch_1.pt"
  cp "$PLDM/config.json" "$WMP/config.json" || die "no arch config"
fi
cmp -s "$WMP/config.json" "$PLDM/config.json" || die "config differs from PLDM arch config"
log "P3: WM_pldmdyna ready"

# ------------------------------------------------------- P4 caches + TD
if [ ! -f "$PF1" ]; then
  [ -f "$PF1F" ] || { log "P4: caching fs1 under WM_pldmdyna";
    CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$TRM/cache_latents.py" \
      --wm "$WMP" --dataset "$EXPERT" --out "$PF1F" --state-key privileged_block_0_pos \
      > "$L/pldm_cache_fs1.log" 2>&1 || die "cache failed"; }
  python3 /workspace/filter_cache_eprange.py "$PF1F" "$PF1" --lo 0 --hi "$EPHI" \
    > "$L/pldm_filter.log" 2>&1 || die "filter failed"
fi
[ -f "$PF5" ] || python3 "$TRM/subsample_cache.py" --in "$PF1" --out "$PF5" --frameskip 5 \
  > "$L/pldm_cache_fs5.log" 2>&1 || die "fs5 failed"
[ -f "$PTD" ] || CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/train_metric.py" \
  --cache "$PF1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
  --steps 6000 --seed 0 --out "$PTD" > "$L/pldm_td.log" 2>&1 || die "TD failed"
log "P4: caches + TD ready"

# -------------------------------------------- P5 POST actors (amax 4.5)
log "P5: POST actors, amax $AMAX, GPUs 0-2"
g=0; for s in $SEEDS; do
  out=/workspace/actors/lip4_pldmpost_s${s}.pt
  if [ ! -f "$out" ]; then
    CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
      --cache "$PF5" --cache-td "$PF1" --h5 "$AH5" --wm "$WMP" --init-value "$PTD" \
      --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$s" \
      --out "$out" --out-value "${out%.pt}_value.pt" \
      > "$L/pldm_lip_post_s${s}.log" 2>&1 || log "  post/s$s TRAIN FAILED" &
  fi
  g=$((g+1))
done; wait
log "P5: POST actors done"

# ---------------------------------------------------------------- P6 evals
log "P6: POST evals (PRE = pldm_a45 s0-2 rows, 75.3)"
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" >/dev/null; do sleep 60; done
for s in $SEEDS; do for d in $DRAWS; do
  nm="pldmpost_s${s}_e${d}"
  c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; continue; }
  actor=/workspace/actors/lip4_pldmpost_s${s}.pt
  [ -f "$actor" ] || { log "  $nm: actor missing, FAIL"; echo "${nm},FAIL" >> "$SUM"; continue; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$WMP" solver=lip "solver.actor_path=$actor" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" \
    || { log "  $nm: ep_range NOT APPLIED"; echo "${nm},FAIL" >> "$SUM"; continue; }
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
done; done

# ------------------------------------------------------------------- P7 card
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
def sm(arm, s):
    vs = [rows.get(f"{arm}_s{s}_e{d}") for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else sum(vs) / 3
print("=== PLDM DYNA CARD (held-out 8000:10000, amax 4.5, egl, 3 seeds) ===")
pre = [sm("pldm_a45", s) for s in (0, 1, 2)]
post = [sm("pldmpost", s) for s in (0, 1, 2)]
for nm, ms in (("PRE  (pldm_a45)", pre), ("POST (pldmpost)", post)):
    if all(m is not None for m in ms):
        print(f"  {nm}: {' '.join(f'{m:5.1f}' for m in ms)}   mean {sum(ms)/3:.1f}")
if all(m is not None for m in pre + post):
    d = [a - b for a, b in zip(post, pre)]
    m = sum(d)/3; sd = math.sqrt(sum((x-m)**2 for x in d)/2)
    t = m/(sd/math.sqrt(3)) if sd > 0 else float("inf")
    print(f"  Dyna on PLDM: delta {m:+.2f} sd {sd:.2f} t {t:.2f} (df=2)")
    print("  refs: TD+CEM 73.3 | LeWM Dyna delta at s0-2 was +4.6")
print("PLDM_DYNA_DONE")
PY
log "done"
