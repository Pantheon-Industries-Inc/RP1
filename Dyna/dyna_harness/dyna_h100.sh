#!/bin/bash
# DYNA WITH h100 COLLECTION -- fixing a confound in the h100 result.
#
# THE CONFOUND. The per-base Dyna loop collected its on-policy data at
# goal_offset_steps=25 / budget 50, because that is the protocol everything in
# this campaign was tuned at. The fine-tuned WM therefore saw ONLY h25
# trajectories. Evaluating that WM at h100 and reporting
#     PLDM  LIP  h100:  PRE 83.11 -> POST 80.22   (-2.89)
# as "Dyna hurts at long horizon" was wrong: the loop was never run at long
# horizon. It is a statement about the COLLECTION DISTRIBUTION, not about Dyna.
#
# The same column already contained the evidence: PLDM POST TD+CEM at h100 went
# 63.33 -> 79.33, the LARGEST gain anywhere in the table. So the world model and
# critic did improve at h100 -- only the LIP actor regressed. That is an actor
# specialising to the distribution it was trained and collected on, which is
# exactly what a mismatched collection horizon would produce.
#
# THE FIX. Re-run the loop with collection at goal_offset 100 / budget 200, so
# the on-policy data matches the horizon we then evaluate at. Everything else is
# byte-identical to the h25 loop -- episode-disjoint 0-7999,
# terminate_at_goal=False, 50/50 expert:on-policy uniform-K, lr 1e-5 x 2 epochs
# with epoch 1 pre-registered, fresh caches -> TD -> 3 LIP actors -- so the only
# difference between this and the banked run is the collection horizon.
#
# INIT IS THE ORIGINAL BASE, not the h25-Dyna WM. That keeps PRE -> POST at h100
# a clean parallel of the h25 result rather than a second round stacked on a
# first, and it isolates "does Dyna work when collected at the right horizon"
# from "does the loop compound".
#
# THE 2x2 THIS PRODUCES. Each h100-collected WM is evaluated at BOTH horizons,
# and the h25-collected numbers are already banked, so the finished table is
#
#                        eval h25      eval h100
#     collect h25        92.00/91.33   85.33/80.22    <- banked
#     collect h100           ?             ?          <- this run
#
# which separates two claims that the current data cannot tell apart:
#   * "Dyna works, but specialises to its collection horizon" -- predicts the
#     diagonal is strong and the off-diagonal weak, in BOTH directions. The
#     h100-collected WM should then LOSE at h25.
#   * "h100 is just harder and Dyna cannot help there" -- predicts the h100
#     column stays weak regardless of what was collected.
# The reverse cell (collect h100, eval h25) is the discriminating one, which is
# why it is measured even though nobody asked for it.
#
# PRE references are already measured and are NOT recomputed:
#   LeWM  h25 88.67 (lwin3k)   h100 81.78 (pl_lip_lewmpre_h100)
#   PLDM  h25 85.33 (pwin3k)   h100 83.11 (pl_lip_pldmpre_h100)
#
# COST. h100 collection carries a 4x budget, so collection is ~85 min rather
# than 21. Per base: ~85 min collect + 3.5h fine-tune + 25 min caches/TD +
# ~50 min actors + evals. Both bases run independently and concurrently, ~6-7h.
#
# Per-base isolation as before: one driver log, one DONE/FAILED marker and one
# card each; bail() exits only its own subshell, so one base cannot stop the
# other.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_h100
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_dynah100.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; COLLECT_RANGE="0:${EPHI}"
SEEDS="0 1 2"; DRAWS="42 43 44"; NCALL=12
COL_OFF=100; COL_BUD=200          # <-- THE FIX: collect at the horizon we test
NSLOT=8; NGPU=8; SLOTDIR=/tmp/h1slots
# shared config selected by the pre-registered rule in dyna_perbase.sh
SEL_FRAC=0.5; SEL_FLAGS="--batch 256 --replay-prob 0.5 --expand-weight 1.0"
mkdir -p "$D" "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do
    for s in $(seq 0 $((NSLOT-1))); do
      mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }
    done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
rmdir "$SLOTDIR"/* 2>/dev/null || true

log "=== DYNA @ h100 COLLECTION (pid $$) ==="
for f in "$AH5" /workspace/build_dyna_mix.py /workspace/filter_cache_eprange.py \
         "$CODE/scripts/train/lewm_expert.py" "$TRM/cache_latents.py" \
         "$TRM/subsample_cache.py"; do [ -e "$f" ] || die "missing $f"; done

log "P0: waiting for the LR sweep and the PWM ladder"
T0=$(date +%s)
until grep -q "PERBASE_SWEEP_DONE" "$L/driver_perbase.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 57600 ] && { log "WARN: proceeding after 16h"; break; }
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_pwm_a[c]" >/dev/null; do sleep 120; done
[ -f /workspace/td_winner.env ] || die "td_winner.env absent"
# shellcheck disable=SC1091
. /workspace/td_winner.env
# The upgrade selected the teacher recipe; reproduce it for this run's OWN
# post-Dyna teacher, and match train_lip_ac.py's critic gamma to it (the LIP
# critic warm-starts from this teacher, so the two gammas must agree in scale).
case "$TD_ARM" in
  g98s12k)    TDFLAGS="--gamma 0.98 --n-step 50 --steps 12000" ;;
  g98s12kn25) TDFLAGS="--gamma 0.98 --n-step 25 --steps 12000" ;;
  *)          TDFLAGS="--gamma 1.0 --n-step 50 --steps 6000" ;;
esac
log "P0: box available; teacher recipe = $TD_ARM ($TDFLAGS), LIP --gamma $TD_GAMMA"
# CAVEAT: the collector actors (lwin3k/pwin3k) were trained under the OLD
# gamma-1.0 teacher. They are still the best-known collectors, but the POST
# actors below train under the upgraded one, so collector and POST are not
# gamma-matched. Stated rather than hidden.

# ============================================================================
run_base(){ # name wmdir initweights amax collector-tag short
  local NM=$1 WM=$2 IW=$3 AM=$4 TAG=$5 SH=$6
  local BD=$D/$SH BDRV=$L/driver_dynah1_${SH}.log
  local MIX=$BD/mix.lance OUT=/workspace/models/dyna_h1_${SH}
  local F1F=/workspace/caches/h1${SH}_full_fs1.pt F1=/workspace/caches/h1${SH}_tr8000_fs1.pt
  local F5=/workspace/caches/h1${SH}_tr8000_fs5.pt TD=/workspace/metrics/h1${SH}_TD.pt
  mkdir -p "$BD"; rm -f "$BD/FAILED"
  blog(){ echo "[$(date -u +%m%d-%H:%M:%S)] [$NM] $*" | tee -a "$BDRV" >> "$DRV"; }
  bail(){ blog "FAILED: $*"; touch "$BD/FAILED"; exit 1; }
  blog "=== DYNA on $NM, collection at goal_offset $COL_OFF / budget $COL_BUD ==="

  blog "A: h100 collection, ${NCALL} calls x 3 actors (4x budget => ~85 min)"
  for i in $(seq 0 $((NCALL-1))); do for a in $SEEDS; do
    local lg="$L/h1col_${SH}_a${a}_c${i}.log"
    grep -q "kept=" "$lg" 2>/dev/null && continue
    local slot; slot=$(acquire)
    (
      SWM_RECORD_PATH=$BD/onpol_a${a}_c${i}.lance SWM_RECORD_OUTCOME=auto \
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) MUJOCO_EGL_DEVICE_ID=$(( slot % NGPU )) \
      timeout 21600 python3 "$P/eval_wm.py" --config-name cube seed=$((9000 + a*100 + i)) \
        eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
        eval.goal_offset_steps=$COL_OFF eval.eval_budget=$COL_BUD eval.num_eval=50 \
        "+eval.ep_range=$COLLECT_RANGE" world.terminate_at_goal=False \
        policy="$WM" solver=lip \
        "solver.actor_path=/workspace/actors/lip4_${TAG}_s${a}.pt" \
        output.filename="h1col_${SH}_a${a}_c${i}.txt" > "$lg" 2>&1
      release "$slot"
    ) &
  done; done
  wait
  local ok; ok=$(grep -l "kept=50 dropped=0" "$L"/h1col_${SH}_a*_c*.log 2>/dev/null | wc -l)
  blog "A: $ok/36 clean collection calls"
  [ "$ok" -ge 30 ] || bail "collection too lossy ($ok/36)"

  if [ ! -f "$MIX/.done" ]; then
    blog "B: 50/50 uniform-K mix, expert < $EPHI"
    python3 /workspace/build_dyna_mix.py --expert "$EXPERT" \
      --onpolicy $BD/onpol_a*_c*.lance --out "$MIX" \
      --onpolicy-frac 0.5 --expert-ep-hi "$EPHI" > "$L/h1mix_${SH}.log" 2>&1 \
      || { tail -5 "$L/h1mix_${SH}.log" >> "$BDRV"; bail "mix failed"; }
    grep -q BUILD_MIX_DONE "$L/h1mix_${SH}.log" || bail "mix incomplete"
    grep -q "\[split\] expert restricted" "$L/h1mix_${SH}.log" || bail "expert slice NOT restricted"
    touch "$MIX/.done"
  fi
  blog "B: $(grep -h 'dup K=' "$L/h1mix_${SH}.log" | tail -1)"

  if [ ! -f "$OUT/weights_epoch_1.pt" ]; then
    blog "C: fine-tune from the ORIGINAL base (lr 1e-5, 2 epochs, epoch 1; ~3.5h)"
    local slot; slot=$(acquire)
    INIT_WEIGHTS=$WM/$IW CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 86400 python3 \
      scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
      "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
      "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
      trainer.devices=1 output_model_name=dyna_h1_${SH} subdir=dyna_h1_${SH} \
      +action_stats_pin=expert wandb.enabled=false > "$L/h1ft_${SH}.log" 2>&1
    release "$slot"
    mkdir -p "$OUT"
    cp "/workspace/swm_home/checkpoints/dyna_h1_${SH}/weights_epoch_1.pt" "$OUT/" 2>/dev/null \
      || { tail -15 "$L/h1ft_${SH}.log" >> "$BDRV"; bail "no epoch-1 checkpoint"; }
    cp "$WM/config.json" "$OUT/config.json" || bail "no arch config"
  fi
  cmp -s "$OUT/config.json" "$WM/config.json" || bail "packaged arch config differs"
  blog "C: fine-tuned WM ready"

  if [ ! -f "$F1" ]; then
    blog "D: caches under the h100-collected WM"
    local slot; slot=$(acquire)
    [ -f "$F1F" ] || CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 \
      "$TRM/cache_latents.py" --wm "$OUT" --dataset "$EXPERT" --out "$F1F" \
      --state-key privileged_block_0_pos > "$L/h1cache_${SH}.log" 2>&1
    release "$slot"
    [ -f "$F1F" ] || { tail -5 "$L/h1cache_${SH}.log" >> "$BDRV"; bail "cache failed"; }
    python3 /workspace/filter_cache_eprange.py "$F1F" "$F1" --lo 0 --hi "$EPHI" \
      > "$L/h1filter_${SH}.log" 2>&1
    grep -q FILTER_CACHE_DONE "$L/h1filter_${SH}.log" || bail "cache filter incomplete"
  fi
  [ -f "$F5" ] || python3 "$TRM/subsample_cache.py" --in "$F1" --out "$F5" --frameskip 5 \
    > "$L/h1fs5_${SH}.log" 2>&1 || bail "fs5 failed"
  if [ ! -f "$TD" ]; then
    local slot; slot=$(acquire)
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_metric.py" \
      --cache "$F1" --learner td --head quasimetric --expectile 0.03 \
      $TDFLAGS --seed 0 --out "$TD" > "$L/h1td_${SH}.log" 2>&1
    release "$slot"
  fi
  [ -f "$TD" ] || bail "TD teacher missing"
  blog "D: caches + TD ready"

  blog "E: POST actors at the shared config, 3 seeds"
  for s in $SEEDS; do
    local out=/workspace/actors/lip4_h1post_${SH}_s${s}.pt
    [ -f "$out" ] && continue
    local slot; slot=$(acquire)
    ( # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
        --cache "$F5" --cache-td "$F1" --h5 "$AH5" --wm "$OUT" --init-value "$TD" \
        --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AM" \
        --actor-lr 3e-4 --actor-lr-final 3e-5 \
        --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
        --arch v4 --seed "$s" --freeze-critic-frac "$SEL_FRAC" --gamma "$TD_GAMMA" $SEL_FLAGS \
        --out "$out" --out-value "${out%.pt}_value.pt" > "$L/h1post_${SH}_s${s}.log" 2>&1
      release "$slot"
    ) &
  done
  wait
  for s in $SEEDS; do
    [ -f "/workspace/actors/lip4_h1post_${SH}_s${s}.pt" ] || bail "POST actor s$s missing"
  done
  blog "E: POST actors done"

  # BOTH horizons: h100 is the target, h25 is the discriminating reverse cell
  blog "F: 18 eval cells -- h100 (target) and h25 (reverse transfer)"
  for hh in "h100|100|200" "h25|25|50"; do
    IFS='|' read -r hz off bud <<< "$hh"
    for s in $SEEDS; do for d in $DRAWS; do
      local nm="h1post_${SH}_s${s}_${hz}_e${d}"
      local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && continue
      local slot; slot=$(acquire)
      (
        CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) MUJOCO_EGL_DEVICE_ID=$(( slot % NGPU )) \
        timeout 14400 python3 "$P/eval_wm.py" --config-name cube seed=$d \
          eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
          eval.goal_offset_steps=$off eval.eval_budget=$bud "+eval.ep_range=$EVAL_RANGE" \
          policy="$OUT" solver=lip \
          "solver.actor_path=/workspace/actors/lip4_h1post_${SH}_s${s}.pt" \
          output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
        sr=""
        grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
          sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
        flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
        release "$slot"
      ) &
    done; done
  done
  wait
  blog "F: evals done"
  touch "$BD/DONE"
  blog "=== $NM h100-Dyna complete ==="
}

log "P1: two independent h100-collected Dyna runs"
( run_base LeWM /workspace/models/v2WM              weights_epoch_22.pt 1.6 lwin3k jl ) &
( run_base PLDM /workspace/models/PLDM_OgBench_lewm weights.pt          4.5 pwin3k jp ) &
wait

# ---------------------------------------------------------------- 2x2 card
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
def sd(xs):
    m = mean(xs); return math.sqrt(sum((x-m)**2 for x in xs)/max(len(xs)-1, 1))
def seeded(fmt):
    o = []
    for s in (0, 1, 2):
        vs = [rows.get(fmt.format(s=s, d=d)) for d in (42, 43, 44)]
        if all(v is not None for v in vs): o.append(mean(vs))
    return o
BASES = [("LeWM", "jl", "lwin3k",  "pl_lip_lewmpre_s{s}_h100_e{d}"),
         ("PLDM", "jp", "pwin3k",  "pl_lip_pldmpre_s{s}_h100_e{d}")]
print("=== DYNA x COLLECTION-HORIZON, 2x2 (held-out 8000:10000, EGL, 3 seeds) ===")
print("    rows = what the on-policy data was collected at")
print("    cols = what the policy was evaluated at")
for NM, SH, pre25, pre100fmt in BASES:
    p25, p100 = seeded(pre25 + "_s{s}_e{d}"), seeded(pre100fmt)
    c25_25 = seeded("pbpost_%s_s{s}_e{d}" % SH)              # banked h25 loop, h25 eval
    c25_100 = seeded("pl_lip_%spost_s{s}_h100_e{d}" % ("lewm" if SH == "jl" else "pldm"))
    c100_25 = seeded("h1post_%s_s{s}_h25_e{d}" % SH)
    c100_100 = seeded("h1post_%s_s{s}_h100_e{d}" % SH)
    f = lambda x: f"{mean(x):7.2f}" if x and len(x) == 3 else "      -"
    print(f"\n  --- {NM} ---")
    print(f"  {'':16s}{'eval h25':>10s}{'eval h100':>11s}")
    print(f"  {'PRE (no Dyna)':16s}{f(p25):>10s}{f(p100):>11s}")
    print(f"  {'collect h25':16s}{f(c25_25):>10s}{f(c25_100):>11s}   <- banked")
    print(f"  {'collect h100':16s}{f(c100_25):>10s}{f(c100_100):>11s}   <- this run")
    if p100 and c100_100 and len(p100) == 3 and len(c100_100) == 3:
        dd = [x-y for x, y in zip(c100_100, p100)]; m, s = mean(dd), sd(dd)
        t = m/(s/math.sqrt(3)) if s > 0 else float("inf")
        print(f"    h100 Dyna delta (collect h100): {m:+.2f} sd {s:.2f} t {t:.2f}"
              f"  {'SIGNIFICANT' if t > 2.92 else 'not significant'}")
    if p100 and c25_100 and len(p100) == 3 and len(c25_100) == 3:
        print(f"    for contrast, collecting at h25 gave "
              f"{mean(c25_100)-mean(p100):+.2f} at h100")
print()
print("  READING. If collect-h100 turns the h100 column positive, the earlier")
print("  -2.89 was a COLLECTION-HORIZON artifact and Dyna works at long horizon")
print("  when you collect there -- the loop is sound, it just specialises.")
print("  The discriminating cell is collect-h100 / eval-h25: if THAT is also")
print("  weak, specialisation is symmetric and the lesson is 'collect at the")
print("  horizon you deploy at'. If it stays strong, h100 data is simply richer")
print("  and should be preferred for collection regardless of deployment.")
print("  If the h100 column stays flat either way, long horizon is a genuine")
print("  limit of this stack rather than a data-distribution problem.")
print("  Caveat: 3 seeds, ~3 pts resolution. Collection used the h25-tuned")
print("  actors -- there is no separate h100-tuned actor to collect with, since")
print("  LIP replans every 25 primitive steps regardless of goal distance.")
print("DYNA_H100_DONE")
PY
log "done"
