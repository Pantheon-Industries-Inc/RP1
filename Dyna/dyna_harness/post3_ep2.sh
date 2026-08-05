#!/bin/bash
# POST-DYNA ON THE 2-EPOCH FINE-TUNE -- a labelled POST-HOC check, not a table row.
#
# WHY THIS IS POST-HOC. exp30_full pre-registered epoch 1 as the deployed
# checkpoint ("epoch 1 pre-registered" in the driver log) precisely so the epoch
# could not be chosen after seeing scores. weights_epoch_2.pt was exported anyway
# (72,299,657 B, both bases, 02:26-02:27) and never deployed. Everything this
# script produces must be reported as an epoch ablation alongside the
# pre-registered epoch-1 numbers -- never substituted for them because it wins.
#
# WHY CACHES ARE REBUILT. The f30${NM}_* caches and f30${NM}_TD.pt were built by
# cache_latents.py --wm f30dyna_${NM}, i.e. from the EPOCH-1 model's latents. A
# 2-epoch world model has different latents, so reusing them would train
# epoch-2 actors on epoch-1 representations -- a mismatch that would masquerade as
# an epoch effect. New namespace f30v2${NM}_* throughout.
#
# SCOPE: RLP (LIPv4) and the Reactive Policy (PWM), 3 seeds per base, as asked.
# latent+CEM is included at 3 draws per base because it needs no training and is
# the only arm that isolates WORLD-MODEL quality with no learned parts -- it is
# the control that says whether a second epoch improved the model at all.
# TD+CEM / TD+Adam are NOT included: their critic would have to be retrained on
# the epoch-2 cache to be a fair measurement, which is a separate run.
#
# Stages: A copy WM -> B caches+TD -> C 3 LIP actors -> D 3 PWM actors -> E evals.
# ~25 min caches+TD, ~90 min LIP, ~25 min PWM, ~20 min evals => ~2.5-3 h.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_ep2.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"
DRAWS="42 43 44"; SEEDS="0 1 2"
OP="--batch 256 --replay-prob 0.5 --expand-weight 3.0"
NSLOT=8; NGPU=4; SLOTDIR=/tmp/ep2slots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }

wm_pre(){  case "$1" in lewm) echo /workspace/models/v2WM ;; *) echo /workspace/models/PLDM_OgBench_lewm ;; esac; }
amax_of(){ case "$1" in lewm) echo 1.6 ;; *) echo 4.5 ;; esac; }
wm2(){ echo /workspace/models/f30dyna2_$1; }
lip2(){ echo /workspace/actors/lip4_f30post2_${1}_s${2}.pt; }
pwm2(){ echo /workspace/actors/pwm_f30_post2_${1}_s${2}.pt; }

log "=== POST-Dyna EPOCH 2 (post-hoc) pid $$  transformers $(python3 -c 'import transformers;print(transformers.__version__)')"
[ "$(python3 -c 'import transformers;print(transformers.__version__)')" = "4.49.0" ] \
  || die "wrong transformers -- must match the rest of exp30"
# self-safe pattern: the bracket stops this script's own command line matching
pgrep -f "post3[.]sh" >/dev/null && die "post3.sh still running"
for nm in lewm pldm; do
  [ -f "/workspace/swm_home/checkpoints/f30dyna_${nm}/weights_epoch_2.pt" ] \
    || die "$nm: no weights_epoch_2.pt to test"
done
log "A: preflight OK, both epoch-2 checkpoints present"

ev(){ # name wm asset solver draw
  local nm=$1 wm=$2 asset=$3 slv=$4 d=$5
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm banked ($c)"; return 0; }
  local slot; slot=$(acquire); local g=$(( slot % NGPU )) extra
  case "$slv" in
    lcem) extra="solver=cem solver.n_steps=10" ;;
    lip)  extra="solver=lip solver.actor_path=$asset" ;;
    pwm)  extra="solver=pwm solver.actor_path=$asset solver.batch_size=10 plan_config.receding_horizon=1" ;;
  esac
  ( # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 14400 \
      python3 "$P/eval_wm.py" --config-name cube seed=$d \
      eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
      eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
      policy="$wm" $extra output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
    sr=""
    grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
      sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
    flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
    log "  $nm = ${sr:-FAIL}"; release "$slot" ) &
}

do_base(){
  local NM=$1 OUT; OUT=$(wm2 "$NM")
  local WM; WM=$(wm_pre "$NM"); local AM; AM=$(amax_of "$NM")
  local CK=/workspace/swm_home/checkpoints/f30dyna_${NM}/weights_epoch_2.pt
  local F1F=/workspace/caches/f30v2${NM}_full_fs1.pt F1=/workspace/caches/f30v2${NM}_tr8000_fs1.pt
  local F5=/workspace/caches/f30v2${NM}_tr8000_fs5.pt TD=/workspace/metrics/f30v2${NM}_TD.pt

  # -- A: the epoch-2 WM, exactly one .pt so the loader is unambiguous --
  if [ ! -f "$OUT/weights_epoch_2.pt" ]; then
    mkdir -p "$OUT"; cp "$CK" "$OUT/" || die "$NM: ckpt copy failed"
    cp "$WM/config.json" "$OUT/config.json" || die "$NM: config copy failed"
    log "[$NM] A: epoch-2 WM staged"
  fi
  cmp -s "$OUT/config.json" "$WM/config.json" || die "$NM: arch config mismatch"
  local n; n=$(find "$OUT" -maxdepth 1 -name '*.pt' | wc -l)
  [ "$n" = 1 ] || die "$NM: $OUT holds $n .pt files, loader needs exactly 1"

  # -- B: caches from the EPOCH-2 latents + its own TD teacher --
  if [ ! -f "$F1" ] || ! grep -q FILTER_CACHE_DONE "$L/ep2filt_${NM}.log" 2>/dev/null; then
    log "[$NM] B: rebuilding caches from epoch-2 latents (unverifiable partials discarded)"
    for stale in "$F1F" "$F1" "$F5"; do
      [ -f "$stale" ] && { log "[$NM]   rm $stale"; rm -f "$stale"; }
    done
    local slot; slot=$(acquire)
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 \
      "$TRM/cache_latents.py" --wm "$OUT" --dataset "$EXPERT" --out "$F1F" \
      --state-key privileged_block_0_pos > "$L/ep2cache_${NM}.log" 2>&1
    release "$slot"
    [ -f "$F1F" ] || die "$NM: cache build failed"
    python3 /workspace/filter_cache_eprange.py "$F1F" "$F1" --lo 0 --hi "$EPHI" \
      > "$L/ep2filt_${NM}.log" 2>&1
    grep -q FILTER_CACHE_DONE "$L/ep2filt_${NM}.log" || die "$NM: filter incomplete"
  fi
  [ -f "$F5" ] || python3 "$TRM/subsample_cache.py" --in "$F1" --out "$F5" --frameskip 5 \
    > "$L/ep2fs5_${NM}.log" 2>&1 || die "$NM: fs5 build failed"
  if [ ! -f "$TD" ]; then
    log "[$NM] B: TD teacher on the epoch-2 cache (12k steps)"
    local slot; slot=$(acquire)
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_metric.py" \
      --cache "$F1" --learner td --head quasimetric --expectile 0.03 \
      --gamma 0.98 --n-step 50 --steps 12000 --seed 0 --out "$TD" > "$L/ep2td_${NM}.log" 2>&1
    release "$slot"
  fi
  [ -f "$TD" ] || die "$NM: TD teacher missing"
  log "[$NM] B: caches + TD ready"

  # -- C: 3 RLP actors on the epoch-2 model --
  log "[$NM] C: RLP actors, seeds $SEEDS (amax $AM)"
  for s in $SEEDS; do
    local out; out=$(lip2 "$NM" "$s"); [ -f "$out" ] && continue
    local slot; slot=$(acquire)
    ( # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
        --cache "$F5" --cache-td "$F1" --h5 "$AH5" --wm "$OUT" --init-value "$TD" \
        --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AM" \
        --actor-lr 3e-4 --actor-lr-final 3e-5 \
        --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
        --arch v4 --seed "$s" --freeze-critic-frac 0.5 --gamma 0.98 $OP \
        --out "$out" --out-value "${out%.pt}_value.pt" > "$L/ep2lip_${NM}_s${s}.log" 2>&1
      release "$slot" ) &
  done
  wait
  for s in $SEEDS; do [ -f "$(lip2 $NM $s)" ] || die "$NM: RLP actor s$s missing"; done

  # -- D: 3 PWM actors on the epoch-2 model --
  log "[$NM] D: PWM actors, seeds $SEEDS"
  for s in $SEEDS; do
    local o; o=$(pwm2 "$NM" "$s"); [ -f "$o" ] && continue
    local slot; slot=$(acquire)
    ( CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_pwm_ac_cube.py" \
        --cache "$F5" --cache-td "$F1" --h5 "$AH5" --wm "$OUT" \
        --init-value "$(lip2 $NM $s | sed 's/\.pt$/_value.pt/')" \
        --amax "$AM" --seed "$s" --terminal \
        --replay-prob 0.5 --expand-weight 3.0 --freeze-at 0.5 \
        --out "$o" --out-value "${o%.pt}_value.pt" > "$L/ep2pwm_${NM}_s${s}.log" 2>&1
      release "$slot" ) &
  done
  wait
  local np=0; for s in $SEEDS; do [ -f "$(pwm2 $NM $s)" ] && np=$((np+1)); done
  log "[$NM] D: $np/3 PWM actors (missing ones are skipped at eval, not faked)"

  # -- E: evals. RLP + PWM (3 seeds), plus latent+CEM as the model-quality control
  log "[$NM] E: evals"
  for d in $DRAWS; do
    ev "f30_lcem3k_post2_${NM}_e${d}" "$OUT" "" lcem "$d"
  done
  for s in $SEEDS; do
    for d in $DRAWS; do
      ev "f30_lip_post2_${NM}_s${s}_e${d}" "$OUT" "$(lip2 $NM $s)" lip "$d"
      [ -f "$(pwm2 $NM $s)" ] && ev "f30_pwm_post2_${NM}_s${s}_e${d}" "$OUT" "$(pwm2 $NM $s)" pwm "$d"
    done
  done
  wait
  log "[$NM] === complete ==="
}

( do_base lewm ) &
( do_base pldm ) &
wait

python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, statistics as st
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"):
        try: rows[k] = float(v)
        except ValueError: pass
D = (42, 43, 44); S = (0, 1, 2)
FL = {d: rows.get("f30_nomove_h25_e%d" % d) for d in D}
def dr(p):
    v = [rows.get("%s_e%d" % (p, d)) for d in D]
    return None if any(x is None for x in v) else st.mean(v)
def seeded(p):
    o = [dr(p % s) for s in S]; o = [x for x in o if x is not None]
    return (st.mean(o), st.stdev(o) if len(o) > 1 else None, len(o)) if o else (None, None, 0)
def hard(v):
    return None if v is None else 100.0 * (v - st.mean(FL.values())) / (100.0 - st.mean(FL.values()))
print("\n=== EPOCH ABLATION (post-hoc; epoch 1 was pre-registered) ===")
print("  %-14s%-6s%10s%10s%10s%10s" % ("arm", "base", "ep1", "ep2", "delta", "ep2 hard"))
for lab, arm, sd_ in (("RLP", "lip", True), ("Reactive PWM", "pwm", True),
                      ("latent+CEM 3k", "lcem3k", False)):
    for b in ("lewm", "pldm"):
        if sd_:
            a, asd, an = seeded("f30_%s_post_%s_s%%d" % (arm, b))
            c, csd, cn = seeded("f30_%s_post2_%s_s%%d" % (arm, b))
            extra = "" if (an == 3 and cn == 3) else "   [ep1 n=%d ep2 n=%d]" % (an, cn)
        else:
            a = dr("f30_%s_post_%s" % (arm, b)); c = dr("f30_%s_post2_%s" % (arm, b)); extra = ""
        f = lambda v: "%10.2f" % v if v is not None else "%10s" % "-"
        d = "%+10.2f" % (c - a) if (a is not None and c is not None) else "%10s" % "-"
        print("  %-14s%-6s%s%s%s%s%s" % (lab, b, f(a), f(c), d, f(hard(c)), extra))
        if sd_ and a is not None and c is not None:
            ds = []
            for s in S:
                x = dr("f30_%s_post_%s_s%d" % (arm, b, s)); y = dr("f30_%s_post2_%s_s%d" % (arm, b, s))
                if x is not None and y is not None: ds.append(y - x)
            if len(ds) > 1:
                m, sd = st.mean(ds), st.stdev(ds)
                t = m / (sd / len(ds) ** 0.5) if sd > 0 else float("inf")
                print("      paired ep2-ep1  %+.2f  sd %.2f  t %.2f  %d/%d pos"
                      % (m, sd, t, sum(1 for x in ds if x > 0), len(ds)))
print("\n  READING. latent+CEM has no learned parts, so its ep1->ep2 delta is the")
print("  only clean read on whether a SECOND fine-tuning epoch improved the WORLD")
print("  MODEL. If it is flat while RLP moves, the epoch changed exploitability,")
print("  not model quality. Epoch 1 remains the pre-registered deployed")
print("  checkpoint: report this as an ablation, do not substitute it.")
print("EP2_DONE")
PY
log "done"
