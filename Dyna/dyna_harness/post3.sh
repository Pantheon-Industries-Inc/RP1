#!/bin/bash
# POST-Dyna at THREE seeds per base -- trimmed continuation of exp30_full.sh.
#
# WHY THIS EXISTS. exp30_full.sh's run_dyna trains stage E/F actors over
# SEEDS="0 1 2 3 4 5", i.e. 6 POST LIPv4 actors and 6 POST PWM actors per base.
# Only 3 per base are wanted. A running bash script cannot be edited in place
# (bash reads the file incrementally; an in-place edit makes the live shell
# execute garbage at its next read offset), so the 6-seed driver is stopped at
# the stage-D/E boundary and this script finishes the job at 3 seeds.
#
# WHAT IT ASSUMES. Nothing that it cannot verify or rebuild:
#   * the Dyna-fine-tuned WM  -- copied from swm_home if /workspace/models is
#     missing it (the 21:32 driver died between writing the checkpoint and
#     copying it, which is what caused the duplicate fine-tune in the first
#     place, so this path is deliberately handled rather than assumed)
#   * caches F1F/F1/F5 + the TD teacher -- rebuilt per base if absent OR if the
#     completeness marker is missing. A cache file whose FILTER_CACHE_DONE
#     marker is absent is treated as TRUNCATED and rebuilt, because the driver
#     may have been killed mid-write.
# Everything is idempotent against the same paths exp30_full uses, so a cache or
# TD teacher the old driver finished is reused, not recomputed.
#
# CONTENTION. Shares /tmp/f30slots with whatever else is on the box, and never
# rmdir's it at startup.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_post3.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"
DRAWS="42 43 44"; SEEDS="0 1 2"          # <-- the whole point of this script
OP="--batch 256 --replay-prob 0.5 --expand-weight 3.0"
NSLOT=8; NGPU=4; SLOTDIR=/tmp/f30slots
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
wm_post(){ echo /workspace/models/f30dyna_$1; }
lip_post(){ echo /workspace/actors/lip4_f30post_${1}_s${2}.pt; }

log "=== POST-Dyna @ 3 seeds (pid $$) transformers $(python3 -c 'import transformers;print(transformers.__version__)')"
[ "$(python3 -c 'import transformers;print(transformers.__version__)')" = "4.49.0" ] \
  || die "wrong transformers -- POST must share PRE's provenance"
pgrep -f "bash /workspace/exp30_full.sh" >/dev/null \
  && die "exp30_full.sh is still running -- stop it first or it will train seeds 3-5"

ev(){ # name wm asset solver draw
  local nm=$1 wm=$2 asset=$3 slv=$4 d=$5
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && return 0
  local slot; slot=$(acquire); local g=$(( slot % NGPU )) extra
  case "$slv" in
    lcem)  extra="solver=cem solver.n_steps=10" ;;
    ladam) extra="solver=adam" ;;
    tcem)  extra="solver=cem solver.n_steps=10 +metric=$asset" ;;
    tcem9k) extra="solver=cem +metric=$asset" ;;
    tadam) extra="solver=adam +metric=$asset" ;;
    lcem9k) extra="solver=cem" ;;
    lip)   extra="solver=lip solver.actor_path=$asset" ;;
    pwm)   extra="solver=pwm solver.actor_path=$asset solver.batch_size=10 plan_config.receding_horizon=1" ;;
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
  local NM=$1 OUT; OUT=$(wm_post "$NM")
  local WM; WM=$(wm_pre "$NM"); local AM; AM=$(amax_of "$NM")
  local F1F=/workspace/caches/f30${NM}_full_fs1.pt F1=/workspace/caches/f30${NM}_tr8000_fs1.pt
  local F5=/workspace/caches/f30${NM}_tr8000_fs5.pt TD=/workspace/metrics/f30${NM}_TD.pt
  local CK=/workspace/swm_home/checkpoints/f30dyna_${NM}/weights_epoch_1.pt

  # -- the fine-tuned WM -------------------------------------------------
  if [ ! -f "$OUT/weights_epoch_1.pt" ]; then
    [ -f "$CK" ] || die "$NM: no epoch-1 checkpoint at $CK"
    mkdir -p "$OUT"; cp "$CK" "$OUT/" || die "$NM: ckpt copy failed"
    cp "$WM/config.json" "$OUT/config.json" || die "$NM: arch config copy failed"
    log "[$NM] copied epoch-1 ckpt into $OUT"
  fi
  cmp -s "$OUT/config.json" "$WM/config.json" || die "$NM: arch config mismatch"
  local n; n=$(find "$OUT" -maxdepth 1 -name '*.pt' | wc -l)
  [ "$n" = 1 ] || die "$NM: $OUT holds $n .pt files, loader needs exactly 1"

  # -- caches (rebuild if the completeness marker is missing) ------------
  if [ ! -f "$F1" ] || ! grep -q FILTER_CACHE_DONE "$L/f30filt_${NM}.log" 2>/dev/null; then
    # No FILTER_CACHE_DONE marker means the previous build either never ran or
    # was interrupted. There is no way to prove a .pt cache is complete rather
    # than truncated, and silently reusing a truncated one would poison every
    # actor and eval downstream -- so the whole chain for this base is discarded
    # and rebuilt. Costs a rebuild; the alternative costs the results.
    log "[$NM] D: no FILTER_CACHE_DONE marker => discarding unverifiable caches and rebuilding"
    for stale in "$F1F" "$F1" "$F5"; do
      [ -f "$stale" ] && { log "[$NM]   rm $stale ($(du -h "$stale" | cut -f1))"; rm -f "$stale"; }
    done
    local slot; slot=$(acquire)
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 \
      "$TRM/cache_latents.py" --wm "$OUT" --dataset "$EXPERT" --out "$F1F" \
      --state-key privileged_block_0_pos > "$L/f30cache_${NM}.log" 2>&1
    release "$slot"
    [ -f "$F1F" ] || die "$NM: full cache build failed"
    python3 /workspace/filter_cache_eprange.py "$F1F" "$F1" --lo 0 --hi "$EPHI" \
      > "$L/f30filt_${NM}.log" 2>&1
    grep -q FILTER_CACHE_DONE "$L/f30filt_${NM}.log" || die "$NM: filter incomplete"
  fi
  [ -f "$F5" ] || python3 "$TRM/subsample_cache.py" --in "$F1" --out "$F5" --frameskip 5 \
    > "$L/f30fs5_${NM}.log" 2>&1 || die "$NM: fs5 build failed"
  if [ ! -f "$TD" ]; then
    log "[$NM] D: TD teacher (12k steps)"
    local slot; slot=$(acquire)
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_metric.py" \
      --cache "$F1" --learner td --head quasimetric --expectile 0.03 \
      --gamma 0.98 --n-step 50 --steps 12000 --seed 0 --out "$TD" > "$L/f30td_${NM}.log" 2>&1
    release "$slot"
  fi
  [ -f "$TD" ] || die "$NM: TD teacher missing"
  log "[$NM] D: caches + TD ready"

  # -- E: 3 POST LIPv4 actors -------------------------------------------
  log "[$NM] E: POST LIPv4 actors, seeds $SEEDS (expand 3.0, amax $AM)"
  for s in $SEEDS; do
    local out; out=$(lip_post "$NM" "$s")
    [ -f "$out" ] && { log "[$NM]   s$s present"; continue; }
    local slot; slot=$(acquire)
    ( # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
        --cache "$F5" --cache-td "$F1" --h5 "$AH5" --wm "$OUT" --init-value "$TD" \
        --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AM" \
        --actor-lr 3e-4 --actor-lr-final 3e-5 \
        --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
        --arch v4 --seed "$s" --freeze-critic-frac 0.5 --gamma 0.98 $OP \
        --out "$out" --out-value "${out%.pt}_value.pt" > "$L/f30post_${NM}_s${s}.log" 2>&1
      release "$slot" ) &
  done
  wait
  for s in $SEEDS; do [ -f "$(lip_post $NM $s)" ] || die "$NM: POST LIP actor s$s missing"; done

  # -- F: 3 POST PWM actors ---------------------------------------------
  log "[$NM] F: POST PWM actors, seeds $SEEDS"
  for s in $SEEDS; do
    local o=/workspace/actors/pwm_f30_post_${NM}_s${s}.pt
    [ -f "$o" ] && continue
    local slot; slot=$(acquire)
    ( CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_pwm_ac_cube.py" \
        --cache "$F5" --cache-td "$F1" --h5 "$AH5" --wm "$OUT" \
        --init-value "$(lip_post $NM $s | sed 's/\.pt$/_value.pt/')" \
        --amax "$AM" --seed "$s" --terminal \
        --replay-prob 0.5 --expand-weight 3.0 --freeze-at 0.5 \
        --out "$o" --out-value "${o%.pt}_value.pt" > "$L/pwmf30_post_${NM}_s${s}.log" 2>&1
      release "$slot" ) &
  done
  wait

  # -- G: POST evals, all planners, 3 seeds ------------------------------
  log "[$NM] G: POST evals"
  for d in $DRAWS; do
    ev "f30_lcem3k_post_${NM}_e${d}" "$OUT" "" lcem   "$d"
    ev "f30_ladam_post_${NM}_e${d}"  "$OUT" "" ladam  "$d"
    ev "f30_lcem_post_${NM}_e${d}"   "$OUT" "" lcem9k "$d"
  done
  for s in $SEEDS; do
    local a m; a=$(lip_post "$NM" "$s"); m=${a%.pt}_value.pt
    for d in $DRAWS; do
      ev "f30_lip_post_${NM}_s${s}_e${d}"    "$OUT" "$a" lip    "$d"
      ev "f30_tcem3k_post_${NM}_s${s}_e${d}" "$OUT" "$m" tcem   "$d"
      ev "f30_tcem_post_${NM}_s${s}_e${d}"   "$OUT" "$m" tcem9k "$d"
      ev "f30_tadam_post_${NM}_s${s}_e${d}"  "$OUT" "$m" tadam  "$d"
      ev "f30_pwm_post_${NM}_s${s}_e${d}"    "$OUT" \
         "/workspace/actors/pwm_f30_post_${NM}_s${s}.pt" pwm "$d"
    done
  done
  wait
  log "[$NM] === complete ==="
}

( do_base lewm ) &
( do_base pldm ) &
wait
log "P: both bases done -- PRE/POST card"

python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, statistics as st
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"):
        try: rows[k] = float(v)
        except ValueError: pass
S = (0, 1, 2)
def dr(p):
    vs = [rows.get("%s_e%d" % (p, d)) for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else st.mean(vs)
def sd(p):
    o = [dr(p % s) for s in S]; o = [v for v in o if v is not None]
    return (st.mean(o), st.stdev(o) if len(o) > 1 else None) if len(o) == 3 else (None, None)
ARMS = [("LIPv4", "lip", True), ("TD+CEM 3k", "tcem3k", True), ("TD+CEM 9k", "tcem", True),
        ("TD+Adam 3k", "tadam", True), ("PWM", "pwm", True),
        ("latent+CEM 3k", "lcem3k", False), ("latent+CEM 9k", "lcem", False),
        ("latent+Adam 3k", "ladam", False)]
print("\n=== PRE vs POST-Dyna, 3 seeds x 3 draws, h25, held-out 8000:10000 ===")
for base in ("lewm", "pldm"):
    print("\n  %s" % base.upper())
    print("    %-16s%10s%10s%10s" % ("planner", "PRE", "POST", "delta"))
    for lab, arm, seeded in ARMS:
        if seeded:
            a, asd = sd("f30_%s_pre_%s_s%%d" % (arm, base))
            b, bsd = sd("f30_%s_post_%s_s%%d" % (arm, base))
        else:
            a = dr("f30_%s_pre_%s" % (arm, base)); b = dr("f30_%s_post_%s" % (arm, base))
            asd = bsd = None
        f = lambda v: "%10.2f" % v if v is not None else "%10s" % "-"
        d = "%+10.2f" % (b - a) if (a is not None and b is not None) else "%10s" % "-"
        print("    %-16s%s%s%s" % (lab, f(a), f(b), d))
    # paired Dyna delta on the seeded rows
    for lab, arm, seeded in ARMS:
        if not seeded: continue
        ds = []
        for s in S:
            a, b = dr("f30_%s_pre_%s_s%d" % (arm, base, s)), dr("f30_%s_post_%s_s%d" % (arm, base, s))
            if a is not None and b is not None: ds.append(b - a)
        if len(ds) == 3:
            m, s_ = st.mean(ds), st.stdev(ds)
            t = m / (s_ / 3 ** 0.5) if s_ > 0 else float("inf")
            print("      paired %-14s %+6.2f  sd %5.2f  t %6.2f  %d/3 pos" % (lab, m, s_, t, sum(1 for x in ds if x > 0)))
print("\n  latent rows carry NO training seed (3 draws only, se ~3.5 pts) -- their")
print("  deltas are not resolvable and are printed for completeness only.")
print("POST3_DONE")
PY
log "done"
