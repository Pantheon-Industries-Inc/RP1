#!/bin/bash
# FULL PIPELINE AT expand-weight 3.0 -- five planners x two bases x PRE/POST-Dyna,
# all re-measured under ONE provenance (transformers 4.49.0).
#
# WHY expand 3.0. It is the config where LIPv4 clears 90 on LeWM, and it now has
# two independent confirmations at the correct transformers version:
#     run 1   89.33 90.67 90.67 -> 90.22
#     retest  89.33 90.67 90.00 -> 90.00     (fresh trains, seeds reproduce to the cell)
#     [5.14.1 87.11 -- contaminated, discarded]
# PLDM at the same config is 83.33, i.e. -1.78 against the expand-1.0 point
# (85.11) in exchange for +1.55 on LeWM.
#
# WHY EVERYTHING IS RE-MEASURED. An unpinned reinstall on 2026-08-03 put
# transformers 5.14.1 on the box between 07:20 and 18:32. Evals reproduce
# exactly across versions (a banked LIP cell returned 90.0 to the digit), but
# TRAINING shifted by -3.11 on LeWM uniformly across seeds. So:
#     valid   anything trained on or before Aug 2, and any eval whenever run
#     invalid the 6 PWM actors from exp30 Phase 1 (trained 15:40-16:40 Aug 3)
# Rather than track that per row, every cell below is re-scored under one naming
# scheme (f30_*) so the finished table has a single provenance. Cells are cheap
# (~2 min at h25); only trains are expensive, and only PWM is retrained.
#
# THE FIVE PLANNERS. Latent rows use NO critic and NO actor, so they are a
# function of the world model alone and are identical at any expand-weight --
# they are included per base/phase for completeness of the column, not because
# expand 3.0 changes them.
#     latent + CEM     solver=cem                     parameter-free
#     latent + Adam    solver=adam                    parameter-free  <- NEW, never run
#     TD + CEM         solver=cem  +metric=<critic>    sampling on the learned metric
#     TD + Adam        solver=adam +metric=<critic>    30 AdamW steps, same metric
#     PWM              solver=pwm  --terminal          amortized, LIPv4's objective
#     LIPv4 (RLP)      solver=lip                      learned refinement, K=8
# TD/PWM/LIP all score against the SAME critic per seed: the EMA teacher
# co-trained inside that seed's LIPv4 run (<actor>_value.pt). That is what makes
# the row-to-row comparison an ablation of the planner and nothing else.
#
# PWM: --terminal (so its objective IS LIPv4's, endpoint only) with the same
# replay 0.5 / expand 3.0 / freeze 0.5, at each base's LIP amax. Its own default
# amax is 2.2 and stays unexplored -- an open confound on PLDM, stated not hidden.
#
# DYNA: collection and mixes from the failed 2026-08-03 run are intact on disk
# (36/36 lances per base) and are reused, so this resumes at the fine-tune. The
# collector actors are the expand-3.0 LIPv4 actors themselves.
#
# 4 GPUs after the container rebuild. 18 trains + 2 fine-tunes + ~168 cells, ~9h.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/exp30
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_f30.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; COLLECT_RANGE="0:${EPHI}"
DRAWS="42 43 44"; SEEDS="0 1 2 3 4 5"; NCALL=12
OP="--batch 256 --replay-prob 0.5 --expand-weight 3.0"
NSLOT=8; NGPU=4; SLOTDIR=/tmp/f30slots
mkdir -p "$D" "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
rmdir "$SLOTDIR"/* 2>/dev/null || true

wm_pre(){  case "$1" in lewm) echo /workspace/models/v2WM ;; *) echo /workspace/models/PLDM_OgBench_lewm ;; esac; }
iw_pre(){  case "$1" in lewm) echo weights_epoch_22.pt ;; *) echo weights.pt ;; esac; }
c5_pre(){  case "$1" in lewm) echo /workspace/caches/v2_tr8000_fs5.pt ;; *) echo /workspace/caches/pldm_tr8000_fs5.pt ;; esac; }
c1_pre(){  case "$1" in lewm) echo /workspace/caches/v2_tr8000_fs1.pt ;; *) echo /workspace/caches/pldm_tr8000_fs1.pt ;; esac; }
amax_of(){ case "$1" in lewm) echo 1.6 ;; *) echo 4.5 ;; esac; }
# PRE LIP actors: from the replay/expand sweep, trained Aug 2 under 4.49 -- valid
# seeds 0-2 come from the replay/expand sweep, 3-5 from stage B. Both were
# trained on Aug 2 under transformers 4.49, i.e. before the 06:00 Aug 3 rebuild.
lip_pre(){ if [ "$2" -le 2 ]; then echo /workspace/actors/lip4_re_${1}_exp30_s${2}.pt
           else echo /workspace/actors/lip4_bcd_${1}_exp30_s${2}.pt; fi; }
wm_post(){ echo /workspace/models/f30dyna_${1}; }
lip_post(){ echo /workspace/actors/lip4_f30post_${1}_s${2}.pt; }

log "=== exp30 FULL (pid $$) transformers $(python3 -c 'import transformers;print(transformers.__version__)')"
[ "$(python3 -c 'import transformers;print(transformers.__version__)')" = "4.49.0" ] \
  || die "wrong transformers -- this run exists to have ONE provenance"
for nm in lewm pldm; do for s in $SEEDS; do
  [ -f "$(lip_pre $nm $s)" ] || die "PRE LIP actor missing: $(lip_pre $nm $s)"
  [ -f "$(lip_pre $nm $s | sed 's/\.pt$/_value.pt/')" ] || die "PRE critic missing for $nm/s$s"
done; done
[ -f "$P/train_pwm_ac_cube.py" ] || die "patched PWM trainer missing"
grep -q "replay-prob" "$P/train_pwm_ac_cube.py" || die "PWM trainer lacks the replay/expand patch"
log "P0: preflight OK"

# ------------------------------------------------------------------ eval helper
ev(){ # name wm asset solver draw
  local nm=$1 wm=$2 asset=$3 slv=$4 d=$5
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && return 0
  local slot; slot=$(acquire); local g=$(( slot % NGPU )) extra
  case "$slv" in
    lcem)   extra="solver=cem solver.n_steps=10" ;;
    ladam)  extra="solver=adam" ;;
    tcem)   extra="solver=cem solver.n_steps=10 +metric=$asset" ;;
    tadam)  extra="solver=adam +metric=$asset" ;;
    lcem9k) extra="solver=cem" ;;
    tcem9k) extra="solver=cem +metric=$asset" ;;
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
pwm_train(){ # base wm c5 c1 td amax tag seed
  local nm=$1 wm=$2 c5=$3 c1=$4 td=$5 am=$6 tag=$7 s=$8
  local out=/workspace/actors/pwm_f30_${tag}_s${s}.pt
  [ -f "$out" ] && return 0
  local slot; slot=$(acquire)
  ( CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_pwm_ac_cube.py" \
      --cache "$c5" --cache-td "$c1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
      --amax "$am" --seed "$s" --terminal \
      --replay-prob 0.5 --expand-weight 3.0 --freeze-at 0.5 \
      --out "$out" --out-value "${out%.pt}_value.pt" > "$L/pwmf30_${tag}_s${s}.log" 2>&1
    release "$slot" ) &
}
# score all five planners on one world model
score_all(){ # phase base wm  (lip actors + critics resolved by phase)
  local ph=$1 nm=$2 wm=$3
  for d in $DRAWS; do
    # 3,000 queries (300x10, the benchmark's published iteration count) so the
    # CEM rows are compute-matched to Adam's 100x30 and the CEM-vs-Adam
    # difference is the optimizer rather than the budget.
    ev "f30_lcem3k_${ph}_${nm}_e${d}" "$wm" "" lcem  "$d"
    ev "f30_ladam_${ph}_${nm}_e${d}"  "$wm" "" ladam "$d"
    # 9,000 (300x30) retained as a budget ablation, not the headline row
    ev "f30_lcem_${ph}_${nm}_e${d}"   "$wm" "" lcem9k "$d"
  done
  for s in $SEEDS; do
    local a m
    if [ "$ph" = pre ]; then a=$(lip_pre "$nm" "$s"); else a=$(lip_post "$nm" "$s"); fi
    m=${a%.pt}_value.pt
    [ -f "$a" ] || { log "  WARN missing actor $a"; continue; }
    for d in $DRAWS; do
      ev "f30_lip_${ph}_${nm}_s${s}_e${d}"   "$wm" "$a" lip   "$d"
      ev "f30_tcem3k_${ph}_${nm}_s${s}_e${d}" "$wm" "$m" tcem   "$d"
      ev "f30_tcem_${ph}_${nm}_s${s}_e${d}"   "$wm" "$m" tcem9k "$d"
      ev "f30_tadam_${ph}_${nm}_s${s}_e${d}" "$wm" "$m" tadam "$d"
      ev "f30_pwm_${ph}_${nm}_s${s}_e${d}"   "$wm" \
         "/workspace/actors/pwm_f30_${ph}_${nm}_s${s}.pt" pwm "$d"
    done
  done
}

# ============================== PHASE 1: PRE ==============================
log "P1: PRE PWM actors (6 trains -- the only contaminated component)"
for nm in lewm pldm; do for s in $SEEDS; do
  pwm_train "$nm" "$(wm_pre $nm)" "$(c5_pre $nm)" "$(c1_pre $nm)" \
    "$(lip_pre $nm $s | sed 's/\.pt$/_value.pt/')" "$(amax_of $nm)" "pre_${nm}" "$s"
done; done
wait
log "P1: $(ls /workspace/actors/pwm_f30_pre_*_s[0-9].pt 2>/dev/null|wc -l)/6 PWM actors"
log "P2: PRE evals, 5 planners x 2 bases (84 cells)"
for nm in lewm pldm; do score_all pre "$nm" "$(wm_pre $nm)"; done
wait
log "P2: PRE complete"

# ========================= PHASE 2: Dyna at expand 3.0 ====================
run_dyna(){ # base
  local NM=$1 BD=$D/$1 BDRV=$L/driver_f30dyna_$1.log
  local MIX=$BD/mix.lance OUT; OUT=$(wm_post "$NM")
  local F1F=/workspace/caches/f30${NM}_full_fs1.pt F1=/workspace/caches/f30${NM}_tr8000_fs1.pt
  local F5=/workspace/caches/f30${NM}_tr8000_fs5.pt TD=/workspace/metrics/f30${NM}_TD.pt
  local WM; WM=$(wm_pre "$NM"); local AM; AM=$(amax_of "$NM")
  mkdir -p "$BD"; rm -f "$BD/FAILED"
  blog(){ echo "[$(date -u +%m%d-%H:%M:%S)] [$NM] $*" | tee -a "$BDRV" >> "$DRV"; }
  bail(){ blog "FAILED: $*"; touch "$BD/FAILED"; exit 1; }

  # collection + mix are reused from the 2026-08-03 run (both intact)
  local ok; ok=$(grep -l "kept=50 dropped=0" "$L"/e30col_${NM}_a*_c*.log 2>/dev/null | wc -l)
  blog "A: reusing banked collection, $ok/36 clean"
  [ "$ok" -ge 30 ] || bail "banked collection too lossy ($ok/36)"
  [ -f "$MIX/.done" ] || bail "banked mix absent"
  blog "B: $(grep -h 'dup K=' "$L/e30mix_${NM}.log" | tail -1)"

  if [ ! -f "$OUT/weights_epoch_1.pt" ]; then
    blog "C: fine-tune (lr 1e-5, 2 epochs, epoch 1 pre-registered; ~3.5h)"
    local slot; slot=$(acquire)
    INIT_WEIGHTS=$WM/$(iw_pre "$NM") CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 86400 python3 \
      scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
      "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
      "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
      trainer.devices=1 output_model_name=f30dyna_${NM} subdir=f30dyna_${NM} \
      +action_stats_pin=expert wandb.enabled=false > "$L/f30ft_${NM}.log" 2>&1
    release "$slot"
    mkdir -p "$OUT"
    cp "/workspace/swm_home/checkpoints/f30dyna_${NM}/weights_epoch_1.pt" "$OUT/" 2>/dev/null \
      || { tail -15 "$L/f30ft_${NM}.log" >> "$BDRV"; bail "no epoch-1 ckpt"; }
    cp "$WM/config.json" "$OUT/config.json" || bail "no arch config"
  fi
  cmp -s "$OUT/config.json" "$WM/config.json" || bail "arch config mismatch"
  blog "C: fine-tuned WM ready"

  if [ ! -f "$F1" ]; then
    blog "D: caches"
    local slot; slot=$(acquire)
    [ -f "$F1F" ] || CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 \
      "$TRM/cache_latents.py" --wm "$OUT" --dataset "$EXPERT" --out "$F1F" \
      --state-key privileged_block_0_pos > "$L/f30cache_${NM}.log" 2>&1
    release "$slot"
    [ -f "$F1F" ] || bail "cache failed"
    python3 /workspace/filter_cache_eprange.py "$F1F" "$F1" --lo 0 --hi "$EPHI" > "$L/f30filt_${NM}.log" 2>&1
    grep -q FILTER_CACHE_DONE "$L/f30filt_${NM}.log" || bail "filter incomplete"
  fi
  [ -f "$F5" ] || python3 "$TRM/subsample_cache.py" --in "$F1" --out "$F5" --frameskip 5 \
    > "$L/f30fs5_${NM}.log" 2>&1 || bail "fs5 failed"
  if [ ! -f "$TD" ]; then
    local slot; slot=$(acquire)
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_metric.py" \
      --cache "$F1" --learner td --head quasimetric --expectile 0.03 \
      --gamma 0.98 --n-step 50 --steps 12000 --seed 0 --out "$TD" > "$L/f30td_${NM}.log" 2>&1
    release "$slot"
  fi
  [ -f "$TD" ] || bail "TD missing"
  blog "D: caches + TD ready"

  blog "E: POST LIPv4 actors at expand 3.0 (3 seeds)"
  for s in $SEEDS; do
    local out; out=$(lip_post "$NM" "$s")
    [ -f "$out" ] && continue
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
  for s in $SEEDS; do [ -f "$(lip_post $NM $s)" ] || bail "POST LIP actor s$s missing"; done

  blog "F: POST PWM actors"
  for s in $SEEDS; do
    pwm_train "$NM" "$OUT" "$F5" "$F1" "$(lip_post $NM $s | sed 's/\.pt$/_value.pt/')" \
      "$AM" "post_${NM}" "$s"
  done
  wait

  blog "G: POST evals, 5 planners (42 cells)"
  score_all post "$NM" "$OUT"
  wait
  touch "$BD/DONE"; blog "=== $NM complete ==="
}
log "P3: two independent Dyna runs at expand 3.0"
( run_dyna lewm ) &
( run_dyna pldm ) &
wait

# ================================== card ===================================
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda x: sum(x)/len(x)
def sd(x):
    a = mean(x); return math.sqrt(sum((y-a)**2 for y in x)/max(len(x)-1, 1))
def draws(p):
    v = [rows.get(f"{p}_e{d}") for d in (42, 43, 44)]
    return mean(v) if all(x is not None for x in v) else None
SEEDS6 = (0, 1, 2, 3, 4, 5)
def seeded(p):
    o = [draws(f"{p}_s{s}") for s in SEEDS6]
    o = [x for x in o if x is not None]
    return o if len(o) == len(SEEDS6) else []
PLAN = [("latent + CEM", "lcem3k", False), ("latent + Adam", "ladam", False),
        ("TD + CEM", "tcem3k", True), ("TD + Adam", "tadam", True),
        ("PWM (terminal)", "pwm", True), ("LIPv4 (RLP)", "lip", True),
        ("[abl] latent+CEM 9k", "lcem", False), ("[abl] TD+CEM 9k", "tcem", True)]
print("=== expand-weight 3.0 -- FULL PIPELINE (h25, held-out 8000:10000, EGL) ===")
print("    transformers 4.49.0 throughout. 6 planner seeds x 3 draws for seeded")
print("    rows; latent rows are parameter-free (3 draws, no seed).")
print("    All TD/PWM/LIP rows score the SAME critic per seed: the EMA teacher")
print("    co-trained inside that seed's LIPv4 run.")
print(f"\n  {'planner':16s}{'LeWM pre':>10s}{'LeWM post':>11s}{'d':>7s}"
      f"{'PLDM pre':>10s}{'PLDM post':>11s}{'d':>7s}")
store = {}
for lab, tag, sd_ok in PLAN:
    line = f"  {lab:16s}"
    for nm in ("lewm", "pldm"):
        vals = {}
        for ph in ("pre", "post"):
            k = f"f30_{tag}_{ph}_{nm}"
            vals[ph] = seeded(k) if sd_ok else ([draws(k)] if draws(k) is not None else [])
        store[(tag, nm)] = vals
        f = lambda x, w: (f"{mean(x):{w}.2f}" if x else f"{'-':>{w}}")
        dd = f"{mean(vals['post'])-mean(vals['pre']):+7.2f}" if vals['pre'] and vals['post'] else "      -"
        line += f(vals['pre'], 10) + f(vals['post'], 11) + dd
    print(line)
print("\n  paired Dyna t-tests on seeded rows (df=5, crit 2.57):")
for lab, tag, sd_ok in PLAN:
    if not sd_ok: continue
    for nm in ("lewm", "pldm"):
        v = store[(tag, nm)]
        if len(v['pre']) != 3 or len(v['post']) != 3: continue
        d = [x-y for x, y in zip(v['post'], v['pre'])]; m, s = mean(d), sd(d)
        t = m/(s/math.sqrt(len(d))) if s > 0 else float("inf")
        print(f"    {lab:16s} {nm:5s} {m:+6.2f}  t {t:6.2f}  "
              f"{'sig' if abs(t) > 2.57 else 'ns'}")
print("\n  LIPv4 minus each other planner (pre-Dyna):")
for lab, tag, _ in PLAN[:-1]:
    o = f"    LIPv4 - {lab:16s}"
    for nm in ("lewm", "pldm"):
        a, b = store[("lip", nm)]['pre'], store[(tag, nm)]['pre']
        o += f"{mean(a)-mean(b):9.2f}" if a and b else f"{'-':>9s}"
    print(o)
print("\n  queries/decision: latent+CEM & TD+CEM 9,000 (300x30); Adam rows 3,000")
print("  (100x30); PWM 0 (one encoder + one MLP pass, rh=1); LIPv4 ~16.")
print("  reference at expand 1.0: LIPv4 88.67->92.00 (LeWM), 85.11->91.33 (PLDM)")
print("  CAVEATS. 6 planner seeds x 3 draws. expand 3.0 confirmed twice on LeWM at 4.49")
print("  (90.22, 90.00) but never at 6 seeds. PWM runs at LIP's amax, not its own")
print("  default 2.2 -- an open confound on PLDM. v2WM saw all 10k episodes, so")
print("  absolutes are upper bounds; Dyna deltas share the base and are clean.")
print("F30_DONE")
PY
log "done"
