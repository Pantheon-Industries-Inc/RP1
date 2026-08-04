#!/bin/bash
# FULL DATA COLLECTION AT expand-weight 3.0 -- the adopted operating point.
#
# WHY THIS CONFIG. Of every arm measured, expand 3.0 is the best PLDM score among
# those that put LeWM above 90 at 3 seeds:
#     expand 3.0   LeWM 90.22  PLDM 83.33     (-1.78 PLDM for +1.55 LeWM)
#     replay 0.25  LeWM 90.00  PLDM 81.78
#     mw 0.0       LeWM 90.44  PLDM 80.89
#     bundle@300   LeWM 90.00  PLDM 79.56
# STATED CAVEAT, because it matters for how these numbers get used: expand 3.0
# is the ONLY one of those four that was re-measured at 6 seeds, and it fell to
# 89.89 (t=1.28). Adopted by decision at 3 seeds, not because it cleared a
# significance bar. Everything below is 3 LIP seeds x 3 eval draws.
#
# CONFIG (identical on both bases except amax):
#   --arch v4 --horizon 5 --iters 8 --steps 6000 --n-step 50
#   --freeze-critic-frac 0.5 --batch 256 --replay-prob 0.5 --expand-weight 3.0
#   --gamma 0.98 --expectile 0.1->0.03 --critic-lr 1e-3->1e-4
#   --actor-lr 3e-4->3e-5 --mean-weight 0.1     amax 1.6 (LeWM) / 4.5 (PLDM)
#   teacher: MRN quasimetric, expectile 0.03, gamma 0.98, n-step 50, 12k steps
#
# FOUR PLANNERS, ONE CRITIC. Every planner here is scored against the SAME
# critic -- the EMA teacher co-trained inside that arm's own LIP run and saved
# as <actor>_value.pt. That is what makes the comparison an ablation of the
# PLANNER rather than of the value function:
#   LIP      solver=lip     8 iterative refinements, actor backprop through WM
#   TD+CEM   solver=cem     sampling search scored by the critic, no actor
#   TD+Adam  solver=adam    GradientSolver: 30 AdamW steps at lr 0.1 on the
#                           action sequence, 100 samples -- direct gradient
#                           descent on the same objective LIP amortises
#   PWM      solver=pwm     amortized feedforward policy, --terminal so its
#                           objective is LIPv4's (endpoint only, not the dense
#                           trajectory mean), and the SAME replay/expand as LIP
#
# PWM gets --terminal --replay-prob 0.5 --expand-weight 3.0 --freeze-at 0.5 per
# the directive, using the patched trainer that has those flags at all. It runs
# at each base's LIP amax so the actor architecture is the only difference.
# Its own default amax is 2.2 and remains unexplored -- an open confound on the
# PLDM side, stated not hidden.
#
# DYNA at this config: collect with the expand-3.0 actors themselves, so the
# collector matches what is being improved. Protocol byte-identical to the
# banked loop -- episode-disjoint 0-7999, terminate_at_goal=False, 50/50
# uniform-K, lr 1e-5 epoch 1 pre-registered, fresh caches -> TD -> 3 actors.
#
# Phase 1 (PRE downstream) needs no new world model and lands first.
# Phase 2 runs the Dyna loop and repeats all four planners on the POST models.
#
# 4 GPUs after the 2026-08-03 container rebuild, so 8 slots at 2/GPU.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/exp30
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_exp30.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; COLLECT_RANGE="0:${EPHI}"
DRAWS="42 43 44"; SEEDS="0 1 2"; NCALL=12
OP="--batch 256 --replay-prob 0.5 --expand-weight 3.0"
NSLOT=8; NGPU=4; SLOTDIR=/tmp/e30slots
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
# PRE LIP actors already exist from the replay/expand sweep
pre_actor(){ echo /workspace/actors/lip4_re_${1}_exp30_s${2}.pt; }

log "=== exp30 CAMPAIGN (pid $$): 4 planners x 2 bases x PRE/POST, 3x3 ==="
log "P0: waiting for the h100 teacher sweep (avoids contending on 4 GPUs)"
T0=$(date +%s)
until grep -q "H100_TD_DONE" "$L/driver_h100td.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 72000 ] && { log "WARN: proceeding after 20h"; break; }
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]|train_pwm_a[c]" >/dev/null; do sleep 60; done
. /workspace/td_winner.env
log "P0: box free; teacher $TD_ARM gamma $TD_GAMMA; op = $OP"
for nm in lewm pldm; do for s in $SEEDS; do
  [ -f "$(pre_actor $nm $s)" ] || die "PRE actor $(pre_actor $nm $s) missing"
  [ -f "$(pre_actor $nm $s | sed 's/\.pt$/_value.pt/')" ] || die "PRE critic for $nm/s$s missing"
done; done
[ -f "$P/train_pwm_ac_cube.py" ] || die "patched PWM trainer missing"
grep -q "replay-prob" "$P/train_pwm_ac_cube.py" || die "PWM trainer lacks replay/expand patch"

# ------------------------------------------------------------------ eval helper
ev(){ # name wm actor-or-metric solver draw
  local nm=$1 wm=$2 asset=$3 slv=$4 d=$5
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && return 0
  local slot; slot=$(acquire); local g=$(( slot % NGPU ))
  local extra
  case "$slv" in
    lip)  extra="solver=lip solver.actor_path=$asset" ;;
    pwm)  extra="solver=pwm solver.actor_path=$asset solver.batch_size=10 plan_config.receding_horizon=1" ;;
    cem)  extra="solver=cem +metric=$asset" ;;
    adam) extra="solver=adam +metric=$asset" ;;
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
    log "  $nm = ${sr:-FAIL}"
    release "$slot" ) &
}
pwm_train(){ # base wm c5 c1 td amax tag seed
  local nm=$1 wm=$2 c5=$3 c1=$4 td=$5 am=$6 tag=$7 s=$8
  local out=/workspace/actors/pwm_e30_${tag}_s${s}.pt
  [ -f "$out" ] && return 0
  local slot; slot=$(acquire)
  ( CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_pwm_ac_cube.py" \
      --cache "$c5" --cache-td "$c1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
      --amax "$am" --seed "$s" --terminal \
      --replay-prob 0.5 --expand-weight 3.0 --freeze-at 0.5 \
      --out "$out" --out-value "${out%.pt}_value.pt" > "$L/pwme30_${tag}_s${s}.log" 2>&1
    release "$slot" ) &
}

# ===================== PHASE 1: PRE downstream (no new WM) =================
log "P1: PRE -- TD+CEM and TD+Adam on each LIP run's OWN critic (36 cells)"
for nm in lewm pldm; do for s in $SEEDS; do
  m=$(pre_actor $nm $s | sed 's/\.pt$/_value.pt/')
  for d in $DRAWS; do
    ev "e30_cem_pre_${nm}_s${s}_e${d}"  "$(wm_pre $nm)" "$m" cem  "$d"
    ev "e30_adam_pre_${nm}_s${s}_e${d}" "$(wm_pre $nm)" "$m" adam "$d"
  done
done; done
wait
log "P2: PRE -- PWM terminal, replay 0.5 / expand 3.0 / freeze 0.5 (6 trains)"
for nm in lewm pldm; do for s in $SEEDS; do
  pwm_train "$nm" "$(wm_pre $nm)" "$(c5_pre $nm)" "$(c1_pre $nm)" \
    "$(eval echo \$TD_$(echo ${nm}pre | tr a-z A-Z))" "$(amax_of $nm)" "pre_${nm}" "$s"
done; done
wait
for nm in lewm pldm; do for s in $SEEDS; do for d in $DRAWS; do
  ev "e30_pwm_pre_${nm}_s${s}_e${d}" "$(wm_pre $nm)" \
     "/workspace/actors/pwm_e30_pre_${nm}_s${s}.pt" pwm "$d"
done; done; done
wait
log "P2: PRE downstream complete"

# ========================= PHASE 2: Dyna at expand 3.0 =====================
run_dyna(){ # base
  local NM=$1 SH=$1
  local BD=$D/$SH BDRV=$L/driver_e30dyna_${SH}.log
  local MIX=$BD/mix.lance OUT=/workspace/models/e30dyna_${SH}
  local F1F=/workspace/caches/e30${SH}_full_fs1.pt F1=/workspace/caches/e30${SH}_tr8000_fs1.pt
  local F5=/workspace/caches/e30${SH}_tr8000_fs5.pt TD=/workspace/metrics/e30${SH}_TD.pt
  local WM; WM=$(wm_pre "$NM"); local AM; AM=$(amax_of "$NM")
  mkdir -p "$BD"; rm -f "$BD/FAILED"
  blog(){ echo "[$(date -u +%m%d-%H:%M:%S)] [$NM] $*" | tee -a "$BDRV" >> "$DRV"; }
  bail(){ blog "FAILED: $*"; touch "$BD/FAILED"; exit 1; }

  blog "A: collection with the expand-3.0 actors, ${NCALL}x3 calls"
  for i in $(seq 0 $((NCALL-1))); do for a in $SEEDS; do
    local lg="$L/e30col_${SH}_a${a}_c${i}.log"
    grep -q "kept=" "$lg" 2>/dev/null && continue
    local slot; slot=$(acquire)
    ( SWM_RECORD_PATH=$BD/onpol_a${a}_c${i}.lance SWM_RECORD_OUTCOME=auto \
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) MUJOCO_EGL_DEVICE_ID=$(( slot % NGPU )) \
      timeout 21600 python3 "$P/eval_wm.py" --config-name cube seed=$((11000 + a*100 + i)) \
        eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
        eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=50 \
        "+eval.ep_range=$COLLECT_RANGE" world.terminate_at_goal=False \
        policy="$WM" solver=lip "solver.actor_path=$(pre_actor $NM $a)" \
        output.filename="e30col_${SH}_a${a}_c${i}.txt" > "$lg" 2>&1
      release "$slot" ) &
  done; done
  wait
  local ok; ok=$(grep -l "kept=50 dropped=0" "$L"/e30col_${SH}_a*_c*.log 2>/dev/null | wc -l)
  blog "A: $ok/36 clean"; [ "$ok" -ge 30 ] || bail "collection lossy ($ok/36)"

  if [ ! -f "$MIX/.done" ]; then
    blog "B: 50/50 uniform-K mix"
    python3 /workspace/build_dyna_mix.py --expert "$EXPERT" --onpolicy $BD/onpol_a*_c*.lance \
      --out "$MIX" --onpolicy-frac 0.5 --expert-ep-hi "$EPHI" > "$L/e30mix_${SH}.log" 2>&1 \
      || { tail -5 "$L/e30mix_${SH}.log" >> "$BDRV"; bail "mix failed"; }
    grep -q BUILD_MIX_DONE "$L/e30mix_${SH}.log" || bail "mix incomplete"
    grep -q "\[split\] expert restricted" "$L/e30mix_${SH}.log" || bail "expert slice NOT restricted"
    touch "$MIX/.done"
  fi
  blog "B: $(grep -h 'dup K=' "$L/e30mix_${SH}.log" | tail -1)"

  if [ ! -f "$OUT/weights_epoch_1.pt" ]; then
    blog "C: fine-tune (lr 1e-5, 2 epochs, epoch 1 pre-registered; ~3.5h)"
    local slot; slot=$(acquire)
    INIT_WEIGHTS=$WM/$(iw_pre "$NM") CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 86400 python3 \
      scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
      "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
      "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
      trainer.devices=1 output_model_name=e30dyna_${SH} subdir=e30dyna_${SH} \
      +action_stats_pin=expert wandb.enabled=false > "$L/e30ft_${SH}.log" 2>&1
    release "$slot"
    mkdir -p "$OUT"
    cp "/workspace/swm_home/checkpoints/e30dyna_${SH}/weights_epoch_1.pt" "$OUT/" 2>/dev/null \
      || { tail -15 "$L/e30ft_${SH}.log" >> "$BDRV"; bail "no epoch-1 ckpt"; }
    cp "$WM/config.json" "$OUT/config.json" || bail "no arch config"
  fi
  cmp -s "$OUT/config.json" "$WM/config.json" || bail "arch config mismatch"
  blog "C: fine-tuned WM ready"

  if [ ! -f "$F1" ]; then
    blog "D: caches"
    local slot; slot=$(acquire)
    [ -f "$F1F" ] || CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 \
      "$TRM/cache_latents.py" --wm "$OUT" --dataset "$EXPERT" --out "$F1F" \
      --state-key privileged_block_0_pos > "$L/e30cache_${SH}.log" 2>&1
    release "$slot"
    [ -f "$F1F" ] || bail "cache failed"
    python3 /workspace/filter_cache_eprange.py "$F1F" "$F1" --lo 0 --hi "$EPHI" > "$L/e30filt_${SH}.log" 2>&1
    grep -q FILTER_CACHE_DONE "$L/e30filt_${SH}.log" || bail "filter incomplete"
  fi
  [ -f "$F5" ] || python3 "$TRM/subsample_cache.py" --in "$F1" --out "$F5" --frameskip 5 \
    > "$L/e30fs5_${SH}.log" 2>&1 || bail "fs5 failed"
  if [ ! -f "$TD" ]; then
    local slot; slot=$(acquire)
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_metric.py" \
      --cache "$F1" --learner td --head quasimetric --expectile 0.03 \
      --gamma 0.98 --n-step 50 --steps 12000 --seed 0 --out "$TD" > "$L/e30td_${SH}.log" 2>&1
    release "$slot"
  fi
  [ -f "$TD" ] || bail "TD missing"
  blog "D: caches + TD ready"

  blog "E: POST LIP actors at expand 3.0 (3 seeds)"
  for s in $SEEDS; do
    local out=/workspace/actors/lip4_e30post_${SH}_s${s}.pt
    [ -f "$out" ] && continue
    local slot; slot=$(acquire)
    ( # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
        --cache "$F5" --cache-td "$F1" --h5 "$AH5" --wm "$OUT" --init-value "$TD" \
        --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AM" \
        --actor-lr 3e-4 --actor-lr-final 3e-5 \
        --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
        --arch v4 --seed "$s" --freeze-critic-frac 0.5 --gamma 0.98 $OP \
        --out "$out" --out-value "${out%.pt}_value.pt" > "$L/e30post_${SH}_s${s}.log" 2>&1
      release "$slot" ) &
  done
  wait
  for s in $SEEDS; do [ -f "/workspace/actors/lip4_e30post_${SH}_s${s}.pt" ] || bail "POST actor s$s missing"; done

  blog "F: POST PWM actors"
  for s in $SEEDS; do
    pwm_train "$NM" "$OUT" "$F5" "$F1" "$TD" "$AM" "post_${SH}" "$s"
  done
  wait

  blog "G: POST evals -- LIP, TD+CEM, TD+Adam, PWM (36 cells)"
  for s in $SEEDS; do
    local a=/workspace/actors/lip4_e30post_${SH}_s${s}.pt
    local m=${a%.pt}_value.pt
    for d in $DRAWS; do
      ev "e30_lip_post_${SH}_s${s}_e${d}"  "$OUT" "$a" lip  "$d"
      ev "e30_cem_post_${SH}_s${s}_e${d}"  "$OUT" "$m" cem  "$d"
      ev "e30_adam_post_${SH}_s${s}_e${d}" "$OUT" "$m" adam "$d"
      ev "e30_pwm_post_${SH}_s${s}_e${d}"  "$OUT" "/workspace/actors/pwm_e30_post_${SH}_s${s}.pt" pwm "$d"
    done
  done
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
def seeded(fmt):
    o = []
    for s in (0, 1, 2):
        vs = [rows.get(fmt.format(s=s, d=d)) for d in (42, 43, 44)]
        if all(v is not None for v in vs): o.append(mean(vs))
    return o
print("=== expand-weight 3.0 -- FULL TABLE (h25, held-out 8000:10000, EGL) ===")
print("    3 LIP seeds x 3 eval draws. Every planner scored against the SAME")
print("    critic: the EMA teacher co-trained inside that arm's own LIP run.")
print("    PWM: --terminal, replay 0.5, expand 3.0, freeze 0.5, LIP's amax.")
PLAN = [("LIP", "lip"), ("TD+CEM", "cem"), ("TD+Adam", "adam"), ("PWM", "pwm")]
PREF = {("lip","pre","lewm"): "reev_lewm_exp30_s{s}_e{d}",
        ("lip","pre","pldm"): "reev_pldm_exp30_s{s}_e{d}"}
def key(sl, ph, nm):
    if (sl, ph, nm) in PREF: return PREF[(sl, ph, nm)]
    return "e30_%s_%s_%s_s{s}_e{d}" % (sl, ph, nm)
print(f"\n  {'planner':10s}{'LeWM PRE':>10s}{'LeWM POST':>11s}{'d':>7s}"
      f"{'PLDM PRE':>10s}{'PLDM POST':>11s}{'d':>7s}")
for lab, sl in PLAN:
    cells, out = {}, f"  {lab:10s}"
    for nm in ("lewm", "pldm"):
        pre, post = seeded(key(sl, "pre", nm)), seeded(key(sl, "post", nm))
        cells[nm] = (pre, post)
        f = lambda x, w: (f"{mean(x):{w}.2f}" if len(x) == 3 else f"{'-':>{w}}")
        dd = "      -"
        if len(pre) == 3 and len(post) == 3:
            dd = f"{mean(post)-mean(pre):+7.2f}"
        out += f(pre, 10) + f(post, 11) + dd
    print(out)
print("\n  paired Dyna t-tests (df=2, crit 2.92):")
for lab, sl in PLAN:
    for nm in ("lewm", "pldm"):
        pre, post = seeded(key(sl, "pre", nm)), seeded(key(sl, "post", nm))
        if len(pre) != 3 or len(post) != 3: continue
        d = [x-y for x, y in zip(post, pre)]; m, s = mean(d), sd(d)
        t = m/(s/math.sqrt(3)) if s > 0 else float("inf")
        print(f"    {lab:8s} {nm:5s} {m:+6.2f}  t {t:6.2f}  "
              f"{'sig' if abs(t) > 2.92 else 'ns'}")
print("\n  reference at the previous operating point (expand 1.0):")
print("    LIP  LeWM 88.67 -> 92.00 (+3.33)   PLDM 85.11 -> 91.33 (+6.00)")
print("  READING. TD+Adam is the direct-optimisation control: 30 AdamW steps on")
print("  the action sequence against the same critic LIP amortises. LIP minus")
print("  TD+Adam is what the learned refiner buys over plain gradient descent;")
print("  LIP minus TD+CEM is what it buys over sampling search; LIP minus PWM is")
print("  what iteration buys over one forward pass.")
print("  CAVEATS. 3 seeds (~3 pts). expand 3.0 was 90.22 at 3 seeds and 89.89 at")
print("  6 -- adopted by decision, not on significance. PWM runs at LIP's amax,")
print("  not its own default 2.2, which is still an open confound on PLDM.")
print("EXP30_DONE")
PY
log "done"
