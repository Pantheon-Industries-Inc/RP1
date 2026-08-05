#!/bin/bash
# JOINT DYNA: run the Dyna loop on BOTH bases at ONE shared critic/data config.
#
# THE CONSTRAINT (user directive). LIP-side knobs may differ per base -- amax is
# genuinely base-specific (LeWM 1.6, PLDM 4.5; porting PLDM's to LeWM costs ~7
# pts). But the TD/critic and data-side knobs -- replay-prob, expand-weight,
# freeze-critic-frac, expectile, n-step, gamma, p-cross -- must be IDENTICAL on
# both. A knob that only works on one base is a base repair; the point of this
# run is to carry forward only what survives on both.
#
# WHAT VARIES AND WHAT DOES NOT:
#   per-base : amax (1.6 / 4.5), the WM, its caches, its TD teacher
#   shared   : freeze-critic-frac, replay-prob, expand-weight  <- SELECTED here
#   fixed    : expectile 0.1->0.03, n-step 50, gamma 1.0, p-cross default 0.3,
#              horizon 5, iters 8, steps 6000, arch v4, mean-weight 0.1
#   p-cross stays at its default by directive: the TD sweep measured 0.6 = +0.2
#   and 0.0 = -3.3, so the default is at the optimum and the knob is retired.
#
# ============================ PRE-REGISTERED SELECTION ======================
# Written BEFORE the sweep data exists. This campaign has produced a phantom
# +2.0 from one lucky seed and has twice had E_final pick a config that planned
# worse, so the choice is made by formula, not by eye.
#
# Candidate space: the 8 cells measured by the freeze ladders + winner cross,
#   {plain, winner} x freeze_at {300, 2000, 3000, 4800}
# where winner = --batch 256 --replay-prob 0.5 --expand-weight 1.0.
#
# Scoring. The bases sit at very different levels (~89 vs ~73), so a raw delta
# would let the higher-variance base dictate the choice. Each base's 8 cell
# means are therefore STANDARDISED within that base (z = (m - mean)/sd), and
#     score(cell) = min(z_LeWM(cell), z_PLDM(cell))
# The min is what encodes "holds for both": a cell that is excellent on one base
# and mediocre on the other cannot win, however large its average.
# Tie-break: (1) higher sum of z, (2) plain over winner, (3) smaller freeze_at
# -- i.e. ties resolve toward the simpler and cheaper configuration.
#
# Guard. The winner must also be no worse than the trainer default (plain @
# freeze_at 4800) on BOTH bases in raw points. If no cell clears that, the
# script falls back to the default config and says so in the card rather than
# shipping a joint config that is jointly worse than doing nothing.
#
# ================================ THE LOOP ==================================
# Byte-identical to the PLDM round-1/round-2 protocol so the deltas stay
# comparable to the banked Dyna results: episode-disjoint collection on 0-7999,
# terminate_at_goal=False, 50/50 expert:on-policy uniform-K, fine-tune at
# lr 1e-5 for 2 epochs with epoch 1 pre-registered, then fresh caches -> TD ->
# 3 LIP actors -> 9 held-out eval cells on 8000-9999.
#
# The two bases run their phases CONCURRENTLY (collection is 8-wide across both;
# the two fine-tunes take one GPU each), which roughly halves wall-clock. This
# is safe because per-GPU MUJOCO_EGL_DEVICE_ID pinning was validated to
# reproduce serial eval values exactly -- the old "evals must be sequential"
# rule was an unpinned-EGL artifact.
#
# PRE for each base is the selected cell's own already-measured score, so no
# baseline work is repeated.
#
# ~10-12h. Queued behind the ladders and the cross.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/joint_dyna
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_jointdyna.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; COLLECT_RANGE="0:${EPHI}"
SEEDS="0 1 2"; DRAWS="42 43 44"; NCALL=12; NGPU=8
mkdir -p "$D" "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== JOINT DYNA (pid $$) ==="
for f in "$AH5" /workspace/build_dyna_mix.py /workspace/filter_cache_eprange.py \
         "$CODE/scripts/train/lewm_expert.py" "$TRM/cache_latents.py" \
         "$TRM/subsample_cache.py"; do [ -e "$f" ] || die "missing $f"; done
log "P0: preflight OK"

# ---------------------------------------------------- P1 wait for the sweeps
log "P1: waiting for the freeze ladders + winner cross"
T0=$(date +%s)
until grep -q "WIN_FREEZE_CROSS_DONE" "$L/driver_wincross.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && die "cross never finished in 12h"
  sleep 300
done
while pgrep -f "train_lip_a[c]|eval_w[m]" >/dev/null; do sleep 60; done
log "P1: sweeps done, box quiet"

# ------------------------------------------------- P2 pre-registered selection
python3 - "$SUM" "$D/choice.env" 2>&1 <<'PY' | tee -a "$DRV"
import sys, math
sumf, outf = sys.argv[1], sys.argv[2]
rows = {}
for ln in open(sumf):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
def arm(a):
    o = []
    for s in (0, 1, 2):
        vs = [rows.get(f"{a}_s{s}_e{d}") for d in (42, 43, 44)]
        if all(v is not None for v in vs): o.append(mean(vs))
    return o
# cell key -> (lewm tag, pldm tag); cell = (winner?, freeze_at)
CELLS = [
 (("plain",  300), "lewmbase", "pfrz05"), (("plain", 2000), "frz2k", "pfrz2k"),
 (("plain", 3000), "frz3k",    "pfrz3k"), (("plain", 4800), "frz48", "pfrz48"),
 (("win",    300), "lewmwin2", "pwin05"), (("win",  2000), "lwin2k", "pwin2k"),
 (("win",   3000), "lwin3k",   "pwin3k"), (("win",  4800), "lwin48", "pwin48"),
]
mL, mP = {}, {}
for cell, lt, pt in CELLS:
    a, b = arm(lt), arm(pt)
    if a: mL[cell] = mean(a)
    if b: mP[cell] = mean(b)
common = [c for c, _, _ in CELLS if c in mL and c in mP]
print(f"  cells complete on BOTH bases: {len(common)}/8")
if len(common) < 4:
    print("  too few complete cells to select on; FALLING BACK to the default")
    common = []
def z(d, keys):
    vs = [d[k] for k in keys]; m = mean(vs)
    s = math.sqrt(sum((x-m)**2 for x in vs)/max(len(vs)-1, 1)) or 1.0
    return {k: (d[k]-m)/s for k in keys}
choice = ("plain", 4800); reason = "fallback: trainer default"
if common:
    zL, zP = z(mL, common), z(mP, common)
    defL, defP = mL.get(("plain", 4800)), mP.get(("plain", 4800))
    print(f"  {'cell':16s} {'LeWM':>7s} {'zL':>6s} {'PLDM':>7s} {'zP':>6s} "
          f"{'min z':>6s} {'>=default':>10s}")
    scored = []
    for c in sorted(common, key=lambda c: (c[0], c[1])):
        ok = (defL is None or mL[c] >= defL - 1e-9) and (defP is None or mP[c] >= defP - 1e-9)
        s = min(zL[c], zP[c])
        scored.append((s, zL[c]+zP[c], 0 if c[0] == "plain" else 1, c[1], c, ok))
        print(f"  {c[0]+'@'+str(c[1]):16s} {mL[c]:7.2f} {zL[c]:6.2f} {mP[c]:7.2f} "
              f"{zP[c]:6.2f} {s:6.2f} {'yes' if ok else 'no':>10s}")
    elig = [x for x in scored if x[5]]
    if elig:
        # rank: max min-z, then max sum-z, then plain over winner, then smaller freeze
        best = sorted(elig, key=lambda x: (-x[0], -x[1], x[2], x[3]))[0]
        choice = best[4]
        reason = (f"max min-z ({best[0]:+.2f}) among cells >= default on both bases")
    else:
        print("  NO cell beats the default on both bases -- falling back")
        reason = "fallback: no cell cleared the default on both bases"
kind, fa = choice
frac = fa / 6000.0
flags = "--batch 256 --replay-prob 0.5 --expand-weight 1.0" if kind == "win" else ""
ltag = dict((c, lt) for c, lt, _ in CELLS)[choice]
ptag = dict((c, pt) for c, _, pt in CELLS)[choice]
print(f"  SELECTED: {kind} @ freeze_at {fa} (frac {frac:.4f})  -- {reason}")
print(f"    shared flags: --freeze-critic-frac {frac:.4f} {flags or '(no extra flags)'}")
print(f"    collector actors: LeWM lip4_{ltag}_s*  |  PLDM lip4_{ptag}_s*")
with open(outf, "w") as f:
    f.write(f"SEL_KIND={kind}\nSEL_FA={fa}\nSEL_FRAC={frac:.4f}\n")
    f.write(f'SEL_FLAGS="{flags}"\nLTAG={ltag}\nPTAG={ptag}\n')
    f.write(f'SEL_REASON="{reason}"\n')
    f.write(f"PRE_L={mL.get(choice, float('nan')):.2f}\nPRE_P={mP.get(choice, float('nan')):.2f}\n")
print("JOINT_SELECT_DONE")
PY
grep -q JOINT_SELECT_DONE "$DRV" || die "selection failed"
# shellcheck disable=SC1090
. "$D/choice.env"
log "P2: shared config = freeze-critic-frac $SEL_FRAC ${SEL_FLAGS:-(plain)}"
for s in $SEEDS; do
  [ -f "/workspace/actors/lip4_${LTAG}_s${s}.pt" ] || die "collector actor lip4_${LTAG}_s${s}.pt missing"
  [ -f "/workspace/actors/lip4_${PTAG}_s${s}.pt" ] || die "collector actor lip4_${PTAG}_s${s}.pt missing"
done

# base|wm-dir|init-weights|amax|actor-tag|shortname
BASES=(
  "lewm|/workspace/models/v2WM|weights_epoch_22.pt|1.6|$LTAG|jl"
  "pldm|/workspace/models/PLDM_OgBench_lewm|weights.pt|4.5|$PTAG|jp"
)

# ------------------------------------------------------------ P3 collection
# 8-wide across BOTH bases in one queue. SWM_RECORD_OUTCOME=auto: outcome
# columns are recorded when the world exposes 'success' and silently skipped
# otherwise -- the 50/50 mix does not need them, so this must not be fatal.
log "P3: collection, both bases, ${NCALL} calls x 3 actors x 2 bases, 8-wide"
g=0
for e in "${BASES[@]}"; do
  IFS='|' read -r nm wm _iw _am tag sh <<< "$e"
  for i in $(seq 0 $((NCALL-1))); do for a in $SEEDS; do
    rec="$D/onpol_${sh}_a${a}_c${i}.lance"; lg="$L/jcol_${sh}_a${a}_c${i}.log"
    grep -q "kept=" "$lg" 2>/dev/null && continue
    (
      SWM_RECORD_PATH=$rec SWM_RECORD_OUTCOME=auto \
      CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 10800 \
      python3 "$P/eval_wm.py" --config-name cube seed=$((7000 + a*100 + i)) \
        eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
        eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=50 \
        "+eval.ep_range=$COLLECT_RANGE" world.terminate_at_goal=False \
        policy="$wm" solver=lip \
        "solver.actor_path=/workspace/actors/lip4_${tag}_s${a}.pt" \
        output.filename="jcol_${sh}_a${a}_c${i}.txt" > "$lg" 2>&1
    ) &
    g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
  done; done
done
wait
for e in "${BASES[@]}"; do
  IFS='|' read -r nm _w _iw _am _t sh <<< "$e"
  ok=$(grep -l "kept=50 dropped=0" "$L"/jcol_${sh}_a*_c*.log 2>/dev/null | wc -l)
  log "  $nm: $ok/36 clean collection calls"
  [ "$ok" -ge 30 ] || die "$nm collection too lossy ($ok/36)"
done

# ------------------------------------------------------- P4 mix + fine-tune
# The two fine-tunes run CONCURRENTLY on separate GPUs (trainer.devices=1 each).
log "P4: mixes + concurrent fine-tunes (~3.5h)"
gi=0
for e in "${BASES[@]}"; do
  IFS='|' read -r nm wm iw am tag sh <<< "$e"
  MIX=$D/mix_${sh}.lance
  if [ ! -f "$MIX/.done" ]; then
    python3 /workspace/build_dyna_mix.py --expert "$EXPERT" \
      --onpolicy $D/onpol_${sh}_a*_c*.lance \
      --out "$MIX" --onpolicy-frac 0.5 --expert-ep-hi "$EPHI" \
      > "$L/jmix_${sh}.log" 2>&1 || { tail -5 "$L/jmix_${sh}.log"; die "$nm mix failed"; }
    grep -q BUILD_MIX_DONE "$L/jmix_${sh}.log" || die "$nm mix incomplete"
    grep -q "\[split\] expert restricted" "$L/jmix_${sh}.log" || die "$nm expert slice NOT restricted"
    touch "$MIX/.done"
  fi
  log "  $nm mix: $(grep -h 'dup K=' "$L/jmix_${sh}.log" | tail -1)"
  OUT=/workspace/models/jdyna_${sh}
  if [ ! -f "$OUT/weights_epoch_1.pt" ]; then
    (
      INIT_WEIGHTS=$wm/$iw CUDA_VISIBLE_DEVICES=$gi timeout 86400 python3 \
        scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
        "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
        "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
        trainer.devices=1 output_model_name=jdyna_${sh} subdir=jdyna_${sh} \
        +action_stats_pin=expert wandb.enabled=false > "$L/jft_${sh}.log" 2>&1
      CK=/workspace/swm_home/checkpoints/jdyna_${sh}
      mkdir -p "$OUT"
      cp "$CK/weights_epoch_1.pt" "$OUT/" 2>/dev/null
      # ARCH config, not the training config the trainer writes next to the ckpt
      cp "$wm/config.json" "$OUT/config.json" 2>/dev/null
    ) &
    gi=$((gi+1))
  fi
done
wait
for e in "${BASES[@]}"; do
  IFS='|' read -r nm wm _iw _am _t sh <<< "$e"
  [ -f "/workspace/models/jdyna_${sh}/weights_epoch_1.pt" ] || \
    { tail -15 "$L/jft_${sh}.log"; die "$nm fine-tune produced no epoch-1 checkpoint"; }
  cmp -s "/workspace/models/jdyna_${sh}/config.json" "$wm/config.json" || die "$nm arch config mismatch"
  grep -h "lr-probe" "$L/jft_${sh}.log" | head -2 | sed "s/^/  $nm /" | tee -a "$DRV"
done
log "P4: both fine-tuned WMs ready"

# ------------------------------------------------------ P5 caches + TD teachers
log "P5: caches + TD, both bases in parallel"
gi=0
for e in "${BASES[@]}"; do
  IFS='|' read -r nm _w _iw am tag sh <<< "$e"
  (
    W=/workspace/models/jdyna_${sh}
    F1F=/workspace/caches/j${sh}_full_fs1.pt; F1=/workspace/caches/j${sh}_tr8000_fs1.pt
    F5=/workspace/caches/j${sh}_tr8000_fs5.pt; TD=/workspace/metrics/j${sh}_TD.pt
    if [ ! -f "$F1" ]; then
      [ -f "$F1F" ] || CUDA_VISIBLE_DEVICES=$gi timeout 28800 python3 "$TRM/cache_latents.py" \
        --wm "$W" --dataset "$EXPERT" --out "$F1F" --state-key privileged_block_0_pos \
        > "$L/jcache_${sh}.log" 2>&1
      python3 /workspace/filter_cache_eprange.py "$F1F" "$F1" --lo 0 --hi "$EPHI" \
        > "$L/jfilter_${sh}.log" 2>&1
    fi
    [ -f "$F5" ] || python3 "$TRM/subsample_cache.py" --in "$F1" --out "$F5" --frameskip 5 \
      > "$L/jfs5_${sh}.log" 2>&1
    # TD teacher: the shared knobs, identical on both bases by directive
    [ -f "$TD" ] || CUDA_VISIBLE_DEVICES=$gi timeout 28800 python3 "$P/train_metric.py" \
      --cache "$F1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
      --steps 6000 --seed 0 --out "$TD" > "$L/jtd_${sh}.log" 2>&1
  ) &
  gi=$((gi+1))
done
wait
for e in "${BASES[@]}"; do
  IFS='|' read -r nm _w _iw _am _t sh <<< "$e"
  grep -q FILTER_CACHE_DONE "$L/jfilter_${sh}.log" 2>/dev/null || die "$nm cache filter incomplete"
  [ -f "/workspace/metrics/j${sh}_TD.pt" ] || die "$nm TD missing"
done
log "P5: caches + TD ready"

# ------------------------------------------------------------- P6 POST actors
log "P6: POST actors at the SHARED config, 3 seeds x 2 bases"
g=0
for e in "${BASES[@]}"; do
  IFS='|' read -r nm _w _iw am tag sh <<< "$e"
  for s in $SEEDS; do
    out=/workspace/actors/lip4_jpost_${sh}_s${s}.pt
    [ -f "$out" ] && continue
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$g timeout 43200 python3 "$P/train_lip_ac.py" \
      --cache /workspace/caches/j${sh}_tr8000_fs5.pt \
      --cache-td /workspace/caches/j${sh}_tr8000_fs1.pt \
      --h5 "$AH5" --wm /workspace/models/jdyna_${sh} \
      --init-value /workspace/metrics/j${sh}_TD.pt \
      --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$am" \
      --actor-lr 3e-4 --actor-lr-final 3e-5 \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --arch v4 --seed "$s" --freeze-critic-frac "$SEL_FRAC" $SEL_FLAGS \
      --out "$out" --out-value "${out%.pt}_value.pt" > "$L/jpost_${sh}_s${s}.log" 2>&1 &
    g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
  done
done
wait
log "P6: POST actors done"

# ---------------------------------------------------------------- P7 evals
ev(){ # gpu name policy actor draw
  local gpu=$1 nm=$2 pol=$3 act=$4 d=$5
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$pol" solver=lip "solver.actor_path=$act" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
}
log "P7: 18 POST eval cells (PRE is the selected cell, already measured)"
g=0
for e in "${BASES[@]}"; do
  IFS='|' read -r nm _w _iw _am _t sh <<< "$e"
  for s in $SEEDS; do for d in $DRAWS; do
    ev "$g" "jpost_${sh}_s${s}_e${d}" "/workspace/models/jdyna_${sh}" \
       "/workspace/actors/lip4_jpost_${sh}_s${s}.pt" "$d" &
    g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
  done; done
done
wait
log "P7: evals done"

# ------------------------------------------------------------------ P8 card
CH="$D/choice.env" python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import os, sys, math
env = {}
for ln in open(os.environ["CH"]):
    k, v = ln.strip().split("=", 1); env[k] = v.strip('"')
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
def sd(xs):
    m = mean(xs); return math.sqrt(sum((x-m)**2 for x in xs)/max(len(xs)-1, 1))
def arm(a):
    o = []
    for s in (0, 1, 2):
        vs = [rows.get(f"{a}_s{s}_e{d}") for d in (42, 43, 44)]
        if all(v is not None for v in vs): o.append(mean(vs))
    return o
print("=== JOINT DYNA at ONE shared critic/data config (held-out 8000:10000, EGL) ===")
print(f"  shared: --freeze-critic-frac {env['SEL_FRAC']} {env['SEL_FLAGS'] or '(no extra flags)'}")
print(f"  chosen by: {env['SEL_REASON']}")
print(f"  per-base: amax 1.6 (LeWM) / 4.5 (PLDM); everything else identical")
print(f"  {'base':6s} {'PRE':>7s} {'POST s0':>8s} {'s1':>6s} {'s2':>6s} {'POST':>7s} "
      f"{'delta':>7s} {'t':>6s}")
for nm, sh, pre_tag in (("LeWM", "jl", env["LTAG"]), ("PLDM", "jp", env["PTAG"])):
    pre, post = arm(pre_tag), arm(f"jpost_{sh}")
    if not post: print(f"  {nm:6s}   (no complete POST seeds)"); continue
    pad = list(post) + [float('nan')]*(3-len(post))
    ps = f"{mean(pre):7.2f}" if pre else "      -"
    ts = "     -"; ds = "      -"
    if pre and len(pre) == 3 and len(post) == 3:
        d = [x-y for x, y in zip(post, pre)]; m, s = mean(d), sd(d)
        t = m/(s/math.sqrt(3)) if s > 0 else float("inf")
        ds = f"{m:+7.2f}"; ts = f"{t:6.2f}"
    print(f"  {nm:6s} {ps} {pad[0]:8.1f} {pad[1]:6.1f} {pad[2]:6.1f} "
          f"{mean(post):7.2f} {ds} {ts}")
print("  paired on training seed, df=2, crit t=2.92")
print("  banked Dyna references: LeWM +4.7 (t=7.00, p~0.02, replicated 3x, plateau ~92)")
print("                          PLDM round 1 +8.0 (t=2.11, n=3, not yet significant)")
print("  READING. Dyna held ~+4 on LeWM across three independent WMs, so a much")
print("  smaller gain here would mean the shared config already captured part of")
print("  what Dyna was recovering -- the two are corrections to the SAME mismatch")
print("  (a critic queried off the expert manifold). A gain that stacks on top")
print("  means they fix different things and compose.")
print("  Caveat: 3 seeds resolves ~3 pts; both PRE and POST are 3-seed means.")
print("JOINT_DYNA_DONE")
PY
log "done"
