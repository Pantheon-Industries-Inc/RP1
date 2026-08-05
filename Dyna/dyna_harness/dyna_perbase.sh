#!/bin/bash
# DYNA, PER BASE -- two INDEPENDENT Dyna runs that happen to share one
# critic/data configuration.
#
# WHAT CHANGED FROM THE FIRST DRAFT AND WHY. The first version put both bases
# in one control flow, which made them a single experiment in three ways that
# all mattered: a `die` on one base's collection aborted the other base's Dyna
# too; the two shared a card; and a stall in one blocked the other's phases.
# Dyna is a per-base experiment -- LeWM's +4.7 and PLDM's +8.0 are separate
# results with separate baselines -- so the bases are now fully isolated:
#   * one driver log, one DONE/FAILED marker and one card EACH
#   * a failure in one base cannot stop the other (each runs in its own subshell
#     whose bail() exits only that subshell)
#   * phases are not synchronised; each base walks its own pipeline as fast as
#     its own work allows
# Nothing is pooled or averaged across bases anywhere.
#
# WHAT IS STILL SHARED, BY DIRECTIVE. Only the KNOB VALUES. The TD/critic and
# data-side settings -- freeze-critic-frac, replay-prob, expand-weight, plus the
# fixed expectile/n-step/gamma/p-cross -- are chosen ONCE and applied to both,
# because a knob that only works on one base is a base repair rather than a
# method. LIP-side settings stay per-base: amax 1.6 for LeWM, 4.5 for PLDM
# (porting PLDM's amax to LeWM costs ~7 pts).
#
# ============================ PRE-REGISTERED SELECTION ======================
# Written before the sweep data exists -- this campaign has produced a phantom
# +2.0 from one lucky seed and twice had E_final pick a config that planned
# worse, so the choice is a formula, not a judgement call.
#
# Candidates: the 8 cells from the freeze ladders + winner cross,
#   {plain, winner} x freeze_at {300, 2000, 3000, 4800},
# winner = --batch 256 --replay-prob 0.5 --expand-weight 1.0.
#
# The bases sit ~89 vs ~73, so raw deltas would let one base dictate the answer.
# Each base's 8 cell means are standardised WITHIN that base, and
#     score(cell) = min(z_LeWM, z_PLDM)
# The min encodes "holds for both": excellent-on-one, mediocre-on-the-other
# cannot win. Ties: higher sum of z, then plain over winner, then smaller
# freeze_at -- resolving toward the simpler, cheaper configuration.
# Guard: the choice must also be no worse than the trainer default (plain @
# 4800) in raw points on BOTH bases, else it falls back to the default and the
# card says so, rather than shipping something jointly worse than doing nothing.
# The selector was unit-tested on synthetic sweeps for all three paths (winner
# cell, plain cell, and the incomplete-data fallback).
#
# ================================ THE LOOP ==================================
# Per base, byte-identical to the banked PLDM round-1/round-2 protocol so the
# deltas stay comparable: episode-disjoint collection on 0-7999 with
# terminate_at_goal=False, 50/50 expert:on-policy uniform-K mix, fine-tune at
# lr 1e-5 for 2 epochs with epoch 1 pre-registered, then fresh caches -> TD ->
# 3 LIP actors -> 9 held-out cells on 8000-9999. PRE is the selected cell's own
# already-measured 3-seed score, so no baseline work is repeated and the paired
# t-test stays properly paired on training seed.
#
# SCHEDULING. All GPU work goes through one 8-slot pool (mkdir as mutex), so the
# two independent pipelines interleave without oversubscribing: while one base
# holds a slot for its 3.5h fine-tune, the other can use the remaining seven for
# collection. Per-GPU MUJOCO_EGL_DEVICE_ID pinning makes concurrent renders safe
# -- the old "evals must be sequential" rule was an unpinned-EGL artifact.
#
# ~8-10h after the sweeps finish.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan; TRM=$CODE/scripts/trm
L=/workspace/logs; R=/workspace/results; D=/workspace/dyna_perbase
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_dynaperbase.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; COLLECT_RANGE="0:${EPHI}"
SEEDS="0 1 2"; DRAWS="42 43 44"; NCALL=12
NSLOT=8; NGPU=8; SLOTDIR=/tmp/dynaslots
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
pgrep -f "train_lip_a[c]|eval_w[m]" >/dev/null || rmdir "$SLOTDIR"/* 2>/dev/null || true

log "=== DYNA PER BASE (pid $$) ==="
for f in "$AH5" /workspace/build_dyna_mix.py /workspace/filter_cache_eprange.py \
         "$CODE/scripts/train/lewm_expert.py" "$TRM/cache_latents.py" \
         "$TRM/subsample_cache.py"; do [ -e "$f" ] || die "missing $f"; done
log "P0: preflight OK"

log "P1: waiting for the freeze ladders + winner cross"
T0=$(date +%s)
until grep -q "WIN_FREEZE_CROSS_DONE" "$L/driver_wincross.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && die "cross never finished in 12h"
  sleep 300
done
while pgrep -f "train_lip_a[c]|eval_w[m]" >/dev/null; do sleep 60; done
log "P1: sweeps done, box quiet"

# ------------------------------------------------- P2 shared config selection
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
        best = sorted(elig, key=lambda x: (-x[0], -x[1], x[2], x[3]))[0]
        choice = best[4]
        reason = f"max min-z ({best[0]:+.2f}) among cells >= default on both bases"
    else:
        print("  NO cell beats the default on both bases -- falling back")
        reason = "fallback: no cell cleared the default on both bases"
kind, fa = choice
flags = "--batch 256 --replay-prob 0.5 --expand-weight 1.0" if kind == "win" else ""
ltag = dict((c, lt) for c, lt, _ in CELLS)[choice]
ptag = dict((c, pt) for c, _, pt in CELLS)[choice]
print(f"  SELECTED: {kind} @ freeze_at {fa} (frac {fa/6000:.4f})  -- {reason}")
print(f"    shared: --freeze-critic-frac {fa/6000:.4f} {flags or '(no extra flags)'}")
print(f"    collectors: LeWM lip4_{ltag}_s*  |  PLDM lip4_{ptag}_s*")
print("    applied to both bases; amax stays per-base (1.6 / 4.5)")
with open(outf, "w") as f:
    f.write(f"SEL_KIND={kind}\nSEL_FA={fa}\nSEL_FRAC={fa/6000:.4f}\n")
    f.write(f'SEL_FLAGS="{flags}"\nLTAG={ltag}\nPTAG={ptag}\nSEL_REASON="{reason}"\n')
print("SELECT_DONE")
PY
grep -q SELECT_DONE "$DRV" || die "selection failed"
# shellcheck disable=SC1090
. "$D/choice.env"
log "P2: shared config = --freeze-critic-frac $SEL_FRAC ${SEL_FLAGS:-(plain)}"
for s in $SEEDS; do
  [ -f "/workspace/actors/lip4_${LTAG}_s${s}.pt" ] || die "LeWM collector lip4_${LTAG}_s${s}.pt missing"
  [ -f "/workspace/actors/lip4_${PTAG}_s${s}.pt" ] || die "PLDM collector lip4_${PTAG}_s${s}.pt missing"
done

# ============================================================================
# ONE SELF-CONTAINED DYNA RUN. Everything below is per base. bail() exits only
# this subshell, so the sibling base keeps running whatever happens here.
# ============================================================================
run_base(){ # name wmdir initweights amax collector-tag short
  local NM=$1 WM=$2 IW=$3 AM=$4 TAG=$5 SH=$6
  local BD=$D/$SH BDRV=$L/driver_dyna_${SH}.log
  local MIX=$BD/mix.lance OUT=/workspace/models/dyna_pb_${SH}
  local F1F=/workspace/caches/pb${SH}_full_fs1.pt F1=/workspace/caches/pb${SH}_tr8000_fs1.pt
  local F5=/workspace/caches/pb${SH}_tr8000_fs5.pt TD=/workspace/metrics/pb${SH}_TD.pt
  mkdir -p "$BD"; rm -f "$BD/FAILED"
  blog(){ echo "[$(date -u +%m%d-%H:%M:%S)] [$NM] $*" | tee -a "$BDRV" >> "$DRV"; }
  bail(){ blog "FAILED: $*"; touch "$BD/FAILED"; exit 1; }
  blog "=== DYNA on $NM (amax $AM, shared cfg frac $SEL_FRAC ${SEL_FLAGS:-plain}) ==="

  # ---------------------------------------------------------- collection
  blog "A: collection, ${NCALL} calls x 3 actors, through the shared slot pool"
  for i in $(seq 0 $((NCALL-1))); do for a in $SEEDS; do
    local lg="$L/pbcol_${SH}_a${a}_c${i}.log"
    grep -q "kept=" "$lg" 2>/dev/null && continue
    local slot; slot=$(acquire)
    (
      SWM_RECORD_PATH=$BD/onpol_a${a}_c${i}.lance SWM_RECORD_OUTCOME=auto \
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) MUJOCO_EGL_DEVICE_ID=$(( slot % NGPU )) \
      timeout 10800 python3 "$P/eval_wm.py" --config-name cube seed=$((7000 + a*100 + i)) \
        eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
        eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=50 \
        "+eval.ep_range=$COLLECT_RANGE" world.terminate_at_goal=False \
        policy="$WM" solver=lip \
        "solver.actor_path=/workspace/actors/lip4_${TAG}_s${a}.pt" \
        output.filename="pbcol_${SH}_a${a}_c${i}.txt" > "$lg" 2>&1
      release "$slot"
    ) &
  done; done
  wait
  local ok; ok=$(grep -l "kept=50 dropped=0" "$L"/pbcol_${SH}_a*_c*.log 2>/dev/null | wc -l)
  blog "A: $ok/36 clean collection calls"
  [ "$ok" -ge 30 ] || bail "collection too lossy ($ok/36)"

  # ---------------------------------------------------------------- mix
  if [ ! -f "$MIX/.done" ]; then
    blog "B: 50/50 uniform-K mix, expert < $EPHI"
    python3 /workspace/build_dyna_mix.py --expert "$EXPERT" \
      --onpolicy $BD/onpol_a*_c*.lance --out "$MIX" \
      --onpolicy-frac 0.5 --expert-ep-hi "$EPHI" > "$L/pbmix_${SH}.log" 2>&1 \
      || { tail -5 "$L/pbmix_${SH}.log" >> "$BDRV"; bail "mix failed"; }
    grep -q BUILD_MIX_DONE "$L/pbmix_${SH}.log" || bail "mix incomplete"
    grep -q "\[split\] expert restricted" "$L/pbmix_${SH}.log" || bail "expert slice NOT restricted"
    touch "$MIX/.done"
  fi
  blog "B: $(grep -h 'dup K=' "$L/pbmix_${SH}.log" | tail -1)"

  # ------------------------------------------------------------ fine-tune
  if [ ! -f "$OUT/weights_epoch_1.pt" ]; then
    blog "C: fine-tune (lr 1e-5, 2 epochs, epoch 1 pre-registered; ~3.5h)"
    local slot; slot=$(acquire)
    INIT_WEIGHTS=$WM/$IW CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 86400 python3 \
      scripts/train/lewm_expert.py data=ogb data.dataset.name="$MIX" \
      "data.dataset.keys_to_load=[pixels,action]" "data.dataset.keys_to_cache=[action]" \
      "data.dataset.keys_to_merge=null" optimizer.lr=1e-5 trainer.max_epochs=2 \
      trainer.devices=1 output_model_name=dyna_pb_${SH} subdir=dyna_pb_${SH} \
      +action_stats_pin=expert wandb.enabled=false > "$L/pbft_${SH}.log" 2>&1
    release "$slot"
    mkdir -p "$OUT"
    cp "/workspace/swm_home/checkpoints/dyna_pb_${SH}/weights_epoch_1.pt" "$OUT/" 2>/dev/null \
      || { tail -15 "$L/pbft_${SH}.log" >> "$BDRV"; bail "no epoch-1 checkpoint"; }
    # ARCH config, not the training config the trainer writes beside the ckpt
    cp "$WM/config.json" "$OUT/config.json" || bail "no arch config"
  fi
  cmp -s "$OUT/config.json" "$WM/config.json" || bail "packaged arch config differs"
  grep -h "lr-probe" "$L/pbft_${SH}.log" | head -2 | sed "s/^/  [$NM] /" >> "$BDRV"
  blog "C: fine-tuned WM ready"

  # -------------------------------------------------------- caches + TD
  if [ ! -f "$F1" ]; then
    blog "D: caching fs1 under the fine-tuned WM"
    local slot; slot=$(acquire)
    [ -f "$F1F" ] || CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 \
      "$TRM/cache_latents.py" --wm "$OUT" --dataset "$EXPERT" --out "$F1F" \
      --state-key privileged_block_0_pos > "$L/pbcache_${SH}.log" 2>&1
    release "$slot"
    [ -f "$F1F" ] || { tail -5 "$L/pbcache_${SH}.log" >> "$BDRV"; bail "cache failed"; }
    python3 /workspace/filter_cache_eprange.py "$F1F" "$F1" --lo 0 --hi "$EPHI" \
      > "$L/pbfilter_${SH}.log" 2>&1
    grep -q FILTER_CACHE_DONE "$L/pbfilter_${SH}.log" || bail "cache filter incomplete"
  fi
  [ -f "$F5" ] || python3 "$TRM/subsample_cache.py" --in "$F1" --out "$F5" --frameskip 5 \
    > "$L/pbfs5_${SH}.log" 2>&1 || bail "fs5 failed"
  if [ ! -f "$TD" ]; then
    local slot; slot=$(acquire)
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 28800 python3 "$P/train_metric.py" \
      --cache "$F1" --learner td --head quasimetric --expectile 0.03 --n-step 50 \
      --steps 6000 --seed 0 --out "$TD" > "$L/pbtd_${SH}.log" 2>&1
    release "$slot"
  fi
  [ -f "$TD" ] || bail "TD teacher missing"
  blog "D: caches + TD ready"

  # ------------------------------------------------------------- actors
  blog "E: POST actors at the shared config, 3 seeds"
  for s in $SEEDS; do
    local out=/workspace/actors/lip4_pbpost_${SH}_s${s}.pt
    [ -f "$out" ] && continue
    local slot; slot=$(acquire)
    ( # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
        --cache "$F5" --cache-td "$F1" --h5 "$AH5" --wm "$OUT" --init-value "$TD" \
        --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AM" \
        --actor-lr 3e-4 --actor-lr-final 3e-5 \
        --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
        --arch v4 --seed "$s" --freeze-critic-frac "$SEL_FRAC" $SEL_FLAGS \
        --out "$out" --out-value "${out%.pt}_value.pt" > "$L/pbpost_${SH}_s${s}.log" 2>&1
      release "$slot"
    ) &
  done
  wait
  for s in $SEEDS; do
    [ -f "/workspace/actors/lip4_pbpost_${SH}_s${s}.pt" ] || bail "POST actor s$s missing"
  done
  blog "E: POST actors done"

  # -------------------------------------------------------------- evals
  blog "F: 9 held-out eval cells"
  for s in $SEEDS; do for d in $DRAWS; do
    local nm="pbpost_${SH}_s${s}_e${d}"
    local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && continue
    local slot; slot=$(acquire)
    (
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) MUJOCO_EGL_DEVICE_ID=$(( slot % NGPU )) \
      timeout 7200 python3 "$P/eval_wm.py" --config-name cube seed=$d \
        eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
        eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
        policy="$OUT" solver=lip \
        "solver.actor_path=/workspace/actors/lip4_pbpost_${SH}_s${s}.pt" \
        output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
      sr=""
      grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
        sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
      flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
      release "$slot"
    ) &
  done; done
  wait
  blog "F: evals done"

  # --------------------------------------------------------------- card
  NM="$NM" SH="$SH" PRETAG="$TAG" AM="$AM" \
  CFG="--freeze-critic-frac $SEL_FRAC ${SEL_FLAGS:-(plain)}" \
  python3 - "$SUM" 2>&1 <<'PY' | tee -a "$BDRV" >> "$DRV"
import os, sys, math
NM, SH, PRETAG = os.environ["NM"], os.environ["SH"], os.environ["PRETAG"]
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
pre, post = arm(PRETAG), arm(f"pbpost_{SH}")
REF = {"LeWM": "banked Dyna on LeWM: +4.7 (t=7.00, p~0.02), replicated 3x, plateau ~92",
       "PLDM": "banked Dyna on PLDM round 1: +8.0 (t=2.11, n=3, not yet significant)"}
print(f"=== DYNA on {NM} (amax {os.environ['AM']}, held-out 8000:10000, EGL) ===")
print(f"  shared critic/data config: {os.environ['CFG']}")
print(f"  PRE = {PRETAG} (the selected cell, already measured)")
if not post:
    print("  no complete POST seeds"); print(f"{NM.upper()}_DYNA_DONE"); sys.exit()
print(f"  {'seed':>4s} {'PRE':>7s} {'POST':>7s} {'delta':>7s}")
for i in range(len(post)):
    p = f"{pre[i]:7.2f}" if i < len(pre) else "      -"
    d = f"{post[i]-pre[i]:+7.2f}" if i < len(pre) else "      -"
    print(f"  {i:4d} {p} {post[i]:7.2f} {d}")
pm = f"{mean(pre):.2f}" if pre else "-"
print(f"  mean  {pm:>7s} {mean(post):7.2f}", end="")
if pre and len(pre) == 3 and len(post) == 3:
    d = [x-y for x, y in zip(post, pre)]; m, s = mean(d), sd(d)
    t = m/(s/math.sqrt(3)) if s > 0 else float("inf")
    print(f" {m:+7.2f}   sd {s:.2f}  t {t:.2f} (df=2, crit 2.92)"
          f"  {'SIGNIFICANT' if t > 2.92 else 'not significant'}")
else:
    print()
print(f"  {REF.get(NM, '')}")
print("  This is a per-base result: its own PRE, its own seeds, nothing pooled")
print("  with the other base. Only the knob VALUES were shared.")
print("  Caveat: 3 seeds resolves ~3 pts; PRE and POST are both 3-seed means.")
print(f"{NM.upper()}_DYNA_DONE")
PY
  touch "$BD/DONE"
  blog "=== $NM Dyna complete ==="
}

log "P3: launching two INDEPENDENT per-base Dyna runs"
( run_base LeWM /workspace/models/v2WM              weights_epoch_22.pt 1.6 "$LTAG" jl ) &
( run_base PLDM /workspace/models/PLDM_OgBench_lewm weights.pt          4.5 "$PTAG" jp ) &
wait

for e in "jl LeWM" "jp PLDM"; do
  set -- $e
  if [ -f "$D/$1/DONE" ]; then log "P4: $2 finished"
  elif [ -f "$D/$1/FAILED" ]; then log "P4: $2 FAILED -- see $L/driver_dyna_$1.log (the other base was unaffected)"
  else log "P4: $2 ended without a marker"; fi
done
log "DYNA_PERBASE_DONE"
