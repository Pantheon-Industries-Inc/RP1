#!/bin/bash
# RESOLVE the PLDM LIP-vs-TD+CEM tie by adding SEEDS, not configs.
#
# The grid winner (mw0.1 / amax4.5 / lr3e-4) sits at 75.0 vs the TD+CEM bar
# 73.0 on draws 43/44: +2.0 with sd 4.58 across 3 seeds => SE 2.64, t 0.76.
# That is a tie only because n=3. More CONFIGS cannot fix it (they raise the
# observable ceiling and the winner's curse with it); more SEEDS shrink the
# error bar on the effect we already have:
#     n= 3  SE 2.64  t 0.76
#     n=16  SE 1.15  t 1.75   (critical 1.753 at df=15 -- exactly borderline)
#     n=20  SE 1.02  t 1.95   (critical 1.729 at df=19 -- decisive)
# Hence 20, not 16: landing on the boundary would waste the run.
#
# Seeds 0-2 exist. This trains 3-19 and evaluates them on all three draws.
# Draw 42 selected the CONFIG (at seed 0 only), so the 43/44 basis stays fully
# clean; the card reports both, with 43/44 as the headline.
#
# SCHEDULING. LIP training never renders, so it is proven safe alongside a
# fine-tune on another GPU (the LeWM composite did exactly this). EVALS are
# different: the standing rule is that egl crashes under concurrent training,
# and while per-GPU pinning retired the parallel-EVAL rule, eval-during-
# TRAINING has NOT been revalidated. So trains run now on GPUs 1-7 beside the
# Dyna fine-tune, and evals wait for the Dyna chain to finish outright. Both
# cards then land together and no untested rule gets broken.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_seedres.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"
TAG=mw01_a45_lr3e-4; AMAX=4.5; MW=0.1; ALR=3e-4; ALRF=3e-5
NEW_SEEDS=$(seq 3 19); TRAIN_GPUS="1 2 3 4 5 6 7"; NGPU=8
mkdir -p "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== SEED RESOLUTION, $TAG, seeds 3-19 (pid $$) ==="
for f in "$QF1" "$QF5" "$QTD" "$AH5" "$PLDM/weights.pt"; do [ -e "$f" ] || die "missing $f"; done

# --------------------------- P1 trains on GPUs 1-7 (GPU 0 = the Dyna fine-tune)
log "P1: training seeds 3-19 on GPUs 1-7 (no rendering; safe beside the fine-tune)"
gi=0; set -- $TRAIN_GPUS
for s in $NEW_SEEDS; do
  out=/workspace/actors/lip4_pldm_${TAG}_s${s}.pt
  [ -f "$out" ] && { log "  s$s present"; continue; }
  eval "gpu=\${$((gi+1))}"
  CUDA_VISIBLE_DEVICES=$gpu timeout 28800 python3 "$P/train_lip_ac.py" \
    --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
    --mean-weight "$MW" --actor-lr "$ALR" --actor-lr-final "$ALRF" \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --arch v4 --seed "$s" --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/seedres_train_s${s}.log" 2>&1 &
  gi=$((gi+1)); [ "$gi" -ge 7 ] && { wait; gi=0; }
done
wait
log "P1: $(ls /workspace/actors/ | grep -cE "lip4_pldm_${TAG}_s[0-9]+\.pt$") actors present"

# ------------------- P2 wait for the Dyna chain, then a quiet box
log "P2: waiting for the Dyna chain (evals must not overlap training)"
T0=$(date +%s)
until grep -q "PLDM_DYNA2_DONE" "$L/driver_pldmdyna2.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && { log "WARN: Dyna chain not done in 12h; proceeding once idle"; break; }
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 60; done
log "P2: box quiet"

# ------------------------------------------------ P3 evals, 8-wide, pinned
ev(){ # gpu seed draw
  local gpu=$1 s=$2 d=$3 nm="grid_${TAG}_s${s}_e${d}"
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$PLDM" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_pldm_${TAG}_s${s}.pt" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
}
log "P3: evals, seeds 3-19 x 3 draws = 51 cells, 8-wide"
g=0
for s in $NEW_SEEDS; do for d in $DRAWS; do
  ev "$g" "$s" "$d" &
  g=$((g+1)); [ "$g" -ge "$NGPU" ] && { wait; g=0; }
done; done
wait
log "P3: evals done"

# ------------------------------------------------------------------ P4 card
TAG="$TAG" python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import os, sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
tag = os.environ["TAG"]
mean = lambda xs: sum(xs)/len(xs)
def sm(s, draws):
    vs = [rows.get(f"grid_{tag}_s{s}_e{d}") for d in draws]
    return None if any(v is None for v in vs) else mean(vs)
def ttest(xs, bar):
    n = len(xs); m = mean(xs)
    sd = math.sqrt(sum((x-m)**2 for x in xs)/(n-1))
    se = sd/math.sqrt(n)
    return m, sd, ((m-bar)/se if se > 0 else float('inf')), n
# one-sided 5% critical values by df
CRIT = {2:2.920, 5:2.015, 9:1.833, 14:1.761, 15:1.753, 19:1.729, 29:1.699}
def crit(df):
    ks = sorted(CRIT); return CRIT[min(ks, key=lambda k: abs(k-df))]
print(f"=== PLDM SEED-RESOLUTION CARD ({tag}, frozen PLDM, held-out, EGL) ===")
td3 = [rows.get(f"ref_cemtd_e{d}") for d in (42,43,44)]
for label, draws in (("draws 43+44 (clean: draw 42 selected the config)", (43,44)),
                     ("all 3 draws", (42,43,44))):
    per = [(s, sm(s, draws)) for s in range(20)]
    have = [(s, v) for s, v in per if v is not None]
    if len(have) < 3: continue
    vals = [v for _, v in have]
    bar = mean([td3[1], td3[2]]) if draws == (43,44) else mean(td3)
    m, sd, t, n = ttest(vals, bar)
    c = crit(n-1)
    print(f"  --- {label} ---")
    print(f"    seeds n={n}: mean {m:.2f}  sd {sd:.2f}  se {sd/math.sqrt(n):.2f}")
    print(f"    vs TD+CEM {bar:.1f}: {m-bar:+.2f}, t={t:.2f} (df={n-1}, crit {c:.3f}) "
          f"-> {'SIGNIFICANT' if t > c else 'not significant'}")
    print(f"    per-seed: {' '.join(f'{v:.0f}' for _, v in have)}")
print("  note: the bar is treated as a fixed constant; TD+CEM's own draw noise")
print("  is not propagated, so the test is very slightly anti-conservative.")
print("PLDM_SEEDRES_DONE")
PY
log "done"
