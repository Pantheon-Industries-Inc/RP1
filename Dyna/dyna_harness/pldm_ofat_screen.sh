#!/bin/bash
# BROAD OFAT SCREEN over the LIP design space on PLDM -- 24 arms, one factor
# at a time, ranked on a COMMON static teacher.
#
# THE SPACE. train_lip_ac.py exposes ~35 knobs; this campaign has swept six
# (amax, horizon, mean-weight, actor-lr, iters, steps). Untouched: the entire
# actor architecture (arch/iter-mode/head-mode/width/layers/s-dim/zero-init/
# gd-init/feat-norm/s0-mode/feed), the sampling knobs (p-cross, max-delta,
# batch), and -- most interesting -- two auxiliary objective terms that are
# switched OFF by default:
#   --expand-weight  feeds the actor's OWN visited states (z0,zT,zg) into the
#                    critic update. On-policy critic training, aimed exactly at
#                    the mismatch where the critic is fit on expert latents but
#                    queried at actor-imagined states.
#   --replay-prob    replays actor-visited states.
# Both are Dyna-like corrections INSIDE the LIP loop, never switched on.
#
# WHY A COMMON FROZEN TEACHER. If every arm co-trains its own critic, E_final
# is measured against a different teacher per arm and cross-arm comparison is
# meaningless. Freezing at 5% in EVERY arm pins all of them to the same
# pretrained TD teacher, so E_final becomes a shared yardstick -- and the
# screen needs no eval cells to rank.
#
# WHY E_final AS THE SCREEN, WITH A CAVEAT. It is a batch-averaged training
# quantity, far lower variance than a 150-task success rate -- and this
# campaign has already been burned once by a single lucky seed inventing a
# +2.0. But E_final is NOT the objective: on LeWM, amax3.5+iters16 reached the
# lowest E_final ever measured (1.68) and planned WORSE (80.0 vs 87.8). So
# this screen is a FILTER, not a selector: it identifies knobs that move
# anything at all. Movers then get success evals with seeds, where the
# direction (better plans vs more exploitation) is actually decided.
#
# NOTE the critic-side knobs (expectile, critic-lr, ema-tau, huber-beta,
# critic-ratio, pretrain, n-step) are deliberately NOT here: with a frozen
# common teacher they would be no-ops. They need their own screen, and the
# cheap way to run it is TD+CEM success (train the teacher, evaluate it with
# search, no actor training at all).
#
# 23 arms, 3 waves of 8, ~2.5h.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; DRV=$L/driver_ofat.log
PLDM=/workspace/models/PLDM_OgBench_lewm
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
AMAX=4.5; NGPU=8
mkdir -p "$L" "$R"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
cd "$CODE"

# ---- arm table: tag|extra args.  Baseline = the current best config.
ARMS=(
  "base|"
  # --- auxiliary objective terms, both OFF by default (highest prior) ---
  "expand01|--expand-weight 0.1"
  "expand10|--expand-weight 1.0"
  "replay25|--replay-prob 0.25"
  "replay50|--replay-prob 0.5"
  # --- objective shape: mean-weight DOWN is untested (0.3/1.0 both hurt) ---
  "mw003|--mean-weight 0.03"
  "mw000|--mean-weight 0.0"
  # --- INSIDE v4 only. --arch stays v4 by user directive: v4 was settled by
  # the lipv2->v4 result and is not up for re-litigation; these vary how v4 is
  # configured, not which architecture it is. ---
  "itemb|--iter-mode emb"
  "headprec|--head-mode precond"
  "wide|--width 512"
  "deep|--layers 4"
  "sdim512|--s-dim 512"
  "rec1024|--rec-hidden 1024"
  "zeroinit|--zero-init"
  "gdinit|--gd-init 0.1"
  "featnorm|--feat-norm"
  "s0z0|--s0-mode z0"
  "feedend|--feed end"
  "feedtraj|--feed traj"
  # --- sampling ---
  "pcross0|--p-cross 0.0"
  "pcross6|--p-cross 0.6"
  "maxd20|--max-delta 20"
  "batch256|--batch 256"
)

log "=== PLDM OFAT SCREEN, ${#ARMS[@]} arms, common frozen teacher (pid $$) ==="
for f in "$QF1" "$QF5" "$QTD" "$AH5" "$PLDM/weights.pt"; do [ -e "$f" ] || die "missing $f"; done
while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do
  log "  waiting for the box to drain"; sleep 120; done
log "P0: box quiet"

g=0
for entry in "${ARMS[@]}"; do
  tag="${entry%%|*}"; extra="${entry#*|}"
  out=/workspace/actors/lip4_pldm_of_${tag}_s0.pt
  [ -f "$out" ] && { log "  $tag present"; continue; }
  # every arm: critic frozen at 5% => identical static teacher => comparable E
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$g timeout 43200 python3 "$P/train_lip_ac.py" \
    --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
    --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
    --actor-lr 3e-4 --actor-lr-final 3e-5 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --arch v4 --seed 0 --freeze-critic-frac 0.05 $extra \
    --out "$out" --out-value "${out%.pt}_value.pt" \
    > "$L/ofat_${tag}_s0.log" 2>&1 &
  g=$((g+1))
  if [ "$g" -ge "$NGPU" ]; then wait; g=0; log "  wave done"; fi
done
wait
log "P1: all arms trained"

python3 - "$L" 2>&1 <<'PY' | tee -a "$DRV"
import sys, re, glob, os
L = sys.argv[1]
def last(path, key):
    try: t = open(path).read()
    except OSError: return None
    m = re.findall(rf"{key} ([0-9.]+)", t)
    return float(m[-1]) if m else None
rows = []
for f in sorted(glob.glob(f"{L}/ofat_*_s0.log")):
    tag = os.path.basename(f)[5:-7]
    ef, e1 = last(f, "E_final"), last(f, "E_first")
    if ef is not None: rows.append((tag, ef, e1))
base = next((e for t, e, _ in rows if t == "base"), None)
rows.sort(key=lambda r: r[1])
print("=== PLDM OFAT SCREEN (common static teacher, seed 0) ===")
print(f"  {'arm':10s} {'E_final':>8s} {'E_first':>8s} {'vs base':>9s}")
for t, ef, e1 in rows:
    d = f"{100*(ef-base)/base:+.1f}%" if base else "?"
    star = "  <<<" if base and abs(ef-base) > 0.05*base else ""
    print(f"  {t:10s} {ef:8.2f} {(e1 or 0):8.2f} {d:>9s}{star}")
movers = [t for t, ef, _ in rows if base and abs(ef-base) > 0.05*base]
print(f"  movers (>5% change in E_final): {' '.join(movers) if movers else 'NONE'}")
print("  NOTE E_final is a FILTER, not a selector -- lower is not automatically")
print("  better (LeWM's lowest-ever E_final planned WORSE). Movers now need")
print("  success evals with SEEDS to decide direction. If NOTHING moves, the")
print("  actor-side design space is not the lever on PLDM and the constraint")
print("  lies in the world model or the critic.")
print("PLDM_OFAT_DONE")
PY
log "done"
