#!/bin/bash
# LIPv4(min0) + AC tandem on the PLDM OGBench-cube checkpoint, vs PLDM+CEM baseline.
# Fresh pod 4xH100. Phases:
#   0 wait dataset/env; verify h5; GPU smoke-load of PLDM (LeWM-wrapped)
#   1 build PLDM latent caches fs1/fs5 (4-GPU shards)
#   2 TD warm-starts (t003n50, t01n50) || harness anchors (champion+v2WM s43 lip: expect 96;
#     v2WM TD+CEM s42: expect 82)
#   3 round-1 tandem sweep: 8 min0 no-gate arms (schedamax center)
#   4 evals: 16 selection (s42+s44 h25 per arm) + 6 PLDM+CEM baseline + 2 TD+CEM diagnostics
# Idempotent: cached rows in results/summary.csv, existing .pt skipped.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa
export TQDM_DISABLE=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
WM_PLDM=/workspace/ckpts/PLDM_OgBench_lewm
WM_V2=/workspace/ckpts/ogbench_cube_single_v2WM
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5
CACHE1=/workspace/caches/cube_pldm_fs1.pt
CACHE5=/workspace/caches/cube_pldm_fs5.pt
CHAMP=$ACT/lip_ac90_schedamax.pt
TD_V2=$MET/cf_dE_t003n50.pt
EVAL_TIMEOUT=14400
mkdir -p "$LOGS" "$RES" "$MET" "$ACT" /workspace/caches
touch "$RES/summary.csv"

log() { echo "[$(date +%H:%M:%S)] PLDM-PIPE: $*" | tee -a "$LOGS/pipeline_pldm.log"; }
die() { log "FATAL: $*"; exit 1; }

# ---------- tiny GPU pool: run_pool JOBS... where each job is "name:cmd" -------
GPUQ=/tmp/gpuq.$$
mkfifo "$GPUQ"
exec 9<>"$GPUQ"
rm -f "$GPUQ"
for g in 0 1 2 3; do echo "$g" >&9; done
run_pool() { # each arg: "name|cmd"  (cmd sees $GPU)
  local pids=()
  for job in "$@"; do
    local name="${job%%|*}" cmd="${job#*|}"
    read -r -u 9 GPU
    (
      export CUDA_VISIBLE_DEVICES=$GPU
      log "start [$name] on gpu$GPU"
      eval "$cmd" > "$LOGS/${name}.log" 2>&1
      rc=$?
      [ $rc -eq 0 ] && log "done  [$name]" || log "FAIL  [$name] rc=$rc (see $LOGS/${name}.log)"
      echo "$GPU" >&9
    ) &
    pids+=($!)
  done
  wait "${pids[@]}"
}

# eval job builder: writes success_rate row into summary.csv
do_eval() { # name seed offset budget extra...
  local name=$1 seed=$2 offset=$3 budget=$4; shift 4
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached ($(grep "^${name}," "$RES/summary.csv" | tail -1 | cut -d, -f2))"; return 0
  fi
  timeout "$EVAL_TIMEOUT" $PY "$PLAN/eval_wm.py" --config-name cube \
    seed="$seed" eval.dataset_name="$H5" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    output.filename="${name}.txt" "$@" > "$LOGS/ev_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/ev_${name}.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "${sr:-}" ]; then echo "${name},FAIL" >> "$RES/summary.csv"; return 1; fi
  echo "${name},${sr}" >> "$RES/summary.csv"
  log "eval ${name}: ${sr}"
}

# ---------------- phase 0: wait + verify ----------------
log "phase 0: waiting for dataset + env"
for i in $(seq 1 720); do
  [ -f /workspace/datasets/lewm_cube_full/extract.done ] && [ -f /workspace/env.done ] && break
  sleep 20
done
[ -f /workspace/datasets/lewm_cube_full/extract.done ] || die "dataset never finished"
[ -f /workspace/env.done ] || die "env never finished"
log "dataset+env ready: $(du -sh $H5 | cut -f1)"

$PY - <<'PYEOF' >> "$LOGS/pipeline_pldm.log" 2>&1 || die "h5 verification failed"
import h5py, hdf5plugin, numpy as np
p = "/workspace/datasets/lewm_cube_full/cube_single_expert.h5"
with h5py.File(p, "r") as h:
    need = {"pixels", "action", "ep_len", "ep_offset", "qpos", "qvel"}
    assert not (need - set(h.keys())), need - set(h.keys())
    ln = h["ep_len"][:]
    assert ln.sum() == 2010000 and (ln == 201).all()
    px = h["pixels"]
    assert px.dtype == np.uint8 and px.shape[1:] == (224, 224, 3)
    print("first/last frame readable:", float(px[0].mean()), float(px[-1].mean()))
print("H5 OK")
PYEOF
log "h5 verified"

$PY - <<'PYEOF' >> "$LOGS/pipeline_pldm.log" 2>&1 || die "PLDM smoke failed"
import torch, stable_worldmodel as swm
import h5py, hdf5plugin, numpy as np
m = swm.wm.utils.load_pretrained("/workspace/ckpts/PLDM_OgBench_lewm").cuda().eval()
with h5py.File("/workspace/datasets/lewm_cube_full/cube_single_expert.h5","r") as h:
    fr = torch.from_numpy(h["pixels"][:3].astype(np.float32)/255.).permute(0,3,1,2).cuda()
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda(); std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
fr = (fr-mean)/std
with torch.no_grad():
    z = m.encode({"pixels": fr.unsqueeze(1)})["emb"]
    assert torch.isfinite(z).all(), "NaN in embeddings"
    px = fr[:3].unsqueeze(0).unsqueeze(0).expand(1,2,3,3,224,224)
    acts = torch.zeros(1,2,8,25).cuda()
    out = m.rollout({"pixels": px}, acts)["predicted_emb"]
    assert torch.isfinite(out).all(), "NaN in rollout"
print("PLDM smoke OK", z.shape, out.shape, float(z.std()))
PYEOF
log "PLDM smoke OK"

# ---------------- phase 1: caches ----------------
if [ ! -f /workspace/caches/pldm_caches.done ]; then
  log "phase 1: building PLDM fs1 cache (4 shards)"
  pids=()
  for i in 0 1 2 3; do
    s=$((i * 2500)); e=$(((i + 1) * 2500))
    CUDA_VISIBLE_DEVICES=$i $PY /workspace/valscripts/cache_cube_full.py \
      --h5 "$H5" --wm "$WM_PLDM" --stride 1 --ep-start "$s" --ep-end "$e" \
      --out /workspace/caches/pldm_fs1_shard$i.pt > "$LOGS/pldm_cache_shard$i.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  for i in 0 1 2 3; do
    [ -f /workspace/caches/pldm_fs1_shard$i.pt ] || die "cache shard $i missing"
  done
  $PY /workspace/valscripts/merge_caches.py "$CACHE1" "$CACHE5" \
    /workspace/caches/pldm_fs1_shard0.pt /workspace/caches/pldm_fs1_shard1.pt \
    /workspace/caches/pldm_fs1_shard2.pt /workspace/caches/pldm_fs1_shard3.pt \
    > "$LOGS/pldm_cache_merge.log" 2>&1 || die "cache merge failed"
  rm -f /workspace/caches/pldm_fs1_shard*.pt
  echo done > /workspace/caches/pldm_caches.done
fi
$PY - <<'PYEOF' >> "$LOGS/pipeline_pldm.log" 2>&1 || die "cache verification failed"
from stable_worldmodel.trm import LatentCache
c1 = LatentCache.load("/workspace/caches/cube_pldm_fs1.pt")
c5 = LatentCache.load("/workspace/caches/cube_pldm_fs5.pt")
assert len(c1.z) == 2010000 and len(c5.z) == 10000 * 41
assert c1.latent_dim == c5.latent_dim == 192
print("PLDM caches OK", c1.z.shape, c5.z.shape)
PYEOF
log "caches verified"

# ---------------- phase 2: TD warm-starts || anchors ----------------
log "phase 2: TD warm-starts + harness anchors"
J_TD1="td003|[ -f $MET/cf_pldm_t003n50.pt ] || $PY $PLAN/train_metric.py --cache $CACHE1 --learner td --expectile 0.03 --n-step 50 --steps 6000 --out $MET/cf_pldm_t003n50.pt"
J_TD2="td01|[ -f $MET/cf_pldm_t01n50.pt ] || $PY $PLAN/train_metric.py --cache $CACHE1 --learner td --expectile 0.1 --n-step 50 --steps 6000 --out $MET/cf_pldm_t01n50.pt"
J_AN1="anchor_champ_s43|do_eval anchor_champ_h25_s43 43 25 50 policy=$WM_V2 solver=lip solver.actor_path=$CHAMP"
J_AN2="anchor_tdcem_s42|do_eval anchor_tdcem_h25_s42 42 25 50 policy=$WM_V2 solver=cem +metric=$TD_V2"
run_pool "$J_TD1" "$J_TD2" "$J_AN1" "$J_AN2"
[ -f "$MET/cf_pldm_t003n50.pt" ] || die "TD warm-start missing"
a=$(grep "^anchor_champ_h25_s43," "$RES/summary.csv" | tail -1 | cut -d, -f2)
b=$(grep "^anchor_tdcem_h25_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
log "ANCHORS: champion s43 = ${a:-none} (expect 96), v2 TD+CEM s42 = ${b:-none} (expect 82)"

# ---------------- phase 3: round-1 tandem sweep (8 arms, min0 no-gate) ----------------
train_arm() { # name extra-args...
  local name=$1; shift
  [ -f "$ACT/pldm_${name}.pt" ] && { log "train ${name}: cached"; return 0; }
  $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm "$WM_PLDM" \
    --init-value "$MET/cf_pldm_t003n50.pt" \
    --horizon 5 --iters 8 --steps 8000 --n-step 50 --batch 128 \
    --drop-z0 --drop-zg --no-gate \
    --out "$ACT/pldm_${name}.pt" --out-value "$MET/pldm_${name}_value.pt" "$@"
}
export -f train_arm 2>/dev/null || true

SMX="--expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 3e-4 --actor-lr-final 3e-5"
log "phase 3: round-1 training (8 arms)"
run_pool \
  "tr_smx_s0|train_arm smx_s0 $SMX --amax 3.5 --seed 0" \
  "tr_smx_s1|train_arm smx_s1 $SMX --amax 3.5 --seed 1" \
  "tr_amax25_s0|train_arm amax25_s0 $SMX --amax 2.5 --seed 0" \
  "tr_mw03_s0|train_arm mw03_s0 $SMX --amax 3.5 --mean-weight 0.3 --seed 0" \
  "tr_flat_s0|train_arm flat_s0 --expectile 0.03 --critic-lr 1e-3 --actor-lr 3e-4 --amax 2.5 --seed 0" \
  "tr_alr1e4_s0|train_arm alr1e4_s0 --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 1e-4 --actor-lr-final 1e-5 --amax 3.5 --seed 0" \
  "tr_k12_s0|train_arm k12_s0 $SMX --amax 3.5 --iters 12 --seed 0" \
  "tr_t02_s0|train_arm t02_s0 --expectile 0.2 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 3e-4 --actor-lr-final 3e-5 --amax 3.5 --seed 0"
log "round-1 training complete"

# ---------------- phase 4: evals ----------------
log "phase 4: selection + baselines + diagnostics"
EVJOBS=()
for arm in smx_s0 smx_s1 amax25_s0 mw03_s0 flat_s0 alr1e4_s0 k12_s0 t02_s0; do
  [ -f "$ACT/pldm_${arm}.pt" ] || { log "skip evals for missing arm $arm"; continue; }
  for seed in 42 44; do
    EVJOBS+=("ev_lip_${arm}_s${seed}|do_eval lip_${arm}_h25_s${seed} $seed 25 50 policy=$WM_PLDM solver=lip solver.actor_path=$ACT/pldm_${arm}.pt")
  done
done
EVJOBS+=("ev_pldmcem_h25_s42|do_eval pldmcem_h25_s42 42 25 50 policy=$WM_PLDM solver=cem")
EVJOBS+=("ev_pldmcem_h25_s43|do_eval pldmcem_h25_s43 43 25 50 policy=$WM_PLDM solver=cem")
EVJOBS+=("ev_pldmcem_h25_s44|do_eval pldmcem_h25_s44 44 25 50 policy=$WM_PLDM solver=cem")
EVJOBS+=("ev_pldmcem_h50_s42|do_eval pldmcem_h50_s42 42 50 100 policy=$WM_PLDM solver=cem")
EVJOBS+=("ev_pldmcem_h50_s43|do_eval pldmcem_h50_s43 43 50 100 policy=$WM_PLDM solver=cem")
EVJOBS+=("ev_pldmcem_h50_s44|do_eval pldmcem_h50_s44 44 50 100 policy=$WM_PLDM solver=cem")
EVJOBS+=("ev_tdcem_t003_s42|do_eval tdcem_pldm_t003_h25_s42 42 25 50 policy=$WM_PLDM solver=cem +metric=$MET/cf_pldm_t003n50.pt")
EVJOBS+=("ev_tdcem_t01_s42|do_eval tdcem_pldm_t01_h25_s42 42 25 50 policy=$WM_PLDM solver=cem +metric=$MET/cf_pldm_t01n50.pt")
run_pool "${EVJOBS[@]}"

log "ROUND 1 COMPLETE. summary:"
sort "$RES/summary.csv" | tee -a "$LOGS/pipeline_pldm.log"
