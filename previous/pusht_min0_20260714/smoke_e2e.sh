#!/bin/bash
# End-to-end smoke of every new code path on PushT (GPU 3, ~15 min):
#   1. velimag mini-train (120 steps, 2xD metric)
#   2. CEM eval with the 2xD metric (eval-hook velocity path), 4 episodes
#   3. tandem min0 mini-train warm-started from it (train_lip_ac vel path)
#   4. LIP eval of the mini actor (LIPSolver velocity path), 4 episodes
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1 MUJOCO_GL=osmesa
CODE=/workspace/code/stable-worldmodel
H5=/workspace/swm_home/datasets/pusht_expert_train.h5
C1=/workspace/caches/pusht_official_fs1.pt
C5=/workspace/caches/pusht_official_fs5.pt
G=${SMOKE_GPU:-3}
S=/workspace/min0/smoke
mkdir -p $S

echo "[smoke] 1/4 velimag mini-train"
CUDA_VISIBLE_DEVICES=$G python3 /workspace/min0/train_metric_velimag.py \
  --cache $C1 --h5 $H5 --wm lewm_pusht_official --expectile 0.03 --n-step 3 \
  --steps 120 --out $S/vi_smoke.pt || exit 1

echo "[smoke] 2/4 CEM eval with 2xD metric (4 eps)"
CUDA_VISIBLE_DEVICES=$G python3 $CODE/scripts/plan/eval_wm.py --config-name pusht_lewm \
  seed=42 eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=4 \
  solver.batch_size=4 "+metric=$S/vi_smoke.pt" output.filename=smoke_cem.txt \
  > $S/smoke_cem.log 2>&1
grep -q "success_rate" $S/smoke_cem.log || { tail -25 $S/smoke_cem.log; exit 1; }
grep -E "metric-hook|success_rate" $S/smoke_cem.log

echo "[smoke] 2b/4 win3 mini-train + CEM eval (4 eps)"
CUDA_VISIBLE_DEVICES=$G python3 /workspace/min0/train_metric_velimag.py \
  --cache $C1 --h5 $H5 --wm lewm_pusht_official --expectile 0.03 --n-step 3 \
  --win-frames 3 --steps 120 --out $S/wn3_smoke.pt || exit 1
CUDA_VISIBLE_DEVICES=$G python3 $CODE/scripts/plan/eval_wm.py --config-name pusht_lewm \
  seed=42 eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=4 \
  solver.batch_size=4 "+metric=$S/wn3_smoke.pt" output.filename=smoke_cem_w3.txt \
  > $S/smoke_cem_w3.log 2>&1
grep -q "success_rate" $S/smoke_cem_w3.log || { tail -25 $S/smoke_cem_w3.log; exit 1; }

echo "[smoke] 3/4 tandem min0 mini-train (40 steps)"
CUDA_VISIBLE_DEVICES=$G python3 $CODE/scripts/plan/train_lip_ac.py \
  --cache $C5 --cache-td $C1 --h5 $H5 --wm lewm_pusht_official \
  --init-value $S/vi_smoke.pt --horizon 5 --iters 8 --steps 40 --n-step 3 \
  --expectile 0.1 --expectile-final 0.03 --p-imag 0.5 --drop-z0 --drop-zg \
  --amax 3.5 --seed 0 --out $S/m0_smoke.pt --out-value $S/m0_smoke_v.pt || exit 1

echo "[smoke] 4/4 LIP eval of mini actor (4 eps, R2)"
CUDA_VISIBLE_DEVICES=$G python3 $CODE/scripts/plan/eval_wm.py --config-name pusht_lewm \
  seed=42 eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=4 \
  solver.batch_size=4 solver=lip "solver.actor_path=$S/m0_smoke.pt" \
  solver.restarts=2 solver.restart_noise=0.5 output.filename=smoke_lip.txt \
  > $S/smoke_lip.log 2>&1
grep -q "success_rate" $S/smoke_lip.log || { tail -25 $S/smoke_lip.log; exit 1; }
grep -q "velocity metric detected" $S/smoke_lip.log || { echo "MISSING velocity detection"; exit 1; }
grep -E "velocity metric|success_rate" $S/smoke_lip.log

echo "[smoke] ALL 4 PASSED"
