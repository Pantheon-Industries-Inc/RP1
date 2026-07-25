#!/bin/bash
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export OMP_NUM_THREADS=16
PLAN=/workspace/code/stable-worldmodel/scripts/plan
CACHE1=/workspace/caches/cube_v2wm_fs1.pt
HWM=/workspace/hwm/hwm_s25m8.pt
TD=/workspace/metrics/cf_dE_t003n50.pt
ACTNPZ=/workspace/datasets/actions_cube.npz
mkdir -p /workspace/logs /workspace/actors /workspace/metrics /workspace/hwm
train_one(){ # gpu seed
  local g=$1 s=$2
  local out=/workspace/actors/lip_hl_s25m8_s${s}.pt
  [ -f "$out" ] && { echo "skip hl s$s"; rmdir /workspace/.gpu$g 2>/dev/null; return; }
  CUDA_VISIBLE_DEVICES=$g python3 "$PLAN/train_lip_hl.py" \
    --cache "$CACHE1" --hwm "$HWM" --init-value "$TD" \
    --horizon 8 --iters 8 --steps 6000 --n-step 50 --amax 3.0 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --seed "$s" \
    --out "$out" --out-value "/workspace/metrics/lip_hl_s25m8_s${s}_value.pt" \
    > "/workspace/logs/train_lip_hl_s${s}.log" 2>&1
  rmdir /workspace/.gpu$g 2>/dev/null
}
booster(){
  local g=3
  local out=/workspace/hwm/hwm_s25m8d10.pt
  [ -f "$out" ] && { rmdir /workspace/.gpu$g 2>/dev/null; return; }
  CUDA_VISIBLE_DEVICES=$g python3 /workspace/scripts/train_hwm.py \
    --cache "$CACHE1" --actions "$ACTNPZ" --out "$out" \
    --stride 25 --macro-dim 8 --ae mlp --loss mse --pred-horizon 8 \
    --depth 10 --steps 30000 --fig6-wm /workspace/ckpts/ogbench_cube_single_v2WM \
    > /workspace/logs/hwm_s25m8d10.log 2>&1
  rmdir /workspace/.gpu$g 2>/dev/null
}
train_one 0 0 & train_one 1 1 & train_one 2 2 & booster &
wait
touch /workspace/actors/hl_s25m8.done
echo HL_TRAININGS_DONE

===SWEEPREF
#!/bin/bash
# HWM high-level sweep: 6 configs x 6 GPUs. Cheap (latents cached).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export OMP_NUM_THREADS=16
CACHE=/workspace/caches/cube_v2wm_fs1.pt
ACT=/workspace/datasets/actions_cube.npz
WM=/workspace/ckpts/ogbench_cube_single_v2WM
OUT=/workspace/hwm
LOGS=/workspace/logs
mkdir -p "$OUT" "$LOGS"
run(){ # gpu name extra...
  local gpu=$1 name=$2; shift 2
  [ -f "$OUT/hwm_${name}.pt" ] && { echo "skip ${name}"; return; }
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 /workspace/scripts/train_hwm.py \
    --cache "$CACHE" --actions "$ACT" --out "$OUT/hwm_${name}.pt" \
    --fig6-wm "$WM" "$@" > "$LOGS/hwm_${name}.log" 2>&1 &
  echo "launched ${name} on gpu${gpu} (pid $!)"
}
run 0 s25m8      --stride 25 --macro-dim 8 --ae mlp --loss mse --pred-horizon 6
run 1 s25m8ae    --stride 25 --macro-dim 8 --ae tf  --loss mse --pred-horizon 6
run 2 s25m4      --stride 25 --macro-dim 4 --ae mlp --loss mse --pred-horizon 6
run 3 s15m8      --stride 15 --macro-dim 8 --ae mlp --loss mse --pred-horizon 8
run 4 s50m8      --stride 50 --macro-dim 8 --ae mlp --loss mse --pred-horizon 3
run 5 s25m8l1    --stride 25 --macro-dim 8 --ae mlp --loss l1  --pred-horizon 6
wait
echo HWM_SWEEP_DONE
