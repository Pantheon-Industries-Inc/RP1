export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
one(){  # gpu tag rh deadline
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$1 CUDA_VISIBLE_DEVICES=$1 timeout 5400 python3     swm_cem/scripts/plan/eval_wm.py --config-name reacher     policy=/workspace/swm_home/checkpoints/lejepa_reacher     eval.dataset_name=/workspace/datasets_canon/lewm-reacher/reacher.h5     dataset.stats=/workspace/datasets_canon/lewm-reacher/reacher.h5     +eval.ep_range=8000:10000 seed=42 eval.goal_offset_steps=25 eval.eval_budget=50     plan_config.receding_horizon=$3 +plan_config.deadline=$4     solver=lip solver.actor_path=/workspace/actors/lip4_leak6_lejepa_s0.pt     solver.rollout_compat=false solver.batch_size=10 output.filename=sm_$2.txt     > logs/sm_$2.log 2>&1
  h=$(grep -oE "HELD-at-end [0-9.]+" logs/sm_$2.log | tail -1 | grep -oE "[0-9.]+")
  t=$(grep -oE "0\.1rad [0-9.]+" logs/sm_$2.log | tail -1 | sed -E "s/0\.1rad //")
  d=$(grep -oE "\[deadline\].*" logs/sm_$2.log | head -1)
  echo "RESULT $2 (rh=$3 deadline=$4): @0.05 ${h:-FAIL} | @0.1 ${t:-FAIL}   $d"
}
one 0 rh1_align 1 50 &
one 1 rh1_noalign 1 0 &
one 2 rh5_align 5 50 &
wait
echo SMOKE3_DONE
