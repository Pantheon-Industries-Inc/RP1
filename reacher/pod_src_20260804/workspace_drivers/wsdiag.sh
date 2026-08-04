export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
one(){  # gpu tag warm_start rh
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$1 CUDA_VISIBLE_DEVICES=$1 timeout 10800 python3     swm_cem/scripts/plan/eval_wm.py --config-name reacher     policy=/workspace/swm_home/checkpoints/lejepa_reacher     eval.dataset_name=/workspace/datasets_canon/lewm-reacher/reacher.h5     dataset.stats=/workspace/datasets_canon/lewm-reacher/reacher.h5     +eval.ep_range=8000:10000 seed=42 eval.goal_offset_steps=25 eval.eval_budget=50     plan_config.receding_horizon=$4 +plan_config.deadline=50 plan_config.warm_start=$3     solver=cem solver.n_steps=10 solver.batch_size=10     "+metric=/workspace/metrics/l2window3.pt" output.filename=ws_$2.txt > logs/ws_$2.log 2>&1
  h=$(grep -oE "HELD-at-end [0-9.]+" logs/ws_$2.log | tail -1 | grep -oE "[0-9.]+")
  t=$(grep -oE "0\.1rad [0-9.]+" logs/ws_$2.log | tail -1 | sed -E "s/0\.1rad //")
  echo "RESULT $2 (rh=$4 warm_start=$3): @0.05 ${h:-FAIL} | @0.1 ${t:-FAIL}"
}
one 3 rh1_ws_on  true  1 &
one 4 rh1_ws_off false 1 &
one 5 rh5_ws_on  true  5 &
wait
echo WSDIAG_DONE
