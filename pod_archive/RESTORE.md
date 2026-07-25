# Pod restore runbook (archive: full_archive_20260704.tgz)

Restores the complete value-metric / learned-planner program on any fresh GPU pod.
Everything here was verified working on 6x H100, Ubuntu 22.04, Python 3.11, CUDA 12.x.

## What the archive contains
- `metrics/` — ALL trained artifacts: TD values (tuned per env), IREM/AC values+actors,
  TMD, LIP planners (lip_sgl=it8 best, lip_dbl_it4=double-best), shooters, PPO-LIP partials
- `valscripts/` — every trainer/solver/patcher (cache builders, train_metric, irem_value,
  actor_critic_value, learned_planner, learned_shooter, lip_ppo, tmd, mppi_actor solver,
  all hook patchers, online_td_continue, stride5, augmenters, gen_multicube_cfgs)
- patched snapshot files: `snapshot2/scripts/plan/eval_wm.py` (metric+collect hooks),
  `eval_wm_episodic.py`, `world.py` (capture), `solver/mppi_actor.py` (MPPI/LIP/shoot/restarts),
  configs `cube{,_double,_triple}.yaml`, `solver/mppi_actor.yaml`
- queue system: `gpu_worker.sh`, `tasks.txt`, `taskdone_list.txt`, all runner scripts
- `final_manifest.txt` — all 170 results; every `*.log`

## Restore steps (~50 min total)
1. **Container deps**: `apt-get install -y libegl1 libgl1 libglvnd0 libgles2 libopengl0 libosmesa6 rsync`
2. **Code tree**: clone github.com/galilai-group/stable-worldmodel @ `ee0c5f4` to /workspace/snapshot2;
   `patch -p1 < snap_ab.patch` (patch is in this project: /tmp/snap_ab.patch or regenerate from
   checkpoints bundle code dir); then `pip install -e "/workspace/snapshot2[all]"`
3. **Unpack archive** over /workspace (restores valscripts/metrics/patched files/queue).
   Copy the patched snapshot files over the fresh clone (eval_wm.py, world.py, mppi_actor.py, configs).
4. **Checkpoints**: scp from Mac bundle `stable-worldmodel/checkpoints/ogbench_cube_full_handoff_*/models/`
   to /workspace/ckpts/<name>/ (single+double lewm minimum; dino optional)
5. **Env patch**: `MUJOCO_GL=osmesa python /workspace/fix_env_proprio.py` (in valscripts)
6. **Datasets**: `python -c "import ogbench; ogbench.download_datasets(['visual-cube-single-play-v0','visual-cube-double-play-v0'], dataset_dir='/workspace/datasets/ogbench_raw')"`;
   convert with `valscripts` converter + `augment_multicube.py` (both splits; delete `observation` col —
   the augmenter does it). Eval sets = VAL split -> datasets/ogbench_eval_h5/ogbench_cube_*_planning_eval.h5;
   train sets -> datasets/ogbench_swm/visual_cube_*_play_v0_FIXED.h5
7. **Caches** (only needed for retraining): `cache_cube_h5.py` (GPU-batched) + `stride5.py` per env (~5 min each)
8. **Evals**: `cd /workspace/snapshot2 && PYTHONPATH=... MUJOCO_GL=osmesa python scripts/plan/eval_wm.py
   --config-name cube policy=/workspace/ckpts/... solver=cem|mppi_actor seed=42
   eval.dataset_name=<planning_eval.h5> "dataset.keys_to_cache=[...]" +metric=<pt> +cost_mode=replacement
   [+solver.actor_path=<lip.pt> solver.n_steps=0]` — see tasks.txt for exact working commands of every cell.
9. **Queue**: regenerate/extend tasks.txt, `rm -rf claims taskdone`, fire `gpu_worker.sh $i` per GPU (setsid).

## Key results at archive time (see final_manifest.txt + memory file trm-project-state.md)
- LIP (learned planner, frozen value) 3-seed: single 79.3 (BEST; TD+CEM 73.3, latent 64.0),
  double it4 75.3 (ties latent 75.3). Depth dial: single it8, double it4.
- Multi-restart hurts (selection exploitation). Shooter weak. AC-tandem 60-70. TMD 66/56.
- PPO-LIP: was mid-training (iter ~1500/2500) when archived — retrain via lip_ppo.py cell in tasks.txt (~1h).
