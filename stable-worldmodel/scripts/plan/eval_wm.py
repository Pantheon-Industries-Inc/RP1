"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import os
import sys

if sys.platform == 'linux':
    os.environ.setdefault('MUJOCO_GL', 'egl')

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = pick_device()


def img_transform(cfg, dtype=torch.float32):
    steps = [
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    ]
    # eval.train_res: bottleneck images through the checkpoint's native
    # training resolution (e.g. 64 for OGBench play retrains, whose 64px
    # frames were upscaled to 224 during training) so the model sees the
    # image domain it was trained on. null/img_size = no-op.
    train_res = cfg.eval.get('train_res', None)
    if train_res and int(train_res) != int(cfg.eval.img_size):
        steps.append(transforms.Resize(size=int(train_res)))
    steps.append(transforms.Resize(size=cfg.eval.img_size))
    return transforms.Compose(steps)


def episode_col(dataset):
    """Episode-index column name. Lance keeps 'episode_idx'/'step_idx' as
    writer-managed index columns outside column_names but serves them via
    get_col_data; h5 eval sets list 'ep_idx' (or 'episode_idx') explicitly."""
    return 'ep_idx' if 'ep_idx' in dataset.column_names else 'episode_idx'


def get_episodes_length(dataset, episodes):
    col_name = episode_col(dataset)

    # lance serves index columns as (N,1); h5 as (N,). Flatten so the boolean
    # mask below is 1-D regardless of source format.
    episode_idx = np.asarray(dataset.get_col_data(col_name)).reshape(-1)
    step_idx = np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    return dataset


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'goal': img_transform(cfg, img_dtype),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    # dataset.stats: source of the action/proprio z-scoring stats. Defaults
    # to the eval dataset; point it at the checkpoint's TRAINING dataset when
    # they differ (training normalizes with train-set stats, so eval must
    # match — e.g. OGBench play-data retrains vs expert-h5 eval).
    stats_name = cfg.dataset.get('stats', None) or cfg.eval.dataset_name
    if str(stats_name) == str(cfg.eval.dataset_name):
        stats_dataset = dataset
    else:
        stats_dataset = get_dataset(cfg, stats_name)
        print(
            f'[eval] z-stats from {stats_name} (train-convention override); '
            f'tasks/goals from {cfg.eval.dataset_name}'
        )
    col_name = episode_col(dataset)
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # -- run evaluation
    policy = cfg.get('policy', 'random')

    if policy not in ('random', 'nomove'):
        model = swm.wm.utils.load_pretrained(cfg.policy)
        if cfg.get('bf16', False):
            model = model.to(torch.bfloat16)
        model = model.to(DEVICE)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get('compile', False):
            encoder_attr = (
                'backbone' if hasattr(model, 'backbone') else 'encoder'
            )
            setattr(
                model,
                encoder_attr,
                torch.compile(getattr(model, encoder_attr)),
            )
            model.predictor = torch.compile(model.predictor)
        config = swm.PlanConfig(**cfg.plan_config)

        # optional plan-score metric hook: replace (or blend with) the raw
        # latent cost by a trained value function (ported from pod snapshot2)
        cost_model = model
        _metric = cfg.get('metric', None)
        if _metric and str(_metric).lower() not in ('none', 'null', 'latent'):
            from stable_worldmodel.trm import load_metric

            _paths = [q.strip() for q in str(_metric).split(',') if q.strip()]
            _mms = [load_metric(q, device=DEVICE) for q in _paths]
            _mm = _mms[0]
            _mode = cfg.get('cost_mode', 'replacement')

            class _MetricCost(torch.nn.Module):
                def __init__(self, base, metric, mode, metrics=None):
                    super().__init__()
                    self.base, self.metric, self.mode = base, metric, mode
                    self.metrics = torch.nn.ModuleList(metrics or [metric])
                    self.blend_w = 1.0
                    # history-conditioned metric (input = [z, dz]): detect 2x input dim
                    self.metric_in = next(
                        m.in_features
                        for m in metric.modules()
                        if isinstance(m, torch.nn.Linear)
                    )

                def parameters(self, *a, **k):
                    return self.base.parameters(*a, **k)

                _shapes_printed = [False]

                def get_cost(self, info_dict, action_candidates):
                    c_lat = self.base.get_cost(info_dict, action_candidates)
                    p_raw = info_dict['predicted_emb']
                    g_raw = info_dict['goal_emb']
                    if not self._shapes_printed[0]:
                        print(
                            f'[metric-hook] pred {tuple(p_raw.shape)} '
                            f'goal {tuple(g_raw.shape)}',
                            flush=True,
                        )
                        self._shapes_printed[0] = True
                    pred = p_raw[..., -1, :]
                    goal = g_raw[..., -1, :]
                    _D = pred.shape[-1]
                    _m = self.metric_in // _D if self.metric_in % _D == 0 else 1
                    if _m == 2:
                        # delta metric: state side [z_T, z_T - z_{T-1}] (block-
                        # scale delta), goal side [z_g, 0] ("arrive at rest")
                        prev = p_raw[..., -2, :]
                        pred = torch.cat([pred, pred - prev], dim=-1)
                        goal = torch.cat([goal, torch.zeros_like(goal)], dim=-1)
                    elif _m >= 3:
                        # window metric: last m imagined frames, goal tiled
                        pred = torch.cat(
                            [p_raw[..., -_m + j, :] for j in range(_m)], dim=-1)
                        goal = torch.cat([goal] * _m, dim=-1)
                    if goal.ndim < pred.ndim:
                        goal = goal.unsqueeze(1)
                    goal = goal.expand_as(pred)
                    # ensemble: pessimistic (max) distance across members.
                    # forward(), NOT cost(): cost() is @torch.no_grad, which
                    # detaches the score, so solver=adam (GradientSolver) trips
                    # its `costs.requires_grad` assert and no TD+Adam cell can
                    # run. CEM and MPPI both solve under @torch.inference_mode,
                    # so forward() builds no graph there and the numbers are
                    # unchanged; only the GD path gains the action gradient it
                    # needs. This is the same call LIPSolver makes on its own
                    # critic (lip_value(...)) for exactly this reason.
                    mcost = torch.stack(
                        [m(pred.float(), goal.float()) for m in self.metrics]
                    ).max(dim=0).values
                    if self.mode == 'replacement':
                        return mcost

                    def _std(x):
                        return (x - x.mean(-1, keepdim=True)) / x.std(
                            -1, keepdim=True
                        ).clamp_min(1e-6)

                    return _std(c_lat) + self.blend_w * _std(mcost)

            cost_model = _MetricCost(cost_model, _mm, _mode, metrics=_mms)
            cost_model.blend_w = float(cfg.get('blend_w', 1.0))
            print(f'[eval] plan-score metric={_metric} mode={_mode}')

        solver = hydra.utils.instantiate(cfg.solver, model=cost_model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = (swm.policy.NoMovePolicy() if policy == 'nomove'
                  else swm.policy.RandomPolicy())

    if cfg.get('video_dir'):
        results_path = Path(cfg.video_dir)
    else:
        results_path = (
            Path(
                swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
                cfg.policy,
            ).parent
            if cfg.policy != 'random'
            else Path(__file__).parent
        )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    # Map each dataset row’s episode_idx to its max_start_idx (flatten index
    # columns: lance serves them as (N,1), h5 as (N,)).
    _row_epi = np.asarray(dataset.get_col_data(col_name)).reshape(-1)
    _row_step = np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in _row_epi]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = _row_step <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    # optional hard-task filter (TwoRoom): keep only (start, goal) pairs on
    # opposite sides of the dividing wall (wall center 112; axis inferred
    # from the first door center in the observation vector)
    if cfg.eval.get('cross_wall', False):
        st_all = np.asarray(dataset.get_col_data('state'))
        first_door = np.asarray(dataset.get_col_data('observation'))[0, 4:6]
        axis = 0 if abs(float(first_door[0]) - 112.0) < 1e-3 else 1
        off = cfg.eval.goal_offset_steps
        s0 = st_all[valid_indices]
        s1 = st_all[valid_indices + off]
        cross = np.sign(s0[:, axis] - 112.0) != np.sign(s1[:, axis] - 112.0)
        valid_indices = valid_indices[cross]
        print(
            f'{len(valid_indices)} cross-wall starting points '
            f'(wall axis={axis})'
        )

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    # index columns may be writer-managed (lance): use column access, not rows.
    # Flatten (lance serves (N,1)) and cast to int (lance stores these as
    # float32; the reader indexes offsets with them and requires int).
    eval_episodes = (
        np.asarray(dataset.get_col_data(col_name)).reshape(-1)[
            random_episode_indices
        ].astype(np.int64)
    )
    eval_start_idx = (
        np.asarray(dataset.get_col_data('step_idx')).reshape(-1)[
            random_episode_indices
        ].astype(np.int64)
    )

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)
    print(
        f'[eval] saving videos to {results_path.resolve()} '
        '(one env_{i}.mp4 per env)'
    )

    autocast_ctx = torch.autocast(
        device_type=DEVICE if DEVICE != 'mps' else 'cpu',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    if cfg.get('compile', False):
        print('Warming up compiled model...')
        warmup_autocast_ctx = torch.autocast(
            device_type=DEVICE if DEVICE != 'mps' else 'cpu',
            dtype=torch.bfloat16,
            enabled=cfg.get('bf16', False),
        )
        with warmup_autocast_ctx:
            n = world.num_envs
            world.evaluate(
                dataset=dataset,
                start_steps=eval_start_idx.tolist()[:n],
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes.tolist()[:n],
                callables=OmegaConf.to_container(
                    cfg.eval.get('callables'), resolve=True
                ),
                video=results_path,
            )
        print('Warmup done.')

    start_time = time.time()
    with autocast_ctx:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get('callables'), resolve=True
            ),
            video=results_path,
        )
    end_time = time.time()

    print(metrics)
    print(f'[eval] videos saved to {results_path.resolve()}')

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()
