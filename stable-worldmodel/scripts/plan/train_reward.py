#!/usr/bin/env python3
"""Train the offline goal-conditioned reward head PWM needs.

PWM's world model has three supervised components -- encoder, dynamics and a
reward predictor R_phi. Our author-released world models supply the first two and
no reward head, so this trains one on cached latents.

The label is the evaluation protocol's own success criterion, computed from the
oracle state stored in the cache, NOT the dataset's `reward` column: that column
refers to each episode's own target, whereas the protocol replays row t as start
and row t+offset as goal. See stable_worldmodel/trm/learners/reward.py.

    python3 train_reward.py \\
        --cache /workspace/caches/tworoom_canon_lejepa_fs1.pt \\
        --out   /workspace/metrics/reward_lejepa.pt

The cache must have been built with `--state-key pos_agent` (canonical TwoRoom).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from loguru import logger as logging

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_worldmodel.trm import LatentCache  # noqa: E402
from stable_worldmodel.trm.learners.reward import RewardConfig, fit  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True, help='fs1 cache built with --state-key')
    p.add_argument('--out', required=True)
    p.add_argument('--success-radius', type=float, default=16.0,
                   help='TwoRoom terminates within 16 px of the target')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n-step', type=int, default=50)
    p.add_argument('--p-cross', type=float, default=0.3)
    p.add_argument('--max-delta', type=int, default=None)
    p.add_argument('--pos-weight', type=float, default=None,
                   help='override the auto-computed class rebalancing')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    a = p.parse_args()

    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    cache = LatentCache.load(a.cache)
    cfg = RewardConfig(
        hidden_dim=a.hidden_dim, depth=a.depth, lr=a.lr,
        batch_size=a.batch_size, steps=a.steps,
        success_radius=a.success_radius, n_step=a.n_step, p_cross=a.p_cross,
        max_delta=a.max_delta, pos_weight=a.pos_weight, seed=a.seed,
    )
    head = fit(cache, cfg, device=dev)

    os.makedirs(Path(a.out).parent, exist_ok=True)
    torch.save({
        'learner': 'reward',
        'latent_dim': cache.latent_dim,
        'arch': {'hidden_dim': a.hidden_dim, 'depth': a.depth,
                 'success_radius': a.success_radius},
        'state_dict': head.cpu().state_dict(),
    }, a.out)
    logging.success(f'saved reward head -> {a.out}')


if __name__ == '__main__':
    main()
