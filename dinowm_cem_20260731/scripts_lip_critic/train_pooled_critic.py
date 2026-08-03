"""Train the pooled 384-d TD critic for the DINO LIP path.

Uses learners.td.fit directly (same as the sweep scripts) rather than
train_metric_sweep.py, which was never generated on this pod.

Applies tonight's one real finding: the stock recipe's 6000 steps x batch 1024 =
6.1M samples is ~4x undertrained. 24.6M samples lifted the full-token teacher
from spearman 0.483 -> 0.510 and pair_acc 0.693 -> 0.706 (past the LeWM
reference), and the disentangler showed it is total DATA that matters, not batch
size -- so 24000 steps at batch 1024, lr unchanged at 3e-4 (scaling lr with
batch measurably hurt: 0.4975 vs 0.5105).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, '/workspace/code/stable-worldmodel')

from stable_worldmodel.trm import LatentCache, learners, save_metric  # noqa: E402
from stable_worldmodel.trm.learners.td import TDConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--expectile', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--steps', type=int, default=24000)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--n-step', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()

    cache = LatentCache.load(a.cache)
    print(f'[critic] cache {len(cache.z)} x {cache.latent_dim}', flush=True)
    cfg = TDConfig(head='quasimetric', n_step=a.n_step, gamma=1.0,
                   expectile=a.expectile, lr=a.lr, batch_size=a.batch_size,
                   steps=a.steps, seed=a.seed)
    t0 = time.time()
    module = learners.td.fit(cache, cfg, a.device)
    print(f'[critic] trained in {(time.time()-t0)/60:.1f} min '
          f'({a.batch_size * a.steps / 1e6:.1f}M samples)', flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_metric(module.cpu(), 'td', cache.latent_dim,
                {'head': 'quasimetric', 'hidden_dim': cfg.hidden_dim,
                 'depth': cfg.depth, 'embed_dim': cfg.embed_dim,
                 'softplus': True, 'symmetric': False, 'num_components': 8},
                a.out)
    print(f'[critic] saved -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
