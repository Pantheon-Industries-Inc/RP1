"""Is the bottleneck the TD SIGNAL or the DINOv2 GEOMETRY?

Account under test: DINOv2 was pretrained for semantic discrimination, not
temporal structure, so feature distance does not track steps-to-go globally.
Latent MSE still plans well (80.0) because CEM only needs LOCAL smoothness over
5 steps; a value metric needs GLOBAL cost-to-go, which the geometry lacks.

But there is a competing explanation. Cube position is linearly decodable from
these features at R^2 0.976 -- the information IS there, and a supervised probe
finds it trivially. So maybe TD's weak bootstrapped signal is the problem, not
the geometry.

These make opposite predictions, and `--learner regression` separates them: it
fits Huber(m(z_i,z_j), |t_i-t_j|/s) with DIRECT supervision on true temporal
separation -- same features, same head budget, no bootstrapping.

  regression >> TD  -> the TD SIGNAL was the bottleneck. The geometry supports a
                      cost-to-go; TD just could not find it. Fix = better
                      supervision (regression teacher, or auxiliary targets).
  regression ~= TD  -> the GEOMETRY is the bottleneck, as argued. Frozen DINOv2
                      features do not admit a good global temporal metric, and no
                      amount of supervision changes that. Fix = a learned
                      projection trained with a temporal objective.

NB regression uses PairwiseMetricHead (MLP-style), not the quasimetric head. The
TD arch sweep found head='mlp' INVERTS under TD (spearman -0.29), so if
regression with an MLP head works, that is itself evidence the metric prior was
compensating for signal weakness rather than for geometry.
"""

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/workspace/code/stable-worldmodel')

from stable_worldmodel.trm import LatentCache, learners, load_metric  # noqa: E402
from stable_worldmodel.trm.learners.regression import RegressionConfig  # noqa: E402

import importlib.util as _iu
_spec = _iu.spec_from_file_location('arch', '/workspace/td_sweep_arch.py')
_arch = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_arch)
probe = _arch.probe


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-cache', default='/workspace/caches/dinopool_tr8000_fs1.pt')
    p.add_argument('--ho-cache', default='/workspace/caches/dinopool_ho.pt')
    p.add_argument('--td-metric', default='/workspace/metrics/dinopool_td_24k.pt')
    p.add_argument('--steps', type=int, default=24000)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    dev = args.device

    cache = LatentCache.load(args.train_cache)
    ho = LatentCache.load(args.ho_cache)
    ep_ho = ho.episode_idx.numpy().reshape(-1)
    st_ho = ho.step_idx.numpy().reshape(-1)
    print(f'[cmp] train {len(cache.z)}x{cache.latent_dim} | '
          f'held-out {len(ho.z)}x{ho.latent_dim}', flush=True)

    rows = []

    # --- existing TD critic (same cache, same sample budget) as the control
    try:
        td = load_metric(args.td_metric, device=dev).eval()
        r = probe(td, ho.z, ep_ho, st_ho, dev)
        rows.append(('TD (quasimetric, bootstrapped)',) + r)
        print(f'[cmp] TD          spearman={r[0]:.4f} pair_acc={r[1]:.4f} '
              f'monotone={r[2]:.4f}', flush=True)
        del td
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'[cmp] TD probe failed: {e}', flush=True)

    # --- supervised temporal regression on the identical features
    cfg = RegressionConfig(steps=args.steps, batch_size=args.batch_size, seed=0)
    t0 = time.time()
    module = learners.regression.fit(cache, cfg, dev)
    print(f'[cmp] regression trained in {(time.time()-t0)/60:.1f} min '
          f'({args.batch_size*args.steps/1e6:.1f}M samples)', flush=True)
    module = module.to(dev).eval()
    r = probe(module, ho.z, ep_ho, st_ho, dev)
    rows.append(('regression (pairwise, supervised)',) + r)
    print(f'[cmp] regression  spearman={r[0]:.4f} pair_acc={r[1]:.4f} '
          f'monotone={r[2]:.4f}', flush=True)

    print('\n=== VERDICT INPUTS (LeWM teacher: 0.484 / 0.691 / 0.923) ===')
    for nm, rho, pa, mo in rows:
        print(f'{nm:34s} spearman={rho:.4f} pair_acc={pa:.4f} monotone={mo:.4f}')
    if len(rows) == 2:
        d_mo = rows[1][3] - rows[0][3]
        d_pa = rows[1][2] - rows[0][2]
        print(f'\nregression - TD:  monotone {d_mo:+.4f}  pair_acc {d_pa:+.4f}')
        print('interpretation: a large positive delta implicates the TD SIGNAL; '
              'a flat delta implicates the DINOv2 GEOMETRY.')


if __name__ == '__main__':
    main()
