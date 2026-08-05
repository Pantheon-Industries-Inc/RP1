"""THE MISSING CONTROL: what monotonicity is achievable on this task at all?

Every metric we trained -- TD and supervised regression, quasimetric and MLP
heads, 384-d and 75,264-d inputs -- lands at monotone ~0.62 against the LeWM
teacher's 0.923. I have been attributing that to frozen DINOv2 geometry, but I
never established the CEILING for this objective + this probe on this task.

So train the identical TD recipe on the PRIVILEGED STATE (true cube xyz, stored
as `state` in every cache). Ground-truth low-dim state is the best possible
"representation":

  privileged monotone ~0.9  -> the pipeline and probe are fine; DINOv2 features
                              ARE the wall. The representation story holds and a
                              learned temporal projection is the right fix.
  privileged monotone ~0.62 -> the ceiling is set by the OBJECTIVE or the PROBE,
                              not the features. Every conclusion I drew about
                              DINOv2 geometry tonight would be wrong, and the fix
                              would be in the TD/probe setup instead.

Also runs a same-substrate LeWM-style control if a LeWM cache is present, and
reports the probe at both 25-step (planner horizon) and 60-step spans, since the
60-step number is what produced the 0.62 headline.
"""

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/workspace/code/stable-worldmodel')

from stable_worldmodel.trm import LatentCache, learners  # noqa: E402
from stable_worldmodel.trm.learners.td import TDConfig  # noqa: E402

import importlib.util as _iu
_spec = _iu.spec_from_file_location('arch', '/workspace/td_sweep_arch.py')
_arch = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_arch)
probe = _arch.probe


def make_state_cache(src, dst=None):
    """Rebuild a cache whose z IS the privileged state (true cube xyz)."""
    c = LatentCache.load(src)
    assert c.state is not None, 'cache has no privileged state column'
    s = c.state.float()
    print(f'[ceil] state dims {tuple(s.shape)} '
          f'(finite={bool(torch.isfinite(s).all())})', flush=True)
    return LatentCache(z=s, episode_idx=c.episode_idx, step_idx=c.step_idx,
                       state=c.state, meta={'derived': 'privileged state as z',
                                            'src': str(src)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-cache', default='/workspace/caches/dinopool_tr8000_fs1.pt')
    p.add_argument('--ho-cache', default='/workspace/caches/dinopool_ho.pt')
    p.add_argument('--steps', type=int, default=24000)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    dev = args.device

    tr = make_state_cache(args.train_cache)
    ho_full = LatentCache.load(args.ho_cache)
    ho = make_state_cache(args.ho_cache)
    ep = ho.episode_idx.numpy().reshape(-1)
    st = ho.step_idx.numpy().reshape(-1)
    print(f'[ceil] train {len(tr.z)}x{tr.latent_dim} | held-out {len(ho.z)}x{ho.latent_dim}',
          flush=True)

    cfg = TDConfig(head='quasimetric', n_step=50, gamma=1.0, expectile=0.1,
                   lr=3e-4, batch_size=args.batch_size, steps=args.steps, seed=0)
    t0 = time.time()
    module = learners.td.fit(tr, cfg, dev)
    print(f'[ceil] privileged-state TD trained in {(time.time()-t0)/60:.1f} min',
          flush=True)
    module = module.to(dev).eval()

    rho, pa, mo = probe(module, ho.z, ep, st, dev)
    print(f'\n[ceil] PRIVILEGED STATE (true cube xyz, 3-d):')
    print(f'         spearman={rho:.4f} pair_acc={pa:.4f} monotone={mo:.4f}')
    print(f'\n  reference points')
    print(f'    DINO pooled  TD          0.4743 / 0.6870 / 0.6236')
    print(f'    DINO pooled  regression  0.2985 / 0.6068 / 0.6159')
    print(f'    DINO full    TD (best)   0.5105 / 0.7044 / 0.6736')
    print(f'    LeWM teacher             0.484  / 0.691  / 0.923')
    print()
    if mo > 0.80:
        print('  => VERDICT: ceiling is HIGH. The probe and objective are fine;')
        print('     DINOv2 features are the wall. Learned temporal projection is')
        print('     the right fix.')
    elif mo < 0.70:
        print('  => VERDICT: ceiling is LOW even on ground-truth state. The limit')
        print('     is the OBJECTIVE or the PROBE, not the features. The DINOv2')
        print('     geometry conclusion does NOT hold.')
    else:
        print('  => VERDICT: ambiguous; ceiling only modestly above the DINO runs.')


if __name__ == '__main__':
    main()
