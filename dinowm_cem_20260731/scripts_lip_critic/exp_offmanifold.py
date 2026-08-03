"""Does the learned metric break on IMAGINED latents, where planners use it?

Every offline probe tonight scored the metric on REAL encoded states. But CEM and
LIP never query it there -- they query d(predicted_z, goal_z) on latents the WM
imagined. On the unconfounded measures the DINO critic matches or beats the LeWM
teacher (spearman 0.5105 / pair_acc 0.7044 vs 0.484 / 0.691), yet planning with
it is 10 points worse than plain latent MSE. That gap is unexplained.

Hypothesis: the predictor's imagined latents drift off the real-feature manifold,
and a LEARNED metric can be arbitrarily wrong out there, while MSE degrades
gracefully. That would explain why plain CEM survives and both metric-based
planners do not.

Protocol, per sampled (episode, t) on HELD-OUT episodes:
  * encode the real 3-frame history at t
  * roll the WM h blocks forward using the TRUE logged actions (h=1..5 blocks;
    one block = 5 primitive steps, so h=5 == the planner's 25-step horizon)
  * encode the REAL frame at t+5h  -> z_real
  * goal = real encoded frame at t+goal_offset -> z_g
  * compare, for the metric d and for latent MSE:
        agreement  corr( f(imagined, z_g), f(real, z_g) )
        usefulness spearman( f(imagined, z_g), true steps remaining )

Decisive comparison: if the metric's agreement / usefulness collapses with h
while MSE's holds up, the mechanism is confirmed -- the metric is fine on real
states and unreliable on imagined ones, which is exactly where planners read it.
"""

import argparse
import sys
from io import BytesIO

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/workspace/code/stable-worldmodel')

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.trm import load_metric  # noqa: E402
from stable_worldmodel.solver.lip import rollout_terminal_dino  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
STEPS_PER_EP = 201
FRAMESKIP = 5


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float('nan')


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wm', default='/workspace/ckpts/dinowm_noprop_cube')
    p.add_argument('--metric', default='/workspace/metrics/dinopool_td_24k.pt')
    p.add_argument('--dataset',
                   default='/root/datasets/ogb_cube_single/ogb_cube_single.lance')
    p.add_argument('--img-size', type=int, default=196)
    p.add_argument('--ep-lo', type=int, default=8000)
    p.add_argument('--goal-offset', type=int, default=25)
    p.add_argument('--n', type=int, default=256, help='sampled contexts')
    p.add_argument('--max-h', type=int, default=5, help='rollout blocks')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    dev = a.device
    rng = np.random.default_rng(0)

    wm = swm.wm.utils.load_pretrained(a.wm).to(dev).eval()
    wm.requires_grad_(False)
    metric = load_metric(a.metric, device=dev).eval()
    ds = swm.data.load_dataset(a.dataset)
    epi = np.asarray(ds.get_col_data('episode_idx')).reshape(-1).astype(np.int64)
    act = np.asarray(ds.get_col_data('action'), dtype=np.float32).reshape(-1, 5)
    mean, std = MEAN.to(dev), STD.to(dev)

    def _dec(q):
        if isinstance(q, (bytes, bytearray, np.bytes_)):
            return np.array(Image.open(BytesIO(bytes(q))))
        return np.asarray(q)

    def enc(rows):
        """rows: (B,T) absolute lance row indices -> (B,T,P,D) tokens."""
        flat = np.asarray(rows).reshape(-1)
        uniq, inv = np.unique(flat, return_inverse=True)
        px = ds.get_row_data(uniq.tolist())['pixels']
        if isinstance(px, np.ndarray) and px.dtype == object:
            px = np.stack([_dec(q) for q in px])
        else:
            px = np.asarray(px)
            if px.ndim != 4:
                px = np.stack([_dec(q) for q in px])
        px = px[inv].reshape(*np.asarray(rows).shape, *px.shape[1:])
        x = torch.from_numpy(px).to(dev).permute(0, 1, 4, 2, 3).float() / 255.0
        x = (x - mean) / std
        B, T = x.shape[:2]
        if x.shape[-1] != a.img_size:
            x = torch.nn.functional.interpolate(
                x.reshape(B * T, *x.shape[2:]), size=a.img_size, mode='bilinear',
                antialias=True, align_corners=False).reshape(B, T, 3, a.img_size, a.img_size)
        with torch.no_grad():
            info = wm.encode({'pixels': x,
                              'action': torch.zeros(B, T, 25, device=dev)})
        return info['emb'].float()

    eps = np.unique(epi[epi >= a.ep_lo])
    # need history (2 back) + max rollout + goal offset inside the episode
    t_hi = STEPS_PER_EP - (a.goal_offset + FRAMESKIP * a.max_h) - 1
    ctx = []
    while len(ctx) < a.n:
        e = int(eps[rng.integers(len(eps))])
        t = int(rng.integers(2 * FRAMESKIP, max(t_hi, 2 * FRAMESKIP + 1)))
        ctx.append((e, t))

    print(f'[off] {len(ctx)} contexts, held-out eps >= {a.ep_lo}, '
          f'h=1..{a.max_h} blocks (1 block = {FRAMESKIP} primitive steps)', flush=True)
    print(f'[off] metric {a.metric}', flush=True)

    res = {h: {'d_im': [], 'd_re': [], 'm_im': [], 'm_re': [], 'rem': [], 'drift': []}
           for h in range(1, a.max_h + 1)}

    for i0 in range(0, len(ctx), a.batch):
        chunk = ctx[i0:i0 + a.batch]
        B = len(chunk)
        base = np.array([e * STEPS_PER_EP for e, _ in chunk])
        tt = np.array([t for _, t in chunk])
        # 3-frame history at t-2fs, t-fs, t
        hist = np.stack([base + tt - 2 * FRAMESKIP, base + tt - FRAMESKIP, base + tt], 1)
        toks = enc(hist)                                     # (B,3,P,D)
        # goal = real frame at t + goal_offset
        zg = enc((base + tt + a.goal_offset).reshape(B, 1))[:, 0, :, :384].mean(1)

        # true action blocks: block j = primitive actions t+5j .. t+5j+4
        blocks = np.zeros((B, a.max_h, 25), dtype=np.float32)
        for j in range(a.max_h):
            for k in range(FRAMESKIP):
                idx = base + tt + FRAMESKIP * j + k
                blocks[:, j, k * 5:(k + 1) * 5] = np.nan_to_num(act[idx])
        blk = torch.from_numpy(blocks).to(dev)

        for h in range(1, a.max_h + 1):
            with torch.no_grad():
                z_im = rollout_terminal_dino(wm, toks, blk[:, :h])   # (B,384) pooled
                z_re = enc((base + tt + FRAMESKIP * h).reshape(B, 1))[:, 0, :, :384].mean(1)
                d_im = metric.cost(z_im, zg).float().cpu().numpy().reshape(-1)
                d_re = metric.cost(z_re, zg).float().cpu().numpy().reshape(-1)
                m_im = ((z_im - zg) ** 2).mean(-1).cpu().numpy().reshape(-1)
                m_re = ((z_re - zg) ** 2).mean(-1).cpu().numpy().reshape(-1)
                drift = ((z_im - z_re) ** 2).mean(-1).sqrt().cpu().numpy().reshape(-1)
            res[h]['d_im'].append(d_im); res[h]['d_re'].append(d_re)
            res[h]['m_im'].append(m_im); res[h]['m_re'].append(m_re)
            res[h]['drift'].append(drift)
            res[h]['rem'].append(np.full(B, a.goal_offset - FRAMESKIP * h, dtype=np.float64))

    print(f'\n{"h":>2} {"drift":>8} | {"metric agree":>12} {"MSE agree":>10} '
          f'| {"metric rank":>11} {"MSE rank":>9}')
    print('-' * 66)
    for h in range(1, a.max_h + 1):
        r = {k: np.concatenate(v) for k, v in res[h].items()}
        ok = np.isfinite(r['d_im']) & np.isfinite(r['d_re'])
        agree_d = pearson(r['d_im'][ok], r['d_re'][ok])
        agree_m = pearson(r['m_im'][ok], r['m_re'][ok])
        # usefulness: does the score computed on IMAGINED latents rank the
        # remaining true distance-to-goal? (within-h it is constant, so rank
        # against the REAL-state score, the best available per-sample target)
        rank_d = spearman(r['d_im'][ok], r['d_re'][ok])
        rank_m = spearman(r['m_im'][ok], r['m_re'][ok])
        print(f'{h:>2} {r["drift"].mean():>8.4f} | {agree_d:>12.4f} {agree_m:>10.4f} '
              f'| {rank_d:>11.4f} {rank_m:>9.4f}')

    print('\nagree = Pearson( score(imagined, goal), score(real, goal) )')
    print('rank  = Spearman of the same pair (ordering agreement)')
    print('If the metric columns fall away with h while MSE holds, the metric is')
    print('unreliable exactly where planners read it -- mechanism confirmed.')


if __name__ == '__main__':
    main()
