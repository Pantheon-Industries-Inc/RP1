"""Which action convention does the DINO-WM predictor actually expect?

LIP and TD+CEM both roll the WM via rollout_terminal_dino / rollout_traj_dino,
which take RAW (de-normalized) actions: the trainers pass `A * ast5 + amu5`.
Plain CEM instead rolls through PreJEPA.rollout with the solver's own action
space, and the codebase's --action-stats-pin help says "eval un-scales with
expert stats" -- i.e. the solver may work in NORMALIZED units.

If the predictor's action embedder was trained on z-scored blocks, every LIP and
TD+CEM rollout has fed it actions ~3x too small (cube action std ~0.25-0.64),
while plain CEM fed the right ones. That would explain the entire pattern:
plain CEM 80.0, every learned-value planner 70-76, and off-manifold agreement
fine at h=1 (history-dominated) collapsing by h=5 (scale error compounding).

Test: from real 3-frame histories, roll ONE step with the TRUE logged action
under each convention and compare the predicted pooled latent to the TRUE
encoded latent at t+frameskip. Lower error = the convention the WM expects.

  raw  <  norm            -> raw is right; the rollouts were correct
  norm <  raw             -> BUG: LIP/TD+CEM have been mis-scaling actions
  raw ~= zero             -> the action pathway is inert at that scale
                             (the 'action-blind' failure mode from the retrain
                             campaign) -- also a bug, different cause

Also rolls 5 steps to see which convention keeps error growing slowest, since
h=5 is the planners' horizon.
"""

import argparse
import sys
from io import BytesIO

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/workspace/code/stable-worldmodel')

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.solver.lip import rollout_traj_dino  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
SPE, FS = 201, 5


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wm', default='/workspace/ckpts/dinowm_noprop_cube')
    p.add_argument('--dataset',
                   default='/root/datasets/ogb_cube_single/ogb_cube_single.lance')
    p.add_argument('--img-size', type=int, default=196)
    p.add_argument('--n', type=int, default=192)
    p.add_argument('--max-h', type=int, default=5)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    dev = a.device
    rng = np.random.default_rng(0)

    wm = swm.wm.utils.load_pretrained(a.wm).to(dev).eval()
    wm.requires_grad_(False)
    ds = swm.data.load_dataset(a.dataset)
    epi = np.asarray(ds.get_col_data('episode_idx')).reshape(-1).astype(np.int64)
    act = np.asarray(ds.get_col_data('action'), dtype=np.float32).reshape(-1, 5)
    amu = np.nanmean(act, 0)
    asd = np.nanstd(act, 0) + 1e-6
    print(f'[conv] action mu={np.round(amu,4).tolist()} sd={np.round(asd,4).tolist()}',
          flush=True)
    amu5 = torch.from_numpy(np.tile(amu, FS).astype(np.float32)).to(dev)
    asd5 = torch.from_numpy(np.tile(asd, FS).astype(np.float32)).to(dev)
    mean, std = MEAN.to(dev), STD.to(dev)

    def _dec(q):
        if isinstance(q, (bytes, bytearray, np.bytes_)):
            return np.array(Image.open(BytesIO(bytes(q))))
        return np.asarray(q)

    def enc(rows):
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

    eps = np.unique(epi)
    t_hi = SPE - FS * a.max_h - 2
    ctx = [(int(eps[rng.integers(len(eps))]),
            int(rng.integers(2 * FS, t_hi))) for _ in range(a.n)]

    err = {k: {h: [] for h in range(1, a.max_h + 1)}
           for k in ('raw', 'norm', 'zero')}
    for i0 in range(0, len(ctx), a.batch):
        ch = ctx[i0:i0 + a.batch]
        B = len(ch)
        base = np.array([e * SPE for e, _ in ch])
        tt = np.array([t for _, t in ch])
        hist = np.stack([base + tt - 2 * FS, base + tt - FS, base + tt], 1)
        toks = enc(hist)

        blocks = np.zeros((B, a.max_h, 25), dtype=np.float32)
        for j in range(a.max_h):
            for k in range(FS):
                blocks[:, j, k * 5:(k + 1) * 5] = np.nan_to_num(
                    act[base + tt + FS * j + k])
        raw = torch.from_numpy(blocks).to(dev)
        norm = (raw - amu5) / asd5
        zero = torch.zeros_like(raw)

        with torch.no_grad():
            trues = {}
            for h in range(1, a.max_h + 1):
                trues[h] = enc((base + tt + FS * h).reshape(B, 1))[:, 0, :, :384].mean(1)
            for name, plan in (('raw', raw), ('norm', norm), ('zero', zero)):
                traj = rollout_traj_dino(wm, toks, plan)      # (B,H,384) pooled
                for h in range(1, a.max_h + 1):
                    e = ((traj[:, h - 1] - trues[h]) ** 2).mean(-1)
                    err[name][h].append(e.cpu().numpy())

    print(f'\n{"h":>2} {"raw":>10} {"norm":>10} {"zero":>10}   verdict')
    print('-' * 52)
    for h in range(1, a.max_h + 1):
        r = float(np.concatenate(err['raw'][h]).mean())
        n = float(np.concatenate(err['norm'][h]).mean())
        z = float(np.concatenate(err['zero'][h]).mean())
        best = min(('raw', r), ('norm', n), ('zero', z), key=lambda x: x[1])[0]
        print(f'{h:>2} {r:>10.5f} {n:>10.5f} {z:>10.5f}   best={best}')
    print('\nlower = closer to the TRUE next latent.')
    print('norm << raw  => LIP/TD+CEM have been feeding mis-scaled actions (BUG).')
    print('raw ~= zero  => action pathway inert at that scale (action-blind).')


if __name__ == '__main__':
    main()
