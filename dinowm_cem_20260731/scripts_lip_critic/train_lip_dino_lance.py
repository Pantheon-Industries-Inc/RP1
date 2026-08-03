"""LIP on DINO-WM (PreJEPA), lance-backed — port of scripts/plan/train_lip_dino.py.

The stock trainer reads history pixels from an h5 and was written for the 224 px
PROPRIO dino variant (P=256, D=404). Ours is the noprop checkpoint (P=196,
D=394, 196 px) and our data is a 20 GB lance, so this port changes exactly three
things and nothing else:

  1. pixels come from the lance, not an h5. The lance is contiguous
     (10,000 episodes x 201 steps), and the fs5 cache RENUMBERS step_idx to
     0..40 (verified), so   row = episode*201 + 5*step_idx.
  2. `encode_hist` normalises at native resolution and THEN resizes to 196 --
     the same order as eval_wm.img_transform and cache_full.py. Doing it the
     other way round (resize uint8, then normalise) puts the actor in a
     different image domain than the critic and the planner.
  3. proprio is dropped. PreJEPA.encode only consumes keys present in
     `extra_encoders` (just `action` here), so proprio was never used by this
     checkpoint anyway.

Everything downstream -- the token rollout, the pooled 384-d value/PlannerNet
interface, the loss, the checkpoint format -- is untouched, so the actor stays
loadable by the stock LIPSolver dino path.
"""

import argparse
from io import BytesIO

import numpy as np
import torch
from PIL import Image

import stable_worldmodel as swm  # noqa: F401
from stable_worldmodel.trm import LatentCache, load_metric
from stable_worldmodel.solver.lip import PlannerNet, rollout_terminal_dino

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
STEPS_PER_EP = 201


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True, help='POOLED 384-d fs5 cache')
    p.add_argument('--dataset', required=True, help='lance dataset')
    p.add_argument('--wm', required=True)
    p.add_argument('--value', required=True, help='POOLED 384-d TD critic')
    p.add_argument('--out', required=True)
    p.add_argument('--img-size', type=int, default=196)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--iters', type=int, default=4)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=32,
                   help='micro-batch; effective batch = batch * accum')
    p.add_argument('--accum', type=int, default=1,
                   help='gradient-accumulation micro-steps')
    p.add_argument('--max-delta', type=int, default=10)
    p.add_argument('--p-cross', type=float, default=0.3)
    p.add_argument('--mean-weight', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--amax', type=float, default=None,
                   help='clip |normalized action| during training (None = off)')
    p.add_argument('--ckpt-every', type=int, default=0,
                   help='save an intermediate actor every N steps (0=off). LIP on\n                         token rollouts is ~6.6 s/step, so a single end-of-run\n                         artifact means hours before anything is evaluable.')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    dev = 'cuda'
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    H = a.horizon

    wm = swm.wm.utils.load_pretrained(a.wm).to(dev).eval()
    wm.requires_grad_(False)
    value = load_metric(a.value, device=dev)
    value.eval()
    for prm in value.parameters():
        prm.requires_grad_(False)

    c = LatentCache.load(a.cache)
    z = c.z.to(dev).float()
    assert z.shape[-1] == 384, f'expected pooled 384-d cache, got {z.shape[-1]}'
    eps = c.episodes()
    keys = [k for k in eps if len(eps[k]) > a.max_delta + 4]
    ep_rows = {e: np.asarray(eps[e]) for e in keys}
    ep_ids = np.array(keys)
    print(f'[lip] cache {tuple(z.shape)} | {len(ep_ids)} usable episodes', flush=True)

    ds = swm.data.load_dataset(a.dataset)
    act_all = np.asarray(ds.get_col_data('action'), dtype=np.float32).reshape(
        -1, 5)
    # The lance stores NaN actions on each episode's FINAL row (no successor),
    # so plain mean/std return NaN and poison amu5/ast5 -> loss NaN. The stock
    # trainer read a cleaned h5; reading the lance directly needs nan-safe stats.
    n_nan = int(np.isnan(act_all).any(axis=1).sum())
    amu = torch.tensor(np.nanmean(act_all, axis=0)).float()
    ast = torch.tensor(np.nanstd(act_all, axis=0) + 1e-6).float()
    assert torch.isfinite(amu).all() and torch.isfinite(ast).all(), 'action stats not finite'
    amu5 = amu.repeat(5).to(dev)
    ast5 = ast.repeat(5).to(dev)
    print(f'[lip] action stats (nan-safe, {n_nan} NaN rows skipped) '
          f'mu={[round(v, 4) for v in amu.tolist()]} '
          f'sd={[round(v, 4) for v in ast.tolist()]}', flush=True)

    mean, std = MEAN.to(dev), STD.to(dev)
    net = PlannerNet(z.shape[-1], horizon=H).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)

    def _decode(q):
        if isinstance(q, (bytes, bytearray, np.bytes_)):
            return np.array(Image.open(BytesIO(bytes(q))))
        return np.asarray(q)

    def sample(B):
        rows_flat, zg = [], []
        for _ in range(B):
            e = ep_ids[rng.integers(len(ep_ids))]
            rows = ep_rows[e]
            L = len(rows)
            t = int(rng.integers(2, L - 2))
            # fs5 step_idx is renumbered 0..40 -> original frame = 5 * step_idx
            for k in (2, 1, 0):
                s5 = int(c.step_idx[rows[t - k]])
                rows_flat.append(int(e) * STEPS_PER_EP + 5 * s5)
            if rng.random() < a.p_cross:
                e2 = ep_ids[rng.integers(len(ep_ids))]
                r2 = ep_rows[e2]
                zg.append(z[r2[rng.integers(len(r2))]])
            else:
                d = int(rng.integers(1, a.max_delta + 1))
                zg.append(z[rows[min(t + d, L - 1)]])
        # lance DEDUPLICATES repeated row indices, so a request for B*3 random
        # rows can come back short (near-certain once B*3 approaches the pool
        # size) and the reshape then fails. Fetch unique rows and scatter back.
        uniq, inv = np.unique(np.asarray(rows_flat), return_inverse=True)
        px = ds.get_row_data(uniq.tolist())['pixels']
        if isinstance(px, np.ndarray) and px.dtype == object:
            px = np.stack([_decode(q) for q in px])
        else:
            px = np.asarray(px)
            if px.ndim != 4:
                px = np.stack([_decode(q) for q in px])
        assert len(px) == len(uniq), f'got {len(px)} rows for {len(uniq)} unique'
        px = px[inv].reshape(B, 3, *px.shape[1:])     # (B,3,h,w,3) uint8
        return torch.from_numpy(px).to(dev), torch.stack(zg)

    def encode_hist(px_u8):
        """Normalise at native res, THEN resize -- matches eval and the cache."""
        px = px_u8.permute(0, 1, 4, 2, 3).float() / 255.0
        px = (px - mean) / std
        B, T = px.shape[:2]
        if px.shape[-1] != a.img_size:
            px = torch.nn.functional.interpolate(
                px.reshape(B * T, *px.shape[2:]), size=a.img_size,
                mode='bilinear', antialias=True, align_corners=False
            ).reshape(B, T, 3, a.img_size, a.img_size)
        with torch.no_grad():
            info = {'pixels': px,
                    'action': torch.zeros(B, 3, 25, device=dev)}  # zero hist, as eval
            info = wm.encode(info)
        return info['emb'].float()

    # Effective batch = batch * accum. Token-space rollouts hold an autograd
    # graph over iters x H predictor calls with 588-token attention, so batch
    # 256 in one go needs >139 GB (measured OOM). Accumulation gets the same
    # effective batch at 1/accum the peak memory.
    eff = a.batch * a.accum
    print(f'[lip] effective batch {eff} = {a.batch} x {a.accum} accum', flush=True)
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        first_e, last_e = None, None
        for micro in range(a.accum):
            px, zg = sample(a.batch)
            toks = encode_hist(px)
            if step == 0 and micro == 0:
                print(f'[lip] tokens {tuple(toks.shape)} (expect B,3,196,394)',
                      flush=True)
            z0 = toks[:, -1, :, :384].mean(dim=1)
            B = z0.shape[0]
            A = torch.zeros(B, H, 25, device=dev)
            e_path = []
            for _ in range(a.iters):
                with torch.enable_grad():
                    A_in = A.detach().requires_grad_(True)
                    zT = rollout_terminal_dino(wm, toks, A_in * ast5 + amu5)
                    (gA,) = torch.autograd.grad(value(zT, zg).sum(), A_in)
                zT_c = rollout_terminal_dino(wm, toks, A * ast5 + amu5)
                E = value(zT_c, zg)
                A = net(A, gA, E, z0, zg)
                if a.amax is not None:
                    A = A.clamp(-a.amax, a.amax)
                e_path.append(
                    value(rollout_terminal_dino(wm, toks, A * ast5 + amu5), zg).mean())
            loss = (e_path[-1] + a.mean_weight * torch.stack(e_path).mean()) / a.accum
            loss.backward()
            if micro == 0:
                first_e = e_path[0].item()
            last_e = e_path[-1].item()
            del toks, e_path, loss
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if step % 20 == 0:
            print(f'step {step}: E_final {last_e:.3f} E_first {first_e:.3f}',
                  flush=True)
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0:
            _save(net, a, z, H, amu5, ast5, f'{a.out[:-3]}_s{step+1}.pt')
            print(f'[lip] checkpoint @ step {step+1}', flush=True)

    net.eval()
    _save(net, a, z, H, amu5, ast5, a.out)
    print(f'saved DINO LIP -> {a.out}', flush=True)


def _save(net, a, z, H, amu5, ast5, path):
    """Snapshot the actor WITHOUT disturbing training. Three traps, all hit:

    1. `net.cpu().state_dict()` MOVES the model to CPU. Calling that from a
       periodic checkpoint left the net on CPU and the next forward died with
       "Expected all tensors to be on the same device". Copy tensors instead.
    2. Write to `path`, not `a.out` -- otherwise every periodic checkpoint
       overwrites the final artifact and the _sN files never appear.
    3. Never store `amax=None`. LIPSolver does ck.get("amax"), so a stored None
       overrides PlannerNet's own 2.5 default and crashes at deploy on
       `-self.amax`. Training used the 2.5 constructor default, so record that.
    """
    was = net.training
    net.eval()
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    amax = a.amax if a.amax is not None else 2.5
    torch.save({'kind': 'lip_dino', 'sd': sd,
                'z_dim': z.shape[-1], 'horizon': H, 'iters': a.iters,
                'value': a.value, 'amu5': amu5.cpu(), 'ast5': ast5.cpu(),
                'img_size': a.img_size, 'amax': amax}, path)
    net.train(was)


if __name__ == '__main__':
    main()
