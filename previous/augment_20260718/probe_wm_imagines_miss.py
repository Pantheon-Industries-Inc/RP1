"""Probe: does v2WM IMAGINE the block lifting during a real grasp-miss?

Method: a linear readout R: latent -> block-height is fit on the expert cache
(which stores privileged_block_0_pos as `state`). Then, using the eval-time
context convention (3 real latent frames + ZERO action history), v2WM is rolled
forward under REAL action sequences and R reads the imagined block height.

- Calibration (expert SUCCESS episodes): real block lifts ~0.25; the WM rollout
  should ALSO imagine a lift => R(imagined) rises. Shows R + the WM both work.
- Test (real grasp-MISS episodes): reality = block stays down (<0.03). Does the
  WM imagine a lift anyway (ATTACHMENT OPTIMISM) or correctly stay flat (it CAN
  imagine the miss)?

Rollout matches training/eval (rollout_traj, 25-d z-scored frameskip-5 blocks).
"""
import h5py

try:
    import hdf5plugin  # noqa
except ImportError:
    pass
import numpy as np
import torch

import stable_worldmodel as swm
from stable_worldmodel.trm import LatentCache
from stable_worldmodel.solver.lip import rollout_traj

DEV = "cuda"
FS = 5
EXP_CACHE = "/workspace/caches/cube_v2wm_fs1.pt"
MISS_CACHE = "/workspace/caches/miss_fs1.pt"
EXP_H5 = "/workspace/datasets/lewm_cube_full/cube_single_expert.h5"
MISS_ALL = "/workspace/datasets/miss_long/miss_all.h5"
LIFT = 0.045          # block-lift threshold used by the collector's miss criterion

# ---- 1. fit linear block-height readout R on the expert cache ----------------
exp = LatentCache.load(EXP_CACHE)
assert exp.state is not None, "expert cache has no state (block pos)"
z_all, bz_all = exp.z, exp.state[:, 2]
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(z_all), generator=g)[:300000]
X = torch.cat([z_all[idx], torch.ones(len(idx), 1)], 1)
W = torch.linalg.lstsq(X, bz_all[idx]).solution
Wd = W.to(DEV)


def R(z):
    return torch.cat([z, torch.ones(z.shape[0], 1, device=z.device)], 1) @ Wd


pred = R(z_all[idx].to(DEV)).cpu()
r2 = 1 - ((bz_all[idx] - pred) ** 2).sum() / ((bz_all[idx] - bz_all[idx].mean()) ** 2).sum()
print(f"block-height readout R2 = {r2:.3f} (on expert latents)")

# ---- 2. WM + action normalization -------------------------------------------
m = swm.wm.utils.load_pretrained("/workspace/ckpts/ogbench_cube_single_v2WM").to(DEV).eval()
m.requires_grad_(False)
with h5py.File(EXP_H5, "r") as h:
    a_stat = h["action"][:1000000]
amu = np.nanmean(a_stat, 0)
astd = np.nanstd(a_stat, 0) + 1e-6


def blocks(act_prim):                     # (L,5) primitive -> (n,25) z-scored fs5 blocks
    an = (act_prim - amu) / astd
    n = len(an) // FS
    return an[:n * FS].reshape(n, FS * 5).astype(np.float32)


def imagine(z_fs5, blk):                  # eval convention: 3 real frames, ZERO a_hist
    T = min(len(z_fs5), len(blk))
    if T < 6:
        return None
    zh = z_fs5[:3].unsqueeze(0).to(DEV)                       # (1,3,D)
    ah = torch.zeros(1, 2, FS * 5, device=DEV)               # zero action history (eval)
    plan = torch.from_numpy(blk[2:T]).unsqueeze(0).to(DEV)   # (1,T-2,25)
    with torch.no_grad():
        traj = rollout_traj(m, zh, ah, plan)[0]              # (T-2, D) imagined steps 2..T-1
    return R(traj).cpu().numpy()                              # imagined block height per step


def episode_lift(cache, ep_rows, act_prim):
    z_fs5 = cache.z[ep_rows[::FS]]                            # real fs5 latents
    real_bz = cache.state[ep_rows, 2].numpy()                # real block height (primitive)
    blk = blocks(act_prim)
    imag_bz = imagine(z_fs5, blk)
    if imag_bz is None:
        return None
    return (real_bz.max() - real_bz[0],                      # real lift
            float(imag_bz.max() - imag_bz[0]))               # imagined lift


# ---- 3. calibration on expert SUCCESS episodes ------------------------------
with h5py.File(EXP_H5, "r") as h:
    e_off = h["ep_offset"][:]
    exp_eps = exp.episodes()
    real_s, imag_s = [], []
    for e in list(exp_eps)[:60]:
        rows = exp_eps[e]
        act = h["action"][int(e_off[e]):int(e_off[e]) + len(rows)]
        nan_rows = np.isnan(act).any(1)          # trim terminal NaN action padding
        if nan_rows.any():
            L = int(np.argmax(nan_rows))
            if L < 40:
                continue
            act, rows = act[:L], rows[:L]
        r = episode_lift(exp, rows, act)
        if r:
            real_s.append(r[0]); imag_s.append(r[1])
print(f"\nEXPERT SUCCESS (n={len(real_s)}): real lift {np.mean(real_s):.3f} | "
      f"WM-imagined lift {np.mean(imag_s):.3f}  "
      f"(WM imagines lift on {np.mean(np.array(imag_s) > LIFT) * 100:.0f}% -> readout+WM work)")

# ---- 4. test on real grasp-MISS episodes ------------------------------------
miss = LatentCache.load(MISS_CACHE)
miss_eps = miss.episodes()
with h5py.File(MISS_ALL, "r") as h:
    m_off = h["ep_offset"][:]
    real_m, imag_m = [], []
    for e in miss_eps:
        rows = miss_eps[e]
        act = h["action"][int(m_off[e]):int(m_off[e]) + len(rows)]
        r = episode_lift(miss, rows, act)
        if r:
            real_m.append(r[0]); imag_m.append(r[1])
real_m, imag_m = np.array(real_m), np.array(imag_m)
print(f"\nGRASP-MISS (n={len(real_m)}): real lift {real_m.mean():.3f} (all <{LIFT} = misses) | "
      f"WM-imagined lift {imag_m.mean():.3f}")
print(f"  WM IMAGINES A LIFT (>{LIFT}) on {np.mean(imag_m > LIFT) * 100:.0f}% of misses "
      f"=> that fraction is ATTACHMENT OPTIMISM (WM cannot imagine the miss)")
print(f"  WM correctly imagines NO lift on {np.mean(imag_m <= LIFT) * 100:.0f}% of misses")
print(f"\nVERDICT: imagined-miss lift {imag_m.mean():.3f} vs imagined-success lift "
      f"{np.mean(imag_s):.3f}; if miss~success, WM is blind to the miss.")
