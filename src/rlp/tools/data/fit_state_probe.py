"""Fit the linear latent -> state probe used by the contact-consistency penalty.

Ridge regression from a frozen-latent cache onto the logged environment state,
predicting ``[agent x, agent y, block x, block y, cos angle, sin angle]``. The
held-out R2 and median px errors are reported and stored in the artifact, so a
consumer can check the probe is good enough for the thresholds it uses.

Example::

    pixi run tool tool=fit_state_probe cache=caches/counterstrike_fs1.pt \
        state_cache=caches/counterstrike_state.pt out=caches/counterstrike_probe.pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from rlp.config import dispatch, run_hydra
from rlp.data import LatentCache
from rlp.logging import logger


def _run(cfg: DictConfig) -> None:
    cache = LatentCache.load(str(cfg.cache), mmap=True)
    state_cache = LatentCache.load(str(cfg.state_cache), mmap=False)
    if len(state_cache.z) != len(cache.z) or not torch.equal(state_cache.episode_idx, cache.episode_idx):
        raise ValueError(f"state rows ({len(state_cache.z)}) do not align with the cache ({len(cache.z)})")
    state = state_cache.z.numpy().astype(np.float64)
    if state.shape[1] < 5:
        raise ValueError(f"state must be (N, >=5) [agent xy, block xy, angle, ...], got {state.shape}")

    rng = np.random.default_rng(int(cfg.seed))
    n = min(int(cfg.max_rows), len(cache.z))
    rows = np.sort(rng.choice(len(cache.z), size=n, replace=False))
    X = cache.z[torch.as_tensor(rows)].float().numpy().astype(np.float64)
    S = state[rows]
    Y = np.column_stack([S[:, 0], S[:, 1], S[:, 2], S[:, 3], np.cos(S[:, 4]), np.sin(S[:, 4])])
    Xb = np.column_stack([X, np.ones(len(X))])
    split = int(0.9 * len(X))
    lam = float(cfg.ridge)
    A = Xb[:split].T @ Xb[:split] + lam * np.eye(Xb.shape[1])
    W = np.linalg.solve(A, Xb[:split].T @ Y[:split])

    pred = Xb[split:] @ W
    truth = Y[split:]
    r2 = 1 - ((pred - truth) ** 2).sum(0) / ((truth - truth.mean(0)) ** 2).sum(0)
    err_agent = np.linalg.norm(pred[:, :2] - truth[:, :2], axis=1)
    err_block = np.linalg.norm(pred[:, 2:4] - truth[:, 2:4], axis=1)
    ang_pred = np.arctan2(pred[:, 5], pred[:, 4])
    ang_true = np.arctan2(truth[:, 5], truth[:, 4])
    err_ang = np.degrees(np.abs((ang_pred - ang_true + np.pi) % (2 * np.pi) - np.pi))
    # Contact calibration: over expert transitions that MOVE the block, how far
    # can the agent centre be from the block centre? The penalty's contact
    # radius adopts a high percentile of that, so it never fires at a
    # separation where the data itself shows block motion.
    episodes = state_cache.episodes()
    seps: list[np.ndarray] = []
    for ep_rows in episodes.values():
        s = state[ep_rows]
        if len(s) < 2:
            continue
        d_pos = np.linalg.norm(s[1:, 2:4] - s[:-1, 2:4], axis=1)
        d_ang = np.abs((s[1:, 4] - s[:-1, 4] + np.pi) % (2 * np.pi) - np.pi)
        move = d_pos + float(cfg.angle_scale) * d_ang
        sep = np.linalg.norm(s[:-1, :2] - s[:-1, 2:4], axis=1)
        seps.append(sep[move > float(cfg.move_eps)])
    moving_sep = np.concatenate(seps) if seps else np.zeros(1)
    logger.info(
        f"contact calibration over {len(moving_sep)} block-moving expert transitions: agent-block separation "
        f"median {np.median(moving_sep):.1f} p95 {np.percentile(moving_sep, 95):.1f} p99 "
        f"{np.percentile(moving_sep, 99):.1f} max {moving_sep.max():.1f} px"
    )
    meta = {
        "rows": int(n),
        "contact_moving_transitions": int(len(moving_sep)),
        "contact_radius_p95": round(float(np.percentile(moving_sep, 95)), 2),
        "contact_radius_p99": round(float(np.percentile(moving_sep, 99)), 2),
        "contact_radius_max": round(float(moving_sep.max()), 2),
        "contact_move_eps": float(cfg.move_eps),
        "contact_angle_scale": float(cfg.angle_scale),
        "ridge": lam,
        "r2": [round(float(v), 4) for v in r2],
        "median_px_agent": round(float(np.median(err_agent)), 2),
        "median_px_block": round(float(np.median(err_block)), 2),
        "p90_px_block": round(float(np.percentile(err_block, 90)), 2),
        "median_deg_angle": round(float(np.median(err_ang)), 2),
        "cache": str(cfg.cache),
        "targets": "agent_x agent_y block_x block_y cos_angle sin_angle",
    }
    logger.info(
        f"probe held-out: R2 {meta['r2']}; median px agent {meta['median_px_agent']} block "
        f"{meta['median_px_block']} (p90 {meta['p90_px_block']}); median angle {meta['median_deg_angle']} deg"
    )
    out = Path(str(cfg.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weight": torch.as_tensor(W[:-1], dtype=torch.float32),
            "bias": torch.as_tensor(W[-1], dtype=torch.float32),
            "meta": meta,
        },
        out,
    )
    logger.success(f"Saved state probe (dim {W.shape[0] - 1} -> 6) to {out}")


def main() -> object:
    return run_hydra(dispatch, config_name="tools/fit_state_probe")


if __name__ == "__main__":
    main()
