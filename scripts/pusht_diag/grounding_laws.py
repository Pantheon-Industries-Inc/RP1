"""Which physical law does the imagined motion UNDER CONTACT violate?

E32's first diagnostic (grounding_terms.py) showed the refiner's fabricated block
motion happens with the agent in contact, so a no-contact penalty cannot see it.
This rescoring asks, per deployed decision and per block of the plan, whether the
imagined block motion obeys quasi-static pushing:

  ratio  -- a pushed block cannot outrun the pusher: |db| / (|dp| + 5 px)
  along  -- the block moves along the push: cos(db, dp) on moving blocks
  away   -- the block moves away from the agent: cos(db, b - p) on moving blocks
  path   -- total imagined block path vs total agent path
  spin   -- imagined block rotation per block (deg), for reference

db = decoded block displacement over the block, dp = the agent's kinematic
displacement over the same block. Scores are the worst block per decision;
each is reported for failed vs successful episodes with the share of failures
above the successes' 90th percentile (10% = chance).

Env: D (experiment dir with probes_<COND>_s<draw>/ and evallog_<COND>_s<draw>.txt),
DRAWS (comma list, default "43,44"), COND (default "rlp").
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch

from rlp.core.grounding import GroundingPenalty

D = Path(os.environ["D"])
DRAWS = [int(x) for x in os.environ.get("DRAWS", "43,44").split(",")]
COND = os.environ.get("COND", "rlp")


def log(msg: str) -> None:
    print(f"[gl] {msg}", flush=True)


ck = torch.load(D / "train_checkpoints" / "planner.pt", map_location="cpu", weights_only=False)
g = GroundingPenalty.from_export(ck["grounding"])


def successes(draw: int) -> np.ndarray:
    text = (D / f"evallog_{COND}_s{draw}.txt").read_text()
    m = re.findall(r"episode_successes': array\(\[(.*?)\]\)", text, re.S)
    return np.array([t == "True" for t in re.findall(r"True|False", m[-1])])


def sep(name: str, score: np.ndarray, okk: np.ndarray) -> None:
    thr = np.quantile(score[okk], 0.9)
    log(
        f"   {name:6s} fail/succ mean {score[~okk].mean():7.3f}/{score[okk].mean():7.3f}  "
        f"median {np.median(score[~okk]):7.3f}/{np.median(score[okk]):7.3f}  "
        f"fails above successes' P90 ({thr:.3f}): {(score[~okk] > thr).mean() * 100:3.0f}%"
    )


for draw in DRAWS:
    files = sorted((D / f"probes_{COND}_s{draw}").glob("probe_*.pt"))
    if not files:
        log(f"draw {draw}: no probe files")
        continue
    ok = successes(draw)
    for k, f in enumerate(files):
        p = torch.load(f, map_location="cpu", weights_only=False)
        z0, traj, A = p["z0"].float(), p["z_traj"].float(), p["A"].float()
        ag0, up = p["ground_agent0"].float(), p["ground_u_prev"].float()
        n = min(len(ok), z0.shape[0])
        okk = ok[:n]
        with torch.no_grad():
            s0, st = g.decode(z0[:n]), g.decode(traj[:n])
            path = g.agent_path(A[:n], ag0[:n], up[:n])
        h = st.shape[1]
        fs = path.shape[1] // h
        b = torch.cat([s0[:, None, 2:4], st[..., 2:4]], dim=1)  # block position at block boundaries (n, H+1, 2)
        pa = torch.cat([ag0[:n, None], path.view(n, h, fs, 2)[:, :, -1]], dim=1)  # agent at block boundaries
        db, dp = b[:, 1:] - b[:, :-1], pa[:, 1:] - pa[:, :-1]
        nb, npa = db.norm(dim=-1), dp.norm(dim=-1)
        moving = nb > 10.0
        ratio = nb / (npa + 5.0)
        along = (db * dp).sum(-1) / (nb * npa + 1e-6)
        rel = b[:, :-1] - pa[:, :-1]
        away = (db * rel).sum(-1) / (nb * rel.norm(dim=-1) + 1e-6)
        ang = torch.atan2(torch.cat([s0[:, None, 5], st[..., 5]], 1), torch.cat([s0[:, None, 4], st[..., 4]], 1))
        spin = torch.rad2deg((ang[:, 1:] - ang[:, :-1] + torch.pi) % (2 * torch.pi) - torch.pi).abs()
        scores = {
            "ratio": ratio.amax(1).numpy(),
            "along": torch.where(moving, -along, torch.zeros_like(along)).amax(1).numpy(),  # > 0: against the push
            "away": torch.where(moving, -away, torch.zeros_like(away)).amax(1).numpy(),  # > 0: towards the agent
            "path": (nb.sum(1) / (npa.sum(1) + 5.0)).numpy(),
            "spin": spin.amax(1).numpy(),
        }
        log(
            f"draw {draw} decision {k}: n={n} fails={int((~okk).sum())} | total imagined block path fail/succ "
            f"{nb.sum(1)[~okk].mean():.1f}/{nb.sum(1)[okk].mean():.1f} px | agent path fail/succ "
            f"{npa.sum(1)[~okk].mean():.1f}/{npa.sum(1)[okk].mean():.1f} px | moving blocks fail/succ "
            f"{moving.float().sum(1)[~okk].mean():.2f}/{moving.float().sum(1)[okk].mean():.2f} of {h}"
        )
        for name, score in scores.items():
            sep(name, score, okk)
log("done")
