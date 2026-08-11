"""LIP on DINO-WM (PreJEPA): learned iterative planner with PATCH-SPACE rollouts.

Differences vs LeWM LIP:
  - WM state = patch tokens (B, T, P=256, D=404) = 384 DINO pixels-emb + 10 proprio-emb
    + 10 action-emb (tiled per patch). Rollout: inject the plan block's action embedding
    into the last frame's action slice, predict next-frame tokens, repeat H times.
  - Value + PlannerNet operate on POOLED pixels-emb (384-d, matches cache & cf_dino value).
  - History action slices are ZERO in training (matches eval, where history actions are
    unavailable); proprio is real (from h5, via the frozen proprio Embedder).
  - Pixels for the 3 history frames are encoded on the fly from the h5 (no patch cache).
"""

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import stable_worldmodel as swm  # noqa
import torch
from omegaconf import DictConfig, OmegaConf

from rlp.config import dispatch, run_hydra
from rlp.core.planner import PlannerNet
from rlp.core.rollout import rollout_terminal_dino
from rlp.core.solver.lip import TokenEncoderWorldModel
from rlp.core.value import load_metric
from rlp.core.world_model import load_pretrained
from rlp.data import LatentCache
from rlp.logging import logger

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _run(cfg: DictConfig) -> None:
    a = OmegaConf.merge(cfg, cfg.core.planner)
    a.iters = a.iterations
    dev = "cuda"
    rng = np.random.default_rng(0)
    H = a.horizon

    wm_module = load_pretrained(a.wm).to(dev).eval()
    wm_module.requires_grad_(False)
    wm = cast(TokenEncoderWorldModel, wm_module)
    value = load_metric(a.value, device=dev)
    value.eval()
    for prm in value.parameters():
        prm.requires_grad_(False)
    c = LatentCache.load(a.cache)
    z = c.z.to(dev).float()  # pooled 384-d
    eps = c.episodes()
    keys = [k for k in eps if len(eps[k]) > a.max_delta + 4]
    ep_rows = {e: np.asarray(eps[e]) for e in keys}
    ep_ids = np.array(keys)
    h5 = h5py.File(a.h5, "r")
    act = h5["action"][:]
    ep_off = h5["ep_offset"][:]
    pro = h5["proprio"]
    amu = torch.tensor(act.mean(0)).float()
    ast = torch.tensor(act.std(0) + 1e-6).float()
    amu5 = amu.repeat(5).to(dev)
    ast5 = ast.repeat(5).to(dev)  # block denorm (25,)
    pix = h5["pixels"]
    mean, std = MEAN.to(dev), STD.to(dev)

    net = PlannerNet(z.shape[-1], horizon=H).to(dev)  # z_dim=384
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)

    def sample(B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zg: list[torch.Tensor] = []
        hframes: list[list[int]] = []
        for _ in range(B):
            e = int(ep_ids[rng.integers(len(ep_ids))])
            rows = ep_rows[e]
            L = len(rows)
            t = int(rng.integers(2, L - 2))
            fr = [int(ep_off[e] + 5 * int(c.step_idx[rows[t - k]])) for k in (2, 1, 0)]
            hframes.append(fr)
            if rng.random() < a.p_cross:
                e2 = int(ep_ids[rng.integers(len(ep_ids))])
                r2 = ep_rows[e2]
                zg.append(z[r2[rng.integers(len(r2))]])
            else:
                d = int(rng.integers(1, a.max_delta + 1))
                zg.append(z[rows[min(t + d, L - 1)]])
        frame_indices = np.array(hframes)
        px = np.stack([[pix[i] for i in row] for row in frame_indices])
        pr = np.stack([[pro[i] for i in row] for row in frame_indices])
        return (
            torch.from_numpy(px).to(dev),
            torch.from_numpy(pr).to(dev).float(),
            torch.stack(zg),
        )

    def encode_hist(px_u8: torch.Tensor, pr: torch.Tensor) -> torch.Tensor:
        B = px_u8.shape[0]
        px = px_u8.permute(0, 1, 4, 2, 3).float() / 255.0
        px = (px - mean) / std
        with torch.no_grad():
            info = {
                "pixels": px,
                "proprio": pr,
                "action": torch.zeros(B, 3, 25, device=dev),
            }  # zero action history (matches eval)
            info = wm.encode(info)
        return info["emb"].float()  # (B,3,P,404)

    for step in range(a.steps):
        px, pr, zg = sample(a.batch)
        toks = encode_hist(px, pr)
        z0 = toks[:, -1, :, :384].mean(dim=1)  # pooled current (B,384)
        B = z0.shape[0]
        A = torch.zeros(B, H, 25, device=dev)
        e_path = []
        for _ in range(a.iters):
            with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch 2.7 context-manager stub is untyped.
                A_in = A.detach().requires_grad_(True)
                zT = rollout_terminal_dino(wm, toks, A_in * ast5 + amu5)
                (gA,) = torch.autograd.grad(value(zT, zg).sum(), A_in)
            zT_c = rollout_terminal_dino(wm, toks, A * ast5 + amu5)
            E = value(zT_c, zg)
            A = net(A, gA, E, z0, zg)
            e_path.append(value(rollout_terminal_dino(wm, toks, A * ast5 + amu5), zg).mean())
        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if step % 100 == 0:
            logger.info(
                f"DINO-LIP step {step}/{a.steps}: E_final={e_path[-1].item():.3f} E_first={e_path[0].item():.3f}"
            )

    net.eval()
    checkpoint = Path(a.run.checkpoints) / a.output.checkpoint
    torch.save(
        {
            "kind": "lip_dino",
            "sd": net.cpu().state_dict(),
            "z_dim": z.shape[-1],
            "horizon": H,
            "iters": a.iters,
            "value": a.value,
            "amu5": amu5.cpu(),
            "ast5": ast5.cpu(),
        },
        checkpoint,
    )
    logger.success(f"Saved DINO LIP to {checkpoint}")


def main() -> object:
    return run_hydra(dispatch, config_name="train/lip_dino")


if __name__ == "__main__":
    main()
