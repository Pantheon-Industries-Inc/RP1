"""Learned Iterative Planner (LIP): a planning ALGORITHM learned purely by
optimizing against the value function. No behavior cloning anywhere.

The planner is an update rule f_theta(A_k, grad_A V, V_k, z0, zg) -> dA applied
for K iterations from A_0 = 0. Training unrolls the frozen WM for the full
horizon, evaluates the frozen value at the terminal latent, and backprops
loss = V(z_T(A_K), g) + 0.1 * mean_k V_k  through the whole refinement chain
to theta ONLY. Gradient features are detached (standard learned-optimizer
practice). Goals via HER (future + cross-episode), no MC labels needed.
"""

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlp.config import dispatch, run_hydra
from rlp.core.planner import PlannerNet, PlannerNetRec
from rlp.core.rollout import rollout_terminal, rollout_traj
from rlp.core.solver.lip import ValueFunction
from rlp.core.value import load_metric
from rlp.core.world_model import load_pretrained
from rlp.core.world_model.protocols import LatentWorldModel
from rlp.data import LatentCache
from rlp.logging import logger


def _run(cfg: DictConfig) -> None:
    a = OmegaConf.merge(cfg, cfg.core.planner)
    a.arch = a.architecture
    a.iters = a.iterations
    a.amax = a.action_limit
    a.s_dim = a.recurrent_state_dim
    a.s0_mode = a.recurrent_state_init
    a.rec_hidden = a.recurrent_hidden_dim
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    wm_module = load_pretrained(a.wm).to(dev).eval()
    wm_module.requires_grad_(False)
    wm = cast(LatentWorldModel, wm_module)
    value = load_metric(a.value, device=dev)
    for prm in value.parameters():
        prm.requires_grad_(False)
    value.eval()
    value_fn = cast(ValueFunction, value)

    c = LatentCache.load(a.cache)
    z = c.z.to(dev).float()
    eps = c.episodes()
    keys = [k for k in eps if len(eps[k]) > a.max_delta + 4]
    ep_rows = {e: np.asarray(eps[e]) for e in keys}
    ep_ids = np.array(keys)
    with h5py.File(a.h5, "r") as h:
        act = h["action"][:]
        ep_off = h["ep_offset"][:]
    amu, astd = act.mean(0), act.std(0) + 1e-6
    act_n = ((act - amu) / astd).astype(np.float32)

    def blocks(e: int, t: int) -> np.ndarray:
        h0 = int(ep_off[e] + 5 * t)
        return np.asarray(act_n[h0 : h0 + 5]).reshape(-1)

    def sample(B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zh: list[torch.Tensor] = []
        ah: list[np.ndarray] = []
        zg: list[torch.Tensor] = []
        for _ in range(B):
            e = int(ep_ids[rng.integers(len(ep_ids))])
            rows = ep_rows[e]
            L = len(rows)
            t = int(rng.integers(2, L - 2))
            zh.append(torch.stack([z[rows[t - 2]], z[rows[t - 1]], z[rows[t]]]))
            ah.append(np.stack([blocks(e, t - 2), blocks(e, t - 1)]))
            if rng.random() < a.p_cross:
                e2 = int(ep_ids[rng.integers(len(ep_ids))])
                r2 = ep_rows[e2]
                zg.append(z[r2[rng.integers(len(r2))]])
            else:
                d = int(rng.integers(1, a.max_delta + 1))
                zg.append(z[rows[min(t + d, L - 1)]])
        return (
            torch.stack(zh),
            torch.from_numpy(np.stack(ah)).to(dev),
            torch.stack(zg),
        )

    a_dim = act.shape[-1] * 5  # action-block dim = env action dim x frameskip
    net: PlannerNet | PlannerNetRec
    if a.arch == "v4r":
        net = PlannerNetRec(
            z.shape[-1],
            horizon=a.horizon,
            a_dim=a_dim,
            hidden=a.rec_hidden,
            s_dim=a.s_dim,
            amax=a.amax,
            use_z0=False,
            use_zg=False,
            s0_mode=a.s0_mode,
            head_scale=a.head_scale,
        ).to(dev)
    elif a.arch == "v4":
        net = PlannerNet(
            z.shape[-1],
            horizon=a.horizon,
            a_dim=a_dim,
            feed=a.feed,
            amax=a.amax,
            use_zg=False,
            use_gate=False,
            use_z0=False,
            head_scale=a.head_scale,
        ).to(dev)
    else:
        net = PlannerNet(
            z.shape[-1],
            horizon=a.horizon,
            a_dim=a_dim,
            feed=a.feed,
            amax=a.amax,
            head_scale=a.head_scale,
        ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)

    for step in range(a.steps):
        zh, ah, zg = sample(a.batch)
        z0 = zh[:, -1]
        A = torch.zeros(a.batch, a.horizon, a_dim, device=dev)
        s = net.init_state(a.batch, z0) if isinstance(net, PlannerNetRec) else None
        e_path: list[torch.Tensor] = []
        for _ in range(a.iters):
            # gradient feature (detached — input to the learned rule, not the training path)
            with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch 2.7 context-manager stub is untyped.
                A_in = A.detach().requires_grad_(True)
                traj = rollout_traj(wm, zh, ah, A_in)
                (gA,) = torch.autograd.grad(value_fn(traj[:, -1], zg).sum(), A_in)
            traj_f = rollout_traj(wm, zh, ah, A).detach()
            E_feat = value_fn(traj_f[:, -1], zg).detach()
            if isinstance(net, PlannerNetRec):
                if s is None:
                    raise RuntimeError("recurrent actor state is unavailable")
                A, s = net(A, gA.detach(), E_feat, z0, zg, s, traj_f)
            else:
                A = net(A, gA.detach(), E_feat, z0, zg, traj_f)  # theta-dependent update
            e_path.append(value_fn(rollout_terminal(wm, zh, ah, A), zg).mean())
        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if step % 500 == 0:
            logger.info(f"LIP step {step}/{a.steps}: E_final={e_path[-1].item():.3f} E_first={e_path[0].item():.3f}")

    net.eval()
    if a.arch == "v4r":
        kind = "lip4r"
    elif a.arch == "v4":
        kind = "lip4"
    else:
        kind = "lip" if a.feed == "none" else "lip2"
    checkpoint = Path(a.run.checkpoints) / a.output.checkpoint
    torch.save(
        {
            "kind": kind,
            "feed": a.feed,
            "sd": net.cpu().state_dict(),
            "z_dim": z.shape[-1],
            "horizon": a.horizon,
            "iters": a.iters,
            "a_dim": a_dim,
            "amax": a.amax,
            "s_dim": a.s_dim,
            "s0_mode": a.s0_mode,
            "hidden": a.rec_hidden,
            "use_gate": net.use_gate,
            "use_zg": net.use_zg,
            "use_z0": net.use_z0,
            "head_scale": a.head_scale,
            "value": a.value,
        },
        checkpoint,
    )
    logger.success(f"Saved learned planner to {checkpoint}")


def main() -> object:
    return run_hydra(dispatch, config_name="train/lip")


if __name__ == "__main__":
    main()
