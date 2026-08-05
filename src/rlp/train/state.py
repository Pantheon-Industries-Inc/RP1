"""Train the lightweight state/proprio world model (StateWM).

JEPA-style latent dynamics with a small reconstruction head to keep the latent
informative (anti-collapse). Trains in minutes on CPU/MPS and is saved via the
repo's ``save_pretrained`` so it round-trips through ``load_pretrained`` /
``AutoCostModel`` and plugs into ``MetricCost`` + ``CEMSolver``.

Example::

    pixi run train model=state \
        dataset=tworoom_expert.lance output.model_name=statewm_tworoom obs_key=proprio steps=4000
"""

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from rlp.config import dispatch, run_hydra
from rlp.core.world_model import save_pretrained
from rlp.core.world_model.runtime import pick_device
from rlp.core.world_model.statewm import StateWM
from rlp.data.protocols import Dataset
from rlp.logging import logger

from .utils import sample_windows


def load_arrays(dataset: Dataset, obs_key: str) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Return per-episode row indices and (state, action) arrays."""
    ep = np.asarray(dataset.get_col_data("episode_idx")).reshape(-1)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    n = len(ep)
    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    for start in range(0, n, 512):
        rows = dataset.get_row_data(list(range(start, min(start + 512, n))))
        state_chunks.append(np.asarray(rows[obs_key]).reshape(len(rows[obs_key]), -1))
        action_chunks.append(np.asarray(rows["action"]).reshape(len(rows["action"]), -1))
    states = np.concatenate(state_chunks, 0).astype(np.float32)
    actions = np.nan_to_num(np.concatenate(action_chunks, 0).astype(np.float32))
    episodes: dict[int, np.ndarray] = {}
    for e in np.unique(ep):
        episode_rows = np.nonzero(ep == e)[0]
        episodes[int(e)] = episode_rows[np.argsort(st[episode_rows])]
    return episodes, states, actions


def _run(cfg: DictConfig) -> None:
    args = OmegaConf.merge(cfg, cfg.core.world_model)
    args.n_preds = args.num_predictions

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset = swm.data.load_dataset(args.dataset)
    episodes, states, actions = load_arrays(dataset, args.obs_key)
    state_dim, action_dim = states.shape[1], actions.shape[1]
    logger.info(
        f"loaded {len(states)} steps, {len(episodes)} episodes; "
        f"state_dim={state_dim} action_dim={action_dim} device={device}"
    )

    wm = StateWM(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        obs_key=args.obs_key,
        goal_key=f"goal_{args.obs_key}",
        state_scale=args.state_scale,
    ).to(device)
    decoder = nn.Sequential(  # train-only anti-collapse reconstruction head
        nn.Linear(args.latent_dim, args.hidden_dim),
        nn.SiLU(),
        nn.Linear(args.hidden_dim, state_dim),
    ).to(device)
    opt = torch.optim.AdamW(
        list(wm.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    win = args.history + args.n_preds
    wm.train()
    for step in range(args.steps):
        s, a = sample_windows(episodes, states, actions, win, args.batch_size, rng)
        s, a = s.to(device), a.to(device)
        z = wm.encode({wm.obs_key: s})["emb"]  # (B, win, D)
        act_emb = wm.action_encoder(a)  # (B, win, A)
        # multi-step latent rollout from t=history-1
        h = args.history
        pred = z[:, h - 1]
        pred_loss = torch.zeros((), device=device)
        for k in range(args.n_preds):
            pred = wm.predictor(pred.unsqueeze(1), act_emb[:, h - 1 + k : h + k])[:, 0]
            pred_loss = pred_loss + F.mse_loss(pred, z[:, h + k].detach())
        recon = decoder(z.reshape(-1, args.latent_dim))
        recon_loss = F.mse_loss(recon, s.reshape(-1, state_dim) / wm.state_scale)
        loss = pred_loss / args.n_preds + args.recon_weight * recon_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            logger.info(
                f"State-WM step {step}/{args.steps}: "
                f"prediction_loss={float(pred_loss / args.n_preds):.6f} reconstruction_loss={float(recon_loss):.6f}"
            )

    wm.eval()
    config = {
        "_target_": "rlp.core.world_model.statewm.StateWM",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "obs_key": args.obs_key,
        "goal_key": f"goal_{args.obs_key}",
        "state_scale": args.state_scale,
    }
    save_pretrained(wm.cpu(), run_name=args.output.model_name, config=config, cache_dir=args.run.directory)
    logger.success(f"StateWM saved to {args.run.checkpoints}/{args.output.model_name}")


def main() -> object:
    return run_hydra(dispatch, config_name="train/state")


if __name__ == "__main__":
    main()
