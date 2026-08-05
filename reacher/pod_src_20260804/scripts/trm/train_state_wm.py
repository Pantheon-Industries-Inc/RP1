"""Train the lightweight state/proprio world model (StateWM).

JEPA-style latent dynamics with a small reconstruction head to keep the latent
informative (anti-collapse). Trains in minutes on CPU/MPS and is saved via the
repo's ``save_pretrained`` so it round-trips through ``load_pretrained`` /
``AutoCostModel`` and plugs into ``MetricCost`` + ``CEMSolver``.

Example::

    python scripts/trm/train_state_wm.py --dataset tworoom_expert.lance \
        --run-name statewm_tworoom --obs-key proprio --steps 4000
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as logging
from torch import nn
from tqdm import tqdm

import stable_worldmodel as swm
from stable_worldmodel.wm.statewm import StateWM
from stable_worldmodel.wm.utils import save_pretrained


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_arrays(dataset, obs_key: str):
    """Return per-episode row indices and (state, action) arrays."""
    ep = np.asarray(dataset.get_col_data("episode_idx")).reshape(-1)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    n = len(ep)
    states, actions = [], []
    for start in range(0, n, 512):
        rows = dataset.get_row_data(list(range(start, min(start + 512, n))))
        states.append(np.asarray(rows[obs_key]).reshape(len(rows[obs_key]), -1))
        actions.append(np.asarray(rows["action"]).reshape(len(rows["action"]), -1))
    states = np.concatenate(states, 0).astype(np.float32)
    actions = np.nan_to_num(np.concatenate(actions, 0).astype(np.float32))
    episodes = {}
    for e in np.unique(ep):
        rows = np.nonzero(ep == e)[0]
        episodes[int(e)] = rows[np.argsort(st[rows])]
    return episodes, states, actions


def sample_windows(episodes, states, actions, win, batch, rng):
    valid = [e for e, r in episodes.items() if len(r) >= win]
    s_out = np.empty((batch, win, states.shape[1]), np.float32)
    a_out = np.empty((batch, win, actions.shape[1]), np.float32)
    for b in range(batch):
        e = valid[rng.integers(0, len(valid))]
        rows = episodes[e]
        t = rng.integers(0, len(rows) - win + 1)
        idx = rows[t : t + win]
        s_out[b] = states[idx]
        a_out[b] = actions[idx]
    return torch.from_numpy(s_out), torch.from_numpy(a_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--obs-key", default="proprio")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--state-scale", type=float, default=224.0)
    p.add_argument("--n-preds", type=int, default=3, help="multi-step rollout length in training")
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset = swm.data.load_dataset(args.dataset)
    episodes, states, actions = load_arrays(dataset, args.obs_key)
    state_dim, action_dim = states.shape[1], actions.shape[1]
    logging.info(
        f"loaded {len(states)} steps, {len(episodes)} episodes; "
        f"state_dim={state_dim} action_dim={action_dim} device={device}"
    )

    wm = StateWM(
        state_dim=state_dim, action_dim=action_dim, latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim, obs_key=args.obs_key, goal_key=f"goal_{args.obs_key}",
        state_scale=args.state_scale,
    ).to(device)
    decoder = nn.Sequential(  # train-only anti-collapse reconstruction head
        nn.Linear(args.latent_dim, args.hidden_dim), nn.SiLU(),
        nn.Linear(args.hidden_dim, state_dim),
    ).to(device)
    opt = torch.optim.AdamW(
        list(wm.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=1e-4
    )

    win = args.history + args.n_preds
    pbar = tqdm(range(args.steps), desc="state-wm")
    wm.train()
    for step in pbar:
        s, a = sample_windows(episodes, states, actions, win, args.batch_size, rng)
        s, a = s.to(device), a.to(device)
        z = wm.encode({wm.obs_key: s})["emb"]               # (B, win, D)
        act_emb = wm.action_encoder(a)                       # (B, win, A)
        # multi-step latent rollout from t=history-1
        h = args.history
        pred = z[:, h - 1]
        pred_loss = 0.0
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
            pbar.set_postfix(pred=float(pred_loss / args.n_preds), recon=float(recon_loss))

    wm.eval()
    config = {
        "_target_": "stable_worldmodel.wm.statewm.StateWM",
        "state_dim": state_dim, "action_dim": action_dim, "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim, "obs_key": args.obs_key,
        "goal_key": f"goal_{args.obs_key}", "state_scale": args.state_scale,
    }
    save_pretrained(wm.cpu(), run_name=args.run_name, config=config)
    logging.success(f"StateWM saved as '{args.run_name}'")


if __name__ == "__main__":
    main()
