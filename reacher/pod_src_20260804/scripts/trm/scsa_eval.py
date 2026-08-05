"""Same-Candidate Selection Audit (SCSA) for a pixel WM — the paper's diagnostic.

Decouples the *metric* from the planner: at each hard planning state we score one
fixed candidate set with each terminal cost and compare its ranking to the
ground-truth task-state oracle (Spearman / best-rank-percentile). Two variants:

  * ``predicted`` — score the WM's rolled-out terminal ``ẑ_{t+H}`` (what the planner
    actually ranks). Low Spearman here with high `true` Spearman ⇒ the encoder→
    predictor distribution gap (rollout drift) is the bottleneck, not the metric.
  * ``true`` — score the *encoded true terminal* (analytic dynamics → render →
    encode). Isolates the metric's quality on on-manifold latents.
"""

import argparse

import numpy as np
import torch
from loguru import logger as logging
from tabulate import tabulate

import stable_worldmodel as swm
from stable_worldmodel.trm import MetricCost, diagnostics, load_metric
from stable_worldmodel.trm.oracle import TwoRoomOracleCost, rollout_dynamics

from _common import load_wm, pick_device
from eval_hard import build_hard_manifest, read_tworoom_geometry
from online_td import ObsCodec

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def render(world, pos):
    return np.stack([
        world.envs.envs[i].unwrapped._render_frame(
            agent_pos=torch.as_tensor(pos[i], dtype=torch.float32)).cpu().numpy().transpose(1, 2, 0)
        for i in range(len(pos))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--condition", action="append", default=[], dest="conditions")  # name=path or 'latent'
    p.add_argument("--n-states", type=int, default=40)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--min-geo", type=float, default=90.0)
    p.add_argument("--max-geo", type=float, default=180.0)
    p.add_argument("--variant", choices=["predicted", "true", "both"], default="both")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    codec = ObsCodec(wm, device)
    dataset = swm.data.load_dataset(args.dataset)
    n = args.n_states
    world = swm.World("swm/TwoRoom-v1", num_envs=n, max_episode_steps=200,
                      image_shape=(224, 224), render_mode="rgb_array")
    geom = read_tworoom_geometry(world)
    oracle = TwoRoomOracleCost(**geom)

    starts, goals, cls = build_hard_manifest(dataset, n // 2, geom["door_points"], args.seed,
                                             args.min_geo, args.max_geo)
    n = len(starts)
    rng = np.random.default_rng(args.seed)
    S, H, A = args.num_samples, args.horizon, args.action_block * 2
    cand = torch.as_tensor(rng.standard_normal((n, S, H, A)) * args.var_scale, dtype=torch.float32)

    tf = codec.tf  # ImageNet transform
    start_imgs = render(world, starts)
    goal_imgs = render(world, goals)
    def norm(imgs):  # (n,H,W,C)->(n,C,H,W)
        return torch.stack([tf(f) for f in imgs])
    pix = norm(start_imgs).unsqueeze(1).expand(n, S, 3, 224, 224).unsqueeze(2)   # (n,S,1,C,H,W)
    gpix = norm(goal_imgs).unsqueeze(1).expand(n, S, 3, 224, 224).unsqueeze(2)
    st = torch.as_tensor(starts, dtype=torch.float32).view(n, 1, 1, 2).expand(n, S, 1, 2)
    gst = torch.as_tensor(goals, dtype=torch.float32).view(n, 1, 1, 2).expand(n, S, 1, 2)

    def base_info():
        return {"pixels": pix.clone().to(device), "goal": gpix.clone().to(device),
                "state": st.clone(), "goal_state": gst.clone(),
                "action": torch.zeros(n, S, 1, 2)}

    # oracle per-candidate cost (true reachability)
    ocost = oracle.get_cost(base_info(), cand).cpu().numpy()  # (n,S)

    # --- TRUE terminals: analytic dynamics -> render -> encode (on-manifold) ---
    z_true = z_goal = None
    if args.variant in ("true", "both"):
        raw = 2
        acts_env = cand.reshape(n * S, H * (A // raw), raw)  # unpack action-blocks
        pos0 = torch.as_tensor(starts, dtype=torch.float32).repeat_interleave(S, 0)
        dpts = torch.as_tensor(geom["door_points"], dtype=torch.float32)
        term_pos = rollout_dynamics(pos0, acts_env, geom["speed"], dpts,
                                    geom["door_half"], geom["wall_thickness"], geom["agent_radius"]).numpy()
        env0 = world.envs.envs[0].unwrapped
        timgs = np.stack([env0._render_frame(agent_pos=torch.as_tensor(p, dtype=torch.float32)).cpu().numpy().transpose(1, 2, 0) for p in term_pos])
        with torch.no_grad():
            z_true = []
            for i in range(0, len(timgs), 512):
                batch = torch.stack([tf(f) for f in timgs[i:i + 512]]).unsqueeze(1).to(device)
                z_true.append(wm.encode({"pixels": batch})["emb"][:, 0].cpu())
            z_true = torch.cat(z_true).reshape(n, S, -1)
            gbatch = torch.stack([tf(f) for f in goal_imgs]).unsqueeze(1).to(device)
            z_goal = wm.encode({"pixels": gbatch})["emb"][:, 0].cpu()  # (n,D)

    rows = []
    for spec in args.conditions:
        name, path = (spec.split("=", 1) + [None])[:2] if "=" in spec else (spec, None)
        mode = "latent" if name == "latent" else "replacement"
        metric = None if mode == "latent" else load_metric(path, device="cpu")
        out = [name]
        if args.variant in ("predicted", "both"):
            mc = MetricCost(wm, metric.to(device) if metric is not None else None, mode=mode)
            mcost = mc.get_cost(base_info(), cand.to(device)).cpu().numpy()
            agg = diagnostics.scsa_aggregate([diagnostics.scsa_pointwise(mcost[b], ocost[b]) for b in range(n)])
            out += [f"{agg['spearman']:.3f}", f"{agg['best_rank_pct']:.3f}"]
            logging.success(f"[{name}] PREDICTED spearman={agg['spearman']:.3f} best_rank_pct={agg['best_rank_pct']:.3f}")
        if args.variant in ("true", "both"):
            if mode == "latent":
                tc = ((z_true - z_goal.unsqueeze(1)) ** 2).sum(-1).numpy()
            else:
                tc = metric.cpu().cost(z_true, z_goal.unsqueeze(1).expand_as(z_true)).numpy()
            agg = diagnostics.scsa_aggregate([diagnostics.scsa_pointwise(tc[b], ocost[b]) for b in range(n)])
            out += [f"{agg['spearman']:.3f}", f"{agg['best_rank_pct']:.3f}"]
            logging.success(f"[{name}] TRUE      spearman={agg['spearman']:.3f} best_rank_pct={agg['best_rank_pct']:.3f}")
        rows.append(out)

    hdr = ["condition"]
    if args.variant in ("predicted", "both"): hdr += ["pred_spear", "pred_rank%"]
    if args.variant in ("true", "both"): hdr += ["true_spear", "true_rank%"]
    print(tabulate(rows, headers=hdr, tablefmt="github"))


if __name__ == "__main__":
    main()
