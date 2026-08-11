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

from pathlib import Path
from typing import cast

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig
from tabulate import tabulate
from torch import nn

from rlp.config import dispatch, run_hydra
from rlp.core.solver.lip import EncoderWorldModel
from rlp.core.value import MetricCost, diagnostics, load_metric
from rlp.core.value.oracle import TwoRoomOracleCost, rollout_dynamics
from rlp.core.value.protocols import ValueMetric
from rlp.core.world_model.runtime import load_wm, pick_device
from rlp.data.protocols import Array
from rlp.environment import World
from rlp.logging import log_multiline, logger
from rlp.train.online_td import ObsCodec

from .hard import build_hard_manifest
from .trm import read_tworoom_geometry

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def render(world: World, pos: Array) -> Array:
    return np.stack(
        [
            world.envs.envs[i]
            .unwrapped._render_frame(agent_pos=torch.as_tensor(pos[i], dtype=torch.float32))
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            for i in range(len(pos))
        ]
    )


def _run(cfg: DictConfig) -> None:
    args = cfg

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    if not callable(getattr(wm, "encode", None)):
        raise TypeError("SCSA requires a world model with encode()")
    encoder = cast(EncoderWorldModel, wm)
    codec = ObsCodec(wm, device)
    dataset = swm.data.load_dataset(args.dataset)
    n = args.n_states
    world = World(
        "swm/TwoRoom-v1",
        num_envs=n,
        max_episode_steps=200,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    geom = read_tworoom_geometry(world)
    oracle = TwoRoomOracleCost(**geom)

    starts, goals, cls = build_hard_manifest(
        dataset,
        n // 2,
        geom["door_points"],
        args.seed,
        args.min_geo,
        args.max_geo,
    )
    n = len(starts)
    rng = np.random.default_rng(args.seed)
    S, H, A = args.num_samples, args.horizon, args.action_block * 2
    cand = torch.as_tensor(rng.standard_normal((n, S, H, A)) * args.var_scale, dtype=torch.float32)

    tf = codec.tf  # ImageNet transform
    if tf is None:
        raise TypeError("SCSA requires a pixel world model")
    start_imgs = render(world, starts)
    goal_imgs = render(world, goals)

    def norm(imgs: Array) -> torch.Tensor:  # (n,H,W,C)->(n,C,H,W)
        return torch.stack([tf(f) for f in imgs])

    pix = norm(start_imgs).unsqueeze(1).expand(n, S, 3, 224, 224).unsqueeze(2)  # (n,S,1,C,H,W)
    gpix = norm(goal_imgs).unsqueeze(1).expand(n, S, 3, 224, 224).unsqueeze(2)
    st = torch.as_tensor(starts, dtype=torch.float32).view(n, 1, 1, 2).expand(n, S, 1, 2)
    gst = torch.as_tensor(goals, dtype=torch.float32).view(n, 1, 1, 2).expand(n, S, 1, 2)

    def base_info() -> dict[str, torch.Tensor]:
        return {
            "pixels": pix.clone().to(device),
            "goal": gpix.clone().to(device),
            "state": st.clone(),
            "goal_state": gst.clone(),
            "action": torch.zeros(n, S, 1, 2),
        }

    # oracle per-candidate cost (true reachability)
    ocost = oracle.get_cost(base_info(), cand).cpu().numpy()  # (n,S)

    # --- TRUE terminals: analytic dynamics -> render -> encode (on-manifold) ---
    z_true: torch.Tensor | None = None
    z_goal: torch.Tensor | None = None
    if args.variant in ("true", "both"):
        raw = 2
        acts_env = cand.reshape(n * S, H * (A // raw), raw)  # unpack action-blocks
        pos0 = torch.as_tensor(starts, dtype=torch.float32).repeat_interleave(S, 0)
        dpts = torch.as_tensor(geom["door_points"], dtype=torch.float32)
        term_pos = rollout_dynamics(
            pos0,
            acts_env,
            geom["speed"],
            dpts,
            geom["door_half"],
            geom["wall_thickness"],
            geom["agent_radius"],
        ).numpy()
        env0 = world.envs.envs[0].unwrapped
        timgs = np.stack(
            [
                env0._render_frame(agent_pos=torch.as_tensor(p, dtype=torch.float32)).cpu().numpy().transpose(1, 2, 0)
                for p in term_pos
            ]
        )
        with torch.no_grad():
            z_batches: list[torch.Tensor] = []
            for i in range(0, len(timgs), 512):
                batch = torch.stack([tf(f) for f in timgs[i : i + 512]]).unsqueeze(1).to(device)
                z_batches.append(encoder.encode({"pixels": batch})["emb"][:, 0].cpu())
            z_true = torch.cat(z_batches).reshape(n, S, -1)
            gbatch = torch.stack([tf(f) for f in goal_imgs]).unsqueeze(1).to(device)
            z_goal = encoder.encode({"pixels": gbatch})["emb"][:, 0].cpu()  # (n,D)

    rows: list[list[str]] = []
    for spec in args.conditions:
        name: str
        path: str | None
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = spec, None
        mode = "latent" if name == "latent" else "replacement"
        if mode == "latent":
            metric: nn.Module | None = None
        else:
            if path is None:
                raise ValueError(f"condition {spec!r} requires a metric checkpoint")
            metric = load_metric(path, device="cpu")
        out = [name]
        if args.variant in ("predicted", "both"):
            mc = MetricCost(
                wm,
                metric.to(device) if metric is not None else None,
                mode=mode,
            )
            mcost = mc.get_cost(base_info(), cand.to(device)).cpu().numpy()
            agg = diagnostics.scsa_aggregate([diagnostics.scsa_pointwise(mcost[b], ocost[b]) for b in range(n)])
            out += [f"{agg['spearman']:.3f}", f"{agg['best_rank_pct']:.3f}"]
            logger.success(
                f"[{name}] PREDICTED spearman={agg['spearman']:.3f} best_rank_pct={agg['best_rank_pct']:.3f}"
            )
        if args.variant in ("true", "both"):
            if z_true is None or z_goal is None:
                raise RuntimeError("true-terminal latents were not computed")
            if mode == "latent":
                tc = ((z_true - z_goal.unsqueeze(1)) ** 2).sum(-1).numpy()
            else:
                if metric is None:
                    raise RuntimeError("metric condition lacks a metric")
                value_metric = cast(ValueMetric, metric.cpu())
                tc = value_metric.cost(z_true, z_goal.unsqueeze(1).expand_as(z_true)).numpy()
            agg = diagnostics.scsa_aggregate([diagnostics.scsa_pointwise(tc[b], ocost[b]) for b in range(n)])
            out += [f"{agg['spearman']:.3f}", f"{agg['best_rank_pct']:.3f}"]
            logger.success(
                f"[{name}] TRUE      spearman={agg['spearman']:.3f} best_rank_pct={agg['best_rank_pct']:.3f}"
            )
        rows.append(out)

    hdr: list[str] = ["condition"]
    if args.variant in ("predicted", "both"):
        hdr += ["pred_spear", "pred_rank%"]
    if args.variant in ("true", "both"):
        hdr += ["true_spear", "true_rank%"]
    table = tabulate(rows, headers=hdr, tablefmt="github")
    log_multiline(table)
    results_path = Path(args.run.metrics) / args.output.filename
    results_path.write_text(table + "\n")
    logger.info(f"SCSA results saved to {results_path}")


def main() -> object:
    return run_hydra(dispatch, config_name="eval/scsa")


if __name__ == "__main__":
    main()
