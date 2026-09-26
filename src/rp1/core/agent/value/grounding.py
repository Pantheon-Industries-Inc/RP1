"""Physics grounding of the planner's energy on PushT.

A frozen world model smooths the contact discontinuity into a ramp, so a
gradient planner can imagine block motion the agent never causes: on PushT a
failed first plan imagined 51 px of block displacement, against 5 px for the
sampling planner on the same model. The grounding term makes that fiction
expensive with information the model does not have:

* a fixed linear probe reads agent position, block position and block angle
  out of every imagined latent (fitted by ridge regression on the offline
  cache, held-out R2 ~0.95-0.97);
* the agent's commanded path is reconstructed from the plan with the
  environment's own kinematics (relative position control, a two-tap gain
  fitted on the dataset), starting from the real agent position;
* imagined block displacement from the plan's start is penalised in every
  block before the first one in which that path comes within contact
  distance of the decoded T.

The penalty is a linear read of latents already computed plus a few dozen
scalar ops: no samples, no second model, and it is identical at training and
deploy (the probe and the calibrated thresholds travel in the planner
checkpoint). Imagined rotation without contact is not penalised; the
fabricated motion was translation.

Geometry follows ``stable_worldmodel.envs.pusht.PushT``: a 512-px world, an
agent disc of radius 15, and a T of two rectangles in the block body frame
(``add_tee``: crossbar 120 x 30 above the body origin, stem 30 x 90 below).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = [
    "ACTION_SCALE",
    "AGENT_RADIUS",
    "GroundingPenalty",
    "calibrate_pusht_grounding",
    "decode_state",
    "fit_agent_kinematics",
    "fit_state_probe",
    "tee_distance",
]

TEE_SCALE = 30.0
TEE_LENGTH = 4.0
# rectangles in the block body frame: (x_min, x_max, y_min, y_max)
TEE_RECTS: tuple[tuple[float, float, float, float], ...] = (
    (-TEE_LENGTH * TEE_SCALE / 2, TEE_LENGTH * TEE_SCALE / 2, 0.0, TEE_SCALE),  # crossbar
    (-TEE_SCALE / 2, TEE_SCALE / 2, TEE_SCALE, TEE_LENGTH * TEE_SCALE),  # stem
)
AGENT_RADIUS = 15.0
ACTION_SCALE = 100.0  # env action in [-1, 1] -> commanded agent displacement in px
STATE_DIM = 6  # decoded: agent x, agent y, block x, block y, cos(angle), sin(angle)


def tee_distance(points: torch.Tensor, block_xy: torch.Tensor, angle_cs: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from ``points`` to the T's surface (0 inside).

    ``points`` (..., 2) in world px; ``block_xy`` (..., 2) body origin; ``angle_cs``
    (..., 2) = (cos, sin) of the body angle, normalised internally so a raw probe
    read can be passed in. Shapes broadcast. Differentiable almost everywhere.
    """
    cs = angle_cs / angle_cs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    c, s = cs[..., 0], cs[..., 1]
    d = points - block_xy
    # world -> body frame: rotate by -angle (pymunk rotates local -> world by +angle)
    lx = d[..., 0] * c + d[..., 1] * s
    ly = -d[..., 0] * s + d[..., 1] * c
    dist: torch.Tensor | None = None
    for x0, x1, y0, y1 in TEE_RECTS:
        qx = ((lx - (x0 + x1) / 2).abs() - (x1 - x0) / 2).clamp_min(0.0)
        qy = ((ly - (y0 + y1) / 2).abs() - (y1 - y0) / 2).clamp_min(0.0)
        r = torch.sqrt(qx * qx + qy * qy + 1e-9)
        dist = r if dist is None else torch.minimum(dist, r)
    if dist is None:  # pragma: no cover - TEE_RECTS is a non-empty constant
        raise RuntimeError("no rectangles")
    return dist


def _state_targets(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64)
    return np.column_stack([s[:, 0], s[:, 1], s[:, 2], s[:, 3], np.cos(s[:, 4]), np.sin(s[:, 4])])


def fit_state_probe(
    z: torch.Tensor,
    state: np.ndarray,
    *,
    ridge: float = 1.0,
    n_fit: int = 200_000,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Ridge probe latent -> (agent xy, block xy, cos, sin), with a held-out report.

    ``z`` (N, D) latents (may be memory-mapped), ``state`` (N, >=5) the env
    state rows aligned with ``z``. Returns the (D+1, 6) weight (bias last) and
    held-out R2 / median pixel errors.
    """
    n = int(z.shape[0])
    if state.shape[0] != n:
        raise ValueError(f"state rows ({state.shape[0]}) do not align with latents ({n})")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=min(n_fit, n), replace=False))
    x = z[torch.as_tensor(idx)].detach().cpu().double().numpy()
    y = _state_targets(np.asarray(state)[idx])
    ntr = max(int(0.9 * len(x)), 1)
    xb = np.column_stack([x, np.ones(len(x))])
    gram = xb[:ntr].T @ xb[:ntr] + ridge * np.eye(xb.shape[1])
    w = np.linalg.solve(gram, xb[:ntr].T @ y[:ntr])
    pred = xb[ntr:] @ w
    resid = ((pred - y[ntr:]) ** 2).sum(0)
    total = ((y[ntr:] - y[ntr:].mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - resid / total
    err_agent = np.linalg.norm(pred[:, :2] - y[ntr:, :2], axis=1)
    err_block = np.linalg.norm(pred[:, 2:4] - y[ntr:, 2:4], axis=1)
    report = {
        "probe_r2_agent": float(r2[:2].mean()),
        "probe_r2_block": float(r2[2:4].mean()),
        "probe_r2_angle": float(r2[4:6].mean()),
        "probe_agent_px": float(np.median(err_agent)),
        "probe_block_px": float(np.median(err_block)),
    }
    return torch.as_tensor(w, dtype=torch.float32), report


def fit_agent_kinematics(
    agent_xy: np.ndarray,
    action_raw: np.ndarray,
    episode_idx: np.ndarray,
    *,
    n_fit: int = 500_000,
    seed: int = 0,
) -> tuple[torch.Tensor, float]:
    """Two-tap gain of the agent's position controller: dp_t = g0 u_t + g1 u_{t-1}.

    ``u`` is the commanded displacement (raw env action x ACTION_SCALE); the PD
    controller does not reach its target inside one control step, so part of
    a command lands in the next step. Returns ``(g0, g1)`` and the fit's R2.
    """
    epi = np.asarray(episode_idx).reshape(-1)
    t = np.flatnonzero(epi[1:] == epi[:-1])  # t and t+1 in the same episode
    rng = np.random.default_rng(seed)
    if len(t) > n_fit:
        t = np.sort(rng.choice(t, size=n_fit, replace=False))
    p = np.asarray(agent_xy, dtype=np.float64)
    u_all = np.nan_to_num(np.asarray(action_raw, dtype=np.float64)) * ACTION_SCALE
    dp = p[t + 1] - p[t]
    u = u_all[t]
    prev_ok = (t >= 1) & (epi[np.maximum(t - 1, 0)] == epi[t])
    u_prev = np.where(prev_ok[:, None], u_all[np.maximum(t - 1, 0)], 0.0)
    f = np.stack([u, u_prev], axis=-1).reshape(-1, 2)  # each coordinate is a sample
    y = dp.reshape(-1)
    g, *_ = np.linalg.lstsq(f, y, rcond=None)
    resid = float(((f @ g - y) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return torch.as_tensor(g, dtype=torch.float32), 1.0 - resid / total


def decode_state(z: torch.Tensor, probe_w: torch.Tensor) -> torch.Tensor:
    """Latent (..., D) -> (..., 6) state read by a probe from :func:`fit_state_probe`."""
    ones = z.new_ones(*z.shape[:-1], 1)
    return torch.cat([z, ones], dim=-1) @ probe_w


def _kinematic_path(agent0: torch.Tensor, u: torch.Tensor, u_prev: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """Agent position after each commanded displacement: (B, T, 2) from (B, 2), (B, T, 2), (B, 2)."""
    prev = torch.cat([u_prev.unsqueeze(1), u[:, :-1]], dim=1)
    step = gains[0] * u + gains[1] * prev
    return agent0.unsqueeze(1) + step.cumsum(dim=1)


class GroundingPenalty(nn.Module):
    """Unsupported imagined block displacement, weighted into the planning energy.

    ``forward(z0, traj, plan, agent0, u_prev)``: ``z0`` (B, D) the real start
    latent, ``traj`` (B, H, D) the imagined latents at block ends, ``plan``
    (B, H, fs * act_dim) the plan in the actor's normalised action space,
    ``agent0`` (B, 2) the real agent position in px, ``u_prev`` (B, act_dim)
    the commanded displacement (px) of the primitive step before the plan.
    Returns ``weight * penalty`` per sample.
    """

    def __init__(
        self,
        probe_w: torch.Tensor,
        amu: torch.Tensor,
        astd: torch.Tensor,
        gains: torch.Tensor,
        *,
        margin: float,
        tau: float,
        deadzone: float,
        ref: float,
        weight: float,
    ) -> None:
        super().__init__()
        if probe_w.shape[-1] != STATE_DIM:
            raise ValueError(f"probe must decode {STATE_DIM} state channels, got {probe_w.shape[-1]}")
        self.probe_w: torch.Tensor
        self.amu: torch.Tensor
        self.astd: torch.Tensor
        self.gains: torch.Tensor
        self.register_buffer("probe_w", probe_w.detach().float().clone())
        self.register_buffer("amu", amu.detach().float().clone().reshape(-1))
        self.register_buffer("astd", astd.detach().float().clone().reshape(-1))
        self.register_buffer("gains", gains.detach().float().clone().reshape(2))
        self.margin = float(margin)
        self.tau = float(tau)
        self.deadzone = float(deadzone)
        self.ref = float(ref)
        self.weight = float(weight)

    @property
    def act_dim(self) -> int:
        return int(self.amu.numel())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent (..., D) -> (..., 6) decoded state."""
        return decode_state(z, self.probe_w)

    def commands(self, plan: torch.Tensor) -> torch.Tensor:
        """Normalised plan (B, H, fs*act_dim) -> commanded displacements (B, H*fs, act_dim) in px."""
        b, h, a = plan.shape
        raw = plan.reshape(b, h * (a // self.act_dim), self.act_dim) * self.astd + self.amu
        return raw * ACTION_SCALE

    def agent_path(self, plan: torch.Tensor, agent0: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        """Agent position after every primitive step of the plan, (B, H*fs, 2)."""
        return _kinematic_path(agent0, self.commands(plan), u_prev, self.gains)

    def terms(
        self,
        z0: torch.Tensor,
        traj: torch.Tensor,
        plan: torch.Tensor,
        agent0: torch.Tensor,
        u_prev: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, h, _ = traj.shape
        s0 = self.decode(z0)  # (B, 6)
        st = self.decode(traj)  # (B, H, 6)
        displacement = (st[..., 2:4] - s0[:, None, 2:4]).norm(dim=-1)  # (B, H) from the plan's start
        start = torch.cat([s0.unsqueeze(1), st[:, :-1]], dim=1)  # block pose at each block's start
        path = self.agent_path(plan, agent0, u_prev)
        fs = path.shape[1] // h
        pts = path.view(b, h, fs, 2)
        gap = tee_distance(pts, start[:, :, None, 2:4], start[:, :, None, 4:6]) - AGENT_RADIUS
        gap_min = gap.amin(dim=2)  # (B, H) closest approach within the block
        contact = torch.sigmoid((self.margin - gap_min) / self.tau)  # soft "touched during block k"
        unsupported = torch.cumprod(1.0 - contact, dim=1)  # no contact through block k
        excess = (displacement - self.deadzone).clamp_min(0.0) / self.ref
        penalty = (unsupported * excess.pow(2)).sum(dim=1)
        return {
            "penalty": penalty,
            "displacement": displacement,
            "contact": contact,
            "unsupported": unsupported,
            "gap": gap_min,
        }

    def forward(
        self,
        z0: torch.Tensor,
        traj: torch.Tensor,
        plan: torch.Tensor,
        agent0: torch.Tensor,
        u_prev: torch.Tensor,
    ) -> torch.Tensor:
        return self.weight * self.terms(z0, traj, plan, agent0, u_prev)["penalty"]

    def export(self) -> dict[str, Any]:
        return {
            "kind": "pusht",
            "probe_w": self.probe_w.detach().cpu(),
            "amu": self.amu.detach().cpu(),
            "astd": self.astd.detach().cpu(),
            "gains": self.gains.detach().cpu(),
            "margin": self.margin,
            "tau": self.tau,
            "deadzone": self.deadzone,
            "ref": self.ref,
            "weight": self.weight,
        }

    @classmethod
    def from_export(cls, payload: dict[str, Any]) -> GroundingPenalty:
        if payload.get("kind") != "pusht":
            raise ValueError(f"unsupported grounding kind {payload.get('kind')!r}")
        return cls(
            payload["probe_w"],
            payload["amu"],
            payload["astd"],
            payload["gains"],
            margin=float(payload["margin"]),
            tau=float(payload["tau"]),
            deadzone=float(payload["deadzone"]),
            ref=float(payload["ref"]),
            weight=float(payload["weight"]),
        )


def calibrate_pusht_grounding(
    z: torch.Tensor,
    state: np.ndarray,
    action_raw: np.ndarray,
    episode_idx: np.ndarray,
    *,
    amu: np.ndarray,
    astd: np.ndarray,
    frameskip: int,
    horizon: int,
    margin: float | None,
    tau: float,
    ref: float,
    weight: float,
    deadzone: float | None,
    quantile: float,
    n_windows: int = 50_000,
    n_check: int = 300_000,
    seed: int = 0,
) -> tuple[GroundingPenalty, dict[str, float]]:
    """Fit the probe and kinematics, check the geometry, calibrate the thresholds.

    All inputs are row-aligned over the offline data: ``z`` (N, D) latents,
    ``state`` (N, >=5) env states, ``action_raw`` (N, act_dim) raw env actions,
    ``episode_idx`` (N,). ``amu``/``astd`` are the planner's action statistics.

    Checks, all reported: the probe's held-out accuracy; the kinematic fit's
    R2; and, on TRUE states, how often the block moves in a step where the
    agent stays more than 10 px clear of the T (must be rare, or the geometry
    is wrong -- raises above 40%).

    Calibrated thresholds (both overridable):

    * ``margin`` -- the contact test must accept real pushes. On windows whose
      true block displacement exceeds 30 px, the closest approach of the
      kinematic agent path to the DECODED T (the quantity the penalty sees) is
      measured; the margin is its ``quantile`` plus 2.2 tau, so that share of
      real pushes registers contact >= 0.9 despite probe noise.
    * ``deadzone`` -- the ``quantile`` of decoded block displacement over
      ``horizon`` blocks on windows where the true block is static: the
      probe's own noise floor.

    Reported power: among static-block windows where the true agent hovers
    within 60 px of the T without touching it, the fraction the calibrated
    test still flags as no-contact.
    """
    epi = np.asarray(episode_idx).reshape(-1)
    n = int(z.shape[0])
    if not (state.shape[0] == action_raw.shape[0] == len(epi) == n):
        raise ValueError("grounding calibration inputs are not row-aligned")
    st = np.asarray(state, dtype=np.float64)
    act = np.nan_to_num(np.asarray(action_raw, dtype=np.float64))
    rng = np.random.default_rng(seed)
    check_clear = 10.0

    probe_w, report = fit_state_probe(z, st, seed=seed)
    gains, kin_r2 = fit_agent_kinematics(st[:, :2], act, epi, seed=seed)
    report["kin_g0"], report["kin_g1"], report["kin_r2"] = float(gains[0]), float(gains[1]), float(kin_r2)

    # geometry check on true states: contact (agent disc touching the T at step t or t+1) vs block motion
    agent = torch.as_tensor(st[:, :2], dtype=torch.float32)
    block = torch.as_tensor(st[:, 2:4], dtype=torch.float32)
    cs = torch.as_tensor(np.column_stack([np.cos(st[:, 4]), np.sin(st[:, 4])]), dtype=torch.float32)
    with torch.no_grad():
        gap_all = (tee_distance(agent, block, cs) - AGENT_RADIUS).numpy()
    same = np.flatnonzero(epi[1:] == epi[:-1])
    chk = np.sort(rng.choice(same, size=min(n_check, len(same)), replace=False))
    gap_step = np.minimum(gap_all[chk], gap_all[chk + 1])
    moved = np.linalg.norm(st[chk + 1, 2:4] - st[chk, 2:4], axis=1) > 2.0
    touching, clear = gap_step <= 0.0, gap_step > check_clear
    report["true_contact_step_frac"] = float(touching.mean())
    report["moved_given_contact"] = float(moved[touching].mean()) if touching.any() else float("nan")
    report["moved_given_clear"] = float(moved[clear].mean()) if clear.any() else float("nan")
    if report["moved_given_clear"] > 0.4:
        raise RuntimeError(
            f"T geometry does not explain block motion: the block moves in {report['moved_given_clear']:.0%} "
            f"of steps with the agent more than {check_clear} px clear of it"
        )

    # windows of `horizon` blocks: true motion, true closest approach, decoded motion, decoded contact
    fs = frameskip
    span = horizon * fs
    starts = np.flatnonzero(epi[span:] == epi[:-span])
    starts = np.sort(rng.choice(starts, size=min(n_windows, len(starts)), replace=False))
    w = len(starts)
    ends = starts[:, None] + fs * np.arange(1, horizon + 1)[None, :]  # (W, H)
    true_disp = np.linalg.norm(st[ends, 2:4] - st[starts, None, 2:4], axis=-1)  # (W, H)
    steps = starts[:, None] + np.arange(span + 1)[None, :]
    gap_window_true = gap_all[steps].min(axis=1)
    static = true_disp.max(axis=1) < 3.0
    push = true_disp[:, -1] > 30.0
    with torch.no_grad():
        dec_start = decode_state(z[torch.as_tensor(starts)].float(), probe_w)  # (W, 6)
        dec_end = decode_state(z[torch.as_tensor(ends.reshape(-1))].float(), probe_w).view(w, horizon, 6)
        decoded_disp = (dec_end[..., 2:4] - dec_start[:, None, 2:4]).norm(dim=-1).numpy()  # (W, H)
        # the kinematic agent path the penalty will see, from the true start position and true actions
        u = torch.as_tensor(act[starts[:, None] + np.arange(span)[None, :]] * ACTION_SCALE, dtype=torch.float32)
        prev_ok = (starts >= 1) & (epi[np.maximum(starts - 1, 0)] == epi[starts])
        u_prev = torch.as_tensor(
            np.where(prev_ok[:, None], act[np.maximum(starts - 1, 0)] * ACTION_SCALE, 0.0), dtype=torch.float32
        )
        path = _kinematic_path(torch.as_tensor(st[starts, :2], dtype=torch.float32), u, u_prev, gains)
        gap_dec = (tee_distance(path, dec_start[:, None, 2:4], dec_start[:, None, 4:6]) - AGENT_RADIUS).amin(dim=1)
        gap_dec_np = gap_dec.numpy()
    report["static_windows"] = float(static.sum())
    report["push_windows"] = float(push.sum())
    if margin is None:
        if push.sum() < 100:
            raise RuntimeError(f"only {int(push.sum())} push windows; cannot calibrate the contact margin")
        margin = max(0.0, float(np.quantile(gap_dec_np[push], quantile))) + 2.2 * tau
    contact_dec = torch.sigmoid(torch.as_tensor((margin - gap_dec_np) / tau)).numpy()
    report["margin"] = float(margin)
    report["push_recognised_frac"] = float((contact_dec[push] >= 0.5).mean()) if push.any() else float("nan")
    hover = static & (gap_window_true > 0.0) & (gap_window_true < 60.0)
    report["hover_windows"] = float(hover.sum())
    report["hover_flagged_frac"] = float((contact_dec[hover] < 0.5).mean()) if hover.any() else float("nan")
    if deadzone is None:
        if static.sum() < 100:
            raise RuntimeError(f"only {int(static.sum())} static windows; cannot calibrate the dead zone")
        deadzone = float(np.quantile(decoded_disp[static], quantile))
    report["deadzone"] = float(deadzone)
    report["decoded_disp_static_end"] = float(decoded_disp[static, -1].mean()) if static.any() else float("nan")
    report["decoded_disp_push_end"] = float(decoded_disp[push, -1].mean()) if push.any() else float("nan")
    report["true_disp_push_end"] = float(true_disp[push, -1].mean()) if push.any() else float("nan")
    module = GroundingPenalty(
        probe_w,
        torch.as_tensor(amu),
        torch.as_tensor(astd),
        gains,
        margin=float(margin),
        tau=tau,
        deadzone=float(deadzone),
        ref=ref,
        weight=weight,
    )
    return module, report
