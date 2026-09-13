"""PushT physics grounding: geometry against pymunk, penalty semantics, calibration."""

from __future__ import annotations

import numpy as np
import pymunk
import torch

from rlp.core.grounding import (
    ACTION_SCALE,
    AGENT_RADIUS,
    GroundingPenalty,
    calibrate_pusht_grounding,
    fit_agent_kinematics,
    tee_distance,
)


def _pymunk_tee(position: tuple[float, float], angle: float) -> list[pymunk.Poly]:
    """The T exactly as stable_worldmodel's PushT.add_tee builds it."""
    scale, length, mass = 30, 4, 1
    v1 = [(-length * scale / 2, scale), (length * scale / 2, scale), (length * scale / 2, 0), (-length * scale / 2, 0)]
    v2 = [(-scale / 2, scale), (-scale / 2, length * scale), (scale / 2, length * scale), (scale / 2, scale)]
    body = pymunk.Body(mass, pymunk.moment_for_poly(mass, vertices=v1) * 2)
    shapes = [pymunk.Poly(body, v1), pymunk.Poly(body, v2)]
    body.position = position
    body.angle = angle
    space = pymunk.Space()
    space.add(body, *shapes)
    space.step(1e-9)  # refresh cached world vertices
    return shapes


def test_tee_distance_matches_pymunk() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        pos = tuple(rng.uniform(100, 400, size=2))
        angle = float(rng.uniform(0, 2 * np.pi))
        shapes = _pymunk_tee(pos, angle)
        pts = rng.uniform(pos[0] - 200, pos[0] + 200, size=(300, 2))
        pts[:, 1] = rng.uniform(pos[1] - 200, pos[1] + 200, size=300)
        ref = np.array([min(s.point_query(tuple(p)).distance for s in shapes) for p in pts])
        ours = tee_distance(
            torch.as_tensor(pts, dtype=torch.float64),
            torch.tensor(pos, dtype=torch.float64),
            torch.tensor([np.cos(angle), np.sin(angle)], dtype=torch.float64),
        ).numpy()
        outside = ref > 1.0
        assert np.allclose(ours[outside], ref[outside], atol=1e-3), "outside distances disagree with pymunk"
        assert (ours[ref < -1.0] < 1e-3).all(), "points pymunk puts inside the T must read ~0"


def test_tee_distance_is_differentiable_and_rotation_covariant() -> None:
    p = torch.tensor([[80.0, 10.0]], requires_grad=True)
    d = tee_distance(p, torch.zeros(1, 2), torch.tensor([[1.0, 0.0]]))
    assert torch.isclose(d[0], torch.tensor(20.0), atol=1e-3)  # 20 px right of the crossbar end
    d.sum().backward()  # type: ignore[no-untyped-call]  # PyTorch stub is untyped.
    assert p.grad is not None and torch.isclose(p.grad[0, 0], torch.tensor(1.0), atol=1e-3)
    # unnormalised (cos, sin) is accepted; rotating point and block together leaves the distance alone
    theta = 0.7
    rot = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=torch.float32)
    q = torch.tensor([[80.0, 10.0]]) @ rot.T
    d_rot = tee_distance(
        q, torch.zeros(1, 2), 3.0 * torch.tensor([[np.cos(theta), np.sin(theta)]], dtype=torch.float32)
    )
    assert torch.isclose(d_rot[0], torch.tensor(20.0), atol=1e-3)


def _identity_probe(dim: int = 8) -> torch.Tensor:
    """Latent = [agent x, agent y, block x, block y, cos, sin, pad...]; the probe reads it back exactly."""
    w = torch.zeros(dim + 1, 6)
    w[:6, :6] = torch.eye(6)
    return w


def _module(deadzone: float = 5.0) -> GroundingPenalty:
    return GroundingPenalty(
        _identity_probe(),
        amu=torch.zeros(2),
        astd=torch.ones(2),
        gains=torch.tensor([1.0, 0.0]),
        margin=5.0,
        tau=1.0,
        deadzone=deadzone,
        ref=10.0,
        weight=2.0,
    )


def _latent(agent: tuple[float, float], block: tuple[float, float], angle: float = 0.0, dim: int = 8) -> torch.Tensor:
    z = torch.zeros(dim)
    z[:6] = torch.tensor([*agent, *block, np.cos(angle), np.sin(angle)])
    return z


def test_penalty_is_zero_for_a_static_block() -> None:
    g = _module()
    z0 = _latent((100.0, 100.0), (300.0, 300.0)).unsqueeze(0)
    traj = z0.unsqueeze(1).expand(1, 5, -1).clone()
    plan = torch.zeros(1, 5, 10)
    pen = g(z0, traj, plan, torch.tensor([[100.0, 100.0]]), torch.zeros(1, 2))
    assert float(pen) == 0.0


def test_unsupported_motion_is_penalised_and_differentiable() -> None:
    g = _module()
    z0 = _latent((100.0, 100.0), (300.0, 300.0)).unsqueeze(0)
    # the block drifts 40 px per block while the agent (far away, plan = 0) never approaches
    traj = torch.stack([_latent((100.0, 100.0), (300.0 + 40.0 * k, 300.0)) for k in range(1, 6)]).unsqueeze(0)
    plan = torch.zeros(1, 5, 10, requires_grad=True)
    terms = g.terms(z0, traj, plan, torch.tensor([[100.0, 100.0]]), torch.zeros(1, 2))
    assert terms["contact"].max() < 1e-3
    assert torch.allclose(terms["unsupported"], torch.ones(1, 5), atol=1e-3)
    expected = sum(((40.0 * k - 5.0) / 10.0) ** 2 for k in range(1, 6))
    assert torch.isclose(terms["penalty"][0], torch.tensor(expected), rtol=1e-4)
    pen = g(z0, traj, plan, torch.tensor([[100.0, 100.0]]), torch.zeros(1, 2))
    assert torch.isclose(pen[0], torch.tensor(2.0 * expected), rtol=1e-4)
    pen.sum().backward()
    assert plan.grad is not None and torch.isfinite(plan.grad).all()


def test_near_the_block_the_gradient_points_at_it() -> None:
    g = _module()
    # agent 8 px short of touching the crossbar's left end (x = 240): the contact sigmoid is not saturated
    agent = (217.0, 315.0)
    z0 = _latent(agent, (300.0, 300.0)).unsqueeze(0)
    traj = torch.stack([_latent(agent, (300.0 + 40.0 * k, 300.0)) for k in range(1, 6)]).unsqueeze(0)
    plan = torch.zeros(1, 5, 10, requires_grad=True)
    terms = g.terms(z0, traj, plan, torch.tensor([agent]), torch.zeros(1, 2))
    assert 0.0 < float(terms["contact"][0, 0]) < 0.5
    g(z0, traj, plan, torch.tensor([agent]), torch.zeros(1, 2)).sum().backward()
    assert plan.grad is not None
    assert plan.grad[0, 0, 0] < 0  # a +x command closes the gap, legitimises the motion, lowers the penalty
    assert plan.grad[0, 0, 1] == 0  # moving along the bar edge changes nothing to first order


def test_contact_before_motion_lifts_the_penalty() -> None:
    g = _module()
    # agent starts just left of the crossbar's left end; block origin at (300, 300), angle 0
    z0 = _latent((225.0, 315.0), (300.0, 300.0)).unsqueeze(0)
    traj = torch.stack([_latent((240.0, 315.0), (300.0 + 40.0 * k, 300.0)) for k in range(1, 6)]).unsqueeze(0)
    plan = torch.zeros(1, 5, 10)
    plan[0, 0, 0] = 0.25  # first primitive step: +25 px in x -> agent centre at x=250, touching the bar end at x=240
    terms = g.terms(z0, traj, plan, torch.tensor([[225.0, 315.0]]), torch.zeros(1, 2))
    assert terms["contact"][0, 0] > 0.99
    assert terms["penalty"][0] < 1e-3


def test_export_round_trip() -> None:
    g = _module(deadzone=7.5)
    h = GroundingPenalty.from_export(g.export())
    assert h.deadzone == 7.5 and h.weight == 2.0 and torch.equal(h.probe_w, g.probe_w)
    z0 = _latent((100.0, 100.0), (300.0, 300.0)).unsqueeze(0)
    traj = torch.stack([_latent((100.0, 100.0), (300.0 + 40.0 * k, 300.0)) for k in range(1, 6)]).unsqueeze(0)
    plan = torch.zeros(1, 5, 10)
    args = (z0, traj, plan, torch.tensor([[100.0, 100.0]]), torch.zeros(1, 2))
    assert torch.equal(g(*args), h(*args))


def test_kinematics_fit_recovers_two_tap_gains() -> None:
    rng = np.random.default_rng(1)
    n, epi = 4000, np.repeat(np.arange(40), 100)
    act = rng.uniform(-1, 1, size=(n, 2))
    u = act * ACTION_SCALE
    pos = np.zeros((n, 2))
    for t in range(1, n):
        if epi[t] != epi[t - 1]:
            pos[t] = rng.uniform(0, 512, size=2)
            continue
        prev = u[t - 2] if t >= 2 and epi[t - 2] == epi[t - 1] else 0.0
        pos[t] = pos[t - 1] + 0.6 * u[t - 1] + 0.3 * prev
    gains, r2 = fit_agent_kinematics(pos, act, epi)
    assert torch.allclose(gains, torch.tensor([0.6, 0.3]), atol=1e-6)
    assert r2 > 0.999


def _synthetic_dataset(seed: int = 0) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Agent random-walks near a block that moves only when the disc touches the T."""
    rng = np.random.default_rng(seed)
    episodes, length = 60, 120
    n = episodes * length
    state = np.zeros((n, 5))
    act = np.zeros((n, 2))
    epi = np.repeat(np.arange(episodes), length)
    for e in range(episodes):
        b = rng.uniform(150, 350, size=2)
        ang = rng.uniform(0, 2 * np.pi)
        p = b + rng.uniform(-120, 120, size=2)
        for t in range(length):
            i = e * length + t
            a = rng.uniform(-0.6, 0.6, size=2)
            act[i] = a
            state[i] = [*p, *b, ang]
            p_next = p + a * ACTION_SCALE
            gap = (
                float(
                    tee_distance(torch.tensor(p_next), torch.tensor(b), torch.tensor([np.cos(ang), np.sin(ang)])).item()
                )
                - AGENT_RADIUS
            )
            if gap <= 0:
                b = b + 0.4 * a * ACTION_SCALE  # pushed
            p = p_next
    targets = np.column_stack([state[:, :4], np.cos(state[:, 4]), np.sin(state[:, 4])])
    # unit-scale latent features (like a real encoder's) mixed by a random rotation, plus mild noise
    feats = (targets - targets.mean(0)) / targets.std(0)
    mix = np.linalg.qr(rng.normal(size=(10, 10)))[0]
    z = torch.as_tensor(np.column_stack([feats, rng.normal(size=(n, 4))]) @ mix, dtype=torch.float32)
    z = z + 0.01 * torch.randn_like(z)
    return z, state, act, epi


def test_calibration_end_to_end() -> None:
    z, state, act, epi = _synthetic_dataset()
    module, report = calibrate_pusht_grounding(
        z, state, act, epi, amu=np.zeros(2), astd=np.ones(2), n_windows=2000, n_check=5000, seed=0
    )
    assert report["probe_r2_block"] > 0.99 and report["probe_r2_angle"] > 0.99 and report["kin_r2"] > 0.99
    assert report["moved_given_clear"] < 0.05  # the geometry explains the motion
    assert report["moved_given_contact"] > 0.5
    assert 0.0 < report["deadzone"] < 5.0  # tiny probe noise -> tiny dead zone
    assert module.deadzone == report["deadzone"] and module.margin == report["margin"]
    assert report["push_recognised_frac"] >= 0.95  # real pushes register as contact
    assert report["margin"] < 15.0  # exact probe: margin is just the 2.2-tau safety band
    assert report["decoded_disp_push_end"] > report["decoded_disp_static_end"]
