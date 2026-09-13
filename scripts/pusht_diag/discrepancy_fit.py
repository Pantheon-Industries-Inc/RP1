"""E33: block-level discrepancy model D, trained OFFLINE, then the decisive rescoring.

D(imagined state at block start, action block) predicts the real-minus-imagined
block motion over one five-step action block. Targets come from the logged data:
every 25-step window of the training episodes is re-imagined through the frozen
world model from its real history (free-running, so blocks 2-5 start from
imagined states, as at deploy), and the residual is the decoded block pose of
the REAL frame minus the decoded pose of the IMAGINED one, in the block's body
frame (probe noise largely cancels; the residual is the latent fiction as the
probe sees it).

Features per block, all differentiable w.r.t. the plan at deploy: decoded
imagined pose at block start, the agent's position relative to the block in the
body frame (real at the plan start, kinematic thereafter), the five commanded
displacements in the body frame, the imagined block motion over the block, and
optionally the imagined latent itself. Two variants: `geo` and `geo+latent`.

Then the test that matters: the recorded E32 decisions (probe dumps of the
seed-1 grounded actor on draws 43/44) are rescored with D along the refiner's
own imagined trajectories -- predicted fiction magnitude, and the terminal block
error to the goal after correction -- and compared between failed and
successful episodes. If D separates them, unlike every physics prior, the
lever is real.

Env: D, CD, H5 (from the job), PROBE_TAG (tag whose probes_rlp_s{43,44} and
evallogs to rescore), N_WINDOWS (default 300000), EPOCHS (default 8).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from rlp.core.grounding import ACTION_SCALE, AGENT_RADIUS, fit_agent_kinematics, fit_state_probe, tee_distance
from rlp.core.rollout import rollout_traj
from rlp.core.world_model import load_pretrained
from rlp.data import LatentCache

D = Path(os.environ["D"])
CD = Path(os.environ.get("CD", "/checkpoints/armin@pantheon.inc/counterstrike/caches"))
H5 = os.environ["H5"]
PROBE_TAG = os.environ.get("PROBE_TAG", "pusht-gterms-g10s1-20260913")
N_WINDOWS = int(os.environ.get("N_WINDOWS", "300000"))
EPOCHS = int(os.environ.get("EPOCHS", "8"))
FS, H, HIST = 5, 5, 3
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def log(msg: str) -> None:
    print(f"[dsc] {msg}", flush=True)


t0 = time.time()
wm = load_pretrained(str(D / "lewm_pusht_official")).to(dev).eval()
wm.requires_grad_(False)
cache = LatentCache.load(str(CD / "counterstrike_fs1.pt"), mmap=True)
n = int(cache.z.shape[0])
with h5py.File(H5, "r") as h:
    epi = np.asarray(h["episode_idx"][:n]).reshape(-1).astype(np.int64)
    state = np.asarray(h["state"][:n], dtype=np.float64)
    act_all = np.asarray(h["action"][:], dtype=np.float64)
if not np.array_equal(epi, cache.episode_idx.numpy()):
    raise SystemExit("[dsc] h5 rows do not align with the fs1 cache")
act = np.nan_to_num(act_all[:n])
amu, astd = np.nanmean(act_all, 0), np.nanstd(act_all, 0) + 1e-6
act_n = ((act - amu) / astd).astype(np.float32)
probe_w, prep = fit_state_probe(cache.z, state, seed=0)
gains, kin_r2 = fit_agent_kinematics(state[:, :2], act, epi, seed=0)
log(f"probe {prep} | kinematics g0 {gains[0]:.3f} g1 {gains[1]:.3f} r2 {kin_r2:.3f} | {time.time() - t0:.0f}s")
W = probe_w.to(dev)
G0, G1 = float(gains[0]), float(gains[1])


def decode(z: torch.Tensor) -> torch.Tensor:  # (..., D) -> (..., 6)
    return torch.cat([z, z.new_ones(*z.shape[:-1], 1)], dim=-1) @ W


def body(v: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors (..., 2) into the block body frame given (cos, sin) (..., 2)."""
    c, s = cs[..., :1], cs[..., 1:2]
    return torch.cat([v[..., :1] * c + v[..., 1:2] * s, -v[..., :1] * s + v[..., 1:2] * c], dim=-1)


def unit(cs: torch.Tensor) -> torch.Tensor:
    return cs / cs.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def angle(cs: torch.Tensor) -> torch.Tensor:
    return torch.atan2(cs[..., 1], cs[..., 0])


def wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + torch.pi) % (2 * torch.pi) - torch.pi


def block_features(
    z_start: torch.Tensor, z_end: torch.Tensor, p_start: torch.Tensor, u_raw: torch.Tensor
) -> torch.Tensor:
    """Per-block features (B, 21): pose(4) rel agent(2) commands(10) imagined motion(3) gap(2).

    z_start/z_end: imagined latents at the block's start/end (B, D); p_start: agent
    position at block start (B, 2) px; u_raw: the block's 5 commanded displacements
    (B, 5, 2) px. The body frame is the decoded pose at block start.
    """
    s0, s1 = decode(z_start), decode(z_end)
    cs0 = unit(s0[:, 4:6])
    rel = body(p_start - s0[:, 2:4], cs0)
    cmd = body(u_raw, cs0.unsqueeze(1)).reshape(u_raw.shape[0], -1)
    dpos = body(s1[:, 2:4] - s0[:, 2:4], cs0)
    dang = wrap(angle(s1[:, 4:6]) - angle(cs0)).unsqueeze(1)
    # kinematic agent path inside the block against the start pose: closest approach and end gap
    steps = G0 * u_raw + G1 * torch.cat([torch.zeros_like(u_raw[:, :1]), u_raw[:, :-1]], dim=1)
    path = p_start.unsqueeze(1) + steps.cumsum(dim=1)
    gap = tee_distance(path, s0[:, None, 2:4], cs0.unsqueeze(1)) - AGENT_RADIUS
    return torch.cat(
        [
            s0[:, 2:4] / 512.0,
            cs0,
            rel / 100.0,
            cmd / 100.0,
            dpos / 10.0,
            dang,
            gap.amin(1, keepdim=True) / 50.0,
            gap[:, -1:] / 50.0,
        ],
        dim=1,
    )


def agent_end(p_start: torch.Tensor, u_raw: torch.Tensor, u_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    steps = G0 * u_raw + G1 * torch.cat([u_prev.unsqueeze(1), u_raw[:, :-1]], dim=1)
    return p_start + steps.sum(dim=1), u_raw[:, -1]


def residual_target(z_real_end: torch.Tensor, z_imag_end: torch.Tensor, z_start: torch.Tensor) -> torch.Tensor:
    """Decoded real minus decoded imagined block pose at block end, body frame of the block start: (B, 3)."""
    sr, si, s0 = decode(z_real_end), decode(z_imag_end), decode(z_start)
    cs0 = unit(s0[:, 4:6])
    dpos = body(sr[:, 2:4] - si[:, 2:4], cs0)
    dang = wrap(angle(sr[:, 4:6]) - angle(si[:, 4:6])).unsqueeze(1)
    return torch.cat([dpos, dang], dim=1)


# ------------------------------------------------------------ windows over the training episodes
rng = np.random.default_rng(0)
starts_all = []
bounds = np.flatnonzero(np.diff(epi)) + 1
ep_start = np.concatenate([[0], bounds])
ep_end = np.concatenate([bounds, [n]])
for a, b in zip(ep_start, ep_end, strict=True):
    lo, hi = a + FS * (HIST - 1), b - FS * H
    if hi > lo:
        starts_all.append(np.arange(lo, hi))
starts_all = np.concatenate(starts_all)
starts = np.sort(rng.choice(starts_all, size=min(N_WINDOWS, len(starts_all)), replace=False))
val_mask = epi[starts] >= 14000
log(f"{len(starts_all)} candidate windows, using {len(starts)} ({val_mask.sum()} val from episodes >= 14000)")

zc = cache.z  # mmap
feats, tgts, meta = [], [], []
BS = 1024
with torch.no_grad():
    for i in range(0, len(starts), BS):
        r = starts[i : i + BS]
        rt = torch.as_tensor(r)
        hist_rows = torch.stack([rt - FS * (HIST - 1 - j) for j in range(HIST)], dim=1)  # (B, 3)
        zh = zc[hist_rows.reshape(-1)].to(dev).float().view(len(r), HIST, -1)
        ah = torch.as_tensor(
            np.stack(
                [
                    act_n[np.add.outer(r, np.arange(-2 * FS, -FS))].reshape(len(r), -1),
                    act_n[np.add.outer(r, np.arange(-FS, 0))].reshape(len(r), -1),
                ],
                axis=1,
            ),
            device=dev,
        )
        plan_idx = np.add.outer(r, np.arange(0, FS * H))
        plan = torch.as_tensor(act_n[plan_idx].reshape(len(r), H, FS * 2), device=dev)
        traj = rollout_traj(wm, zh, ah, plan).float()  # (B, H, D) imagined
        z_prev = zh[:, -1]
        p = torch.as_tensor(state[r, :2], device=dev, dtype=torch.float32)
        u_prev = torch.as_tensor(act[r - 1] * ACTION_SCALE, device=dev, dtype=torch.float32)
        u_prev = torch.where(
            torch.as_tensor(epi[r - 1] == epi[r], device=dev)[:, None], u_prev, torch.zeros_like(u_prev)
        )
        gap_true = (
            tee_distance(
                torch.as_tensor(state[plan_idx, :2], device=dev, dtype=torch.float32),
                torch.as_tensor(state[plan_idx, 2:4], device=dev, dtype=torch.float32),
                torch.as_tensor(
                    np.stack([np.cos(state[plan_idx, 4]), np.sin(state[plan_idx, 4])], -1),
                    device=dev,
                    dtype=torch.float32,
                ),
            )
            - AGENT_RADIUS
        )  # (B, 25) true gap at each primitive step of the window
        for k in range(H):
            u_raw = torch.as_tensor(
                act[np.add.outer(r, np.arange(k * FS, (k + 1) * FS))] * ACTION_SCALE, device=dev, dtype=torch.float32
            )
            z_real_end = zc[torch.as_tensor(r + (k + 1) * FS)].to(dev).float()
            f = block_features(z_prev, traj[:, k], p, u_raw)
            y = residual_target(z_real_end, traj[:, k], z_prev)
            feats.append(torch.cat([f, z_prev], dim=1).cpu())
            tgts.append(y.cpu())
            contact = (gap_true[:, k * FS : (k + 1) * FS].amin(1) <= 0).cpu()
            meta.append(torch.stack([torch.as_tensor(val_mask[i : i + BS]), contact, torch.full((len(r),), k)], dim=1))
            p, u_prev = agent_end(p, u_raw, u_prev)
            z_prev = traj[:, k]
X = torch.cat(feats)
Y = torch.cat(tgts)
M = torch.cat(meta)
NF = 21
log(
    f"built {len(X)} block samples in {time.time() - t0:.0f}s | residual |pos| mean {Y[:, :2].norm(dim=1).mean():.2f} px, median {Y[:, :2].norm(dim=1).median():.2f} px, |ang| mean {torch.rad2deg(Y[:, 2].abs()).mean():.1f} deg"  # noqa: E501
)
for k in range(H):
    sel = M[:, 2] == k
    c = M[sel, 1].bool()
    log(
        f"  block {k + 1}: residual |pos| mean {Y[sel, :2].norm(dim=1).mean():.2f} px | contact blocks {c.float().mean() * 100:.0f}%: {Y[sel][c][:, :2].norm(dim=1).mean():.2f} px | no-contact: {Y[sel][~c][:, :2].norm(dim=1).mean():.2f} px"  # noqa: E501
    )

# ------------------------------------------------------------ train D (two variants)
val = M[:, 0].bool()
Ysc = torch.tensor([10.0, 10.0, 0.25])  # px, px, rad scales -> O(1) targets
mu, sd = X[~val].mean(0), X[~val].std(0) + 1e-6


class Disc(nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_variant(name: str, cols: slice) -> tuple[Disc, dict[str, float]]:
    xin = ((X - mu) / sd)[:, cols]
    yin = Y / Ysc
    model = Disc(xin.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_idx = torch.nonzero(~val).squeeze(1)
    steps_per_epoch = len(tr_idx) // 4096
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    xin_d, yin_d = xin.to(dev), yin.to(dev)
    for ep in range(EPOCHS):
        perm = tr_idx[torch.randperm(len(tr_idx))]
        tot = 0.0
        for j in range(steps_per_epoch):
            b = perm[j * 4096 : (j + 1) * 4096].to(dev)
            loss = nn.functional.smooth_l1_loss(model(xin_d[b]), yin_d[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        with torch.no_grad():
            pv = model(xin_d[val.to(dev)]) * Ysc.to(dev)
            yv = Y[val].to(dev)
            mae_d = (pv[:, :2] - yv[:, :2]).norm(dim=1).mean()
            mae_0 = yv[:, :2].norm(dim=1).mean()
        log(
            f"  {name} epoch {ep + 1}/{EPOCHS}: train loss {tot / steps_per_epoch:.4f} | val |pos residual| after D {mae_d:.2f} px vs before {mae_0:.2f} px"  # noqa: E501
        )
    model.eval()
    with torch.no_grad():
        pv = (model(xin_d[val.to(dev)]) * Ysc.to(dev)).cpu()
        yv = Y[val]
        r2 = 1 - ((pv - yv) ** 2).sum(0) / ((yv - yv.mean(0)) ** 2).sum(0)
        rep = {
            "r2_x": float(r2[0]),
            "r2_y": float(r2[1]),
            "r2_ang": float(r2[2]),
            "mae_after": float((pv[:, :2] - yv[:, :2]).norm(dim=1).mean()),
            "mae_before": float(yv[:, :2].norm(dim=1).mean()),
        }
        mv = M[val]
        for k in range(H):
            sel = mv[:, 2] == k
            rep[f"mae_after_k{k + 1}"] = float((pv[sel, :2] - yv[sel, :2]).norm(dim=1).mean())
            rep[f"mae_before_k{k + 1}"] = float(yv[sel, :2].norm(dim=1).mean())
        c = mv[:, 1].bool()
        rep["mae_after_contact"], rep["mae_before_contact"] = (
            float((pv[c, :2] - yv[c, :2]).norm(dim=1).mean()),
            float(yv[c, :2].norm(dim=1).mean()),
        )
        rep["mae_after_free"], rep["mae_before_free"] = (
            float((pv[~c, :2] - yv[~c, :2]).norm(dim=1).mean()),
            float(yv[~c, :2].norm(dim=1).mean()),
        )
    log(f"{name}: " + " ".join(f"{k}={v:.3f}" for k, v in rep.items()))
    torch.save(
        {
            "sd": model.state_dict(),
            "cols": (cols.start, cols.stop),
            "mu": mu,
            "sd_x": sd,
            "ysc": Ysc,
            "probe_w": probe_w,
            "gains": gains,
            "amu": amu,
            "astd": astd,
            "report": rep,
        },
        D / f"discrepancy_{name}.pt",
    )
    return model, rep


models = {}
models["geo"], _ = train_variant("geo", slice(0, NF))
models["geolat"], _ = train_variant("geolat", slice(0, X.shape[1]))
log(f"training done in {time.time() - t0:.0f}s")

# ------------------------------------------------------------ the decisive test: rescore the E32 decisions
PD = Path(f"/checkpoints/armin@pantheon.inc/{PROBE_TAG}")


def successes(draw: int) -> np.ndarray:
    text = (PD / f"evallog_rlp_s{draw}.txt").read_text()
    m = re.findall(r"episode_successes': array\(\[(.*?)\]\)", text, re.S)
    return np.array([t == "True" for t in re.findall(r"True|False", m[-1])])


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    """P(score of a failure > score of a success) via ranks (pos = failure mask)."""
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sep(name: str, score: np.ndarray, fail: np.ndarray) -> None:
    thr = np.quantile(score[~fail], 0.9)
    log(
        f"     {name:28s} fail/succ mean {score[fail].mean():8.2f}/{score[~fail].mean():8.2f}  AUC(fail>succ) {auc(score, fail):.2f}  fails above successes' P90: {(score[fail] > thr).mean() * 100:3.0f}%"  # noqa: E501
    )


for draw in (43, 44):
    files = sorted((PD / f"probes_rlp_s{draw}").glob("probe_*.pt"))
    if not files:
        log(f"draw {draw}: no probes at {PD}")
        continue
    ok = successes(draw)
    for kdec, f in enumerate(files):
        p = torch.load(f, map_location="cpu", weights_only=False)
        nb = min(len(ok), p["z0"].shape[0])
        fail = ~ok[:nb]
        z0, traj, A, zg = (
            p["z0"][:nb].float().to(dev),
            p["z_traj"][:nb].float().to(dev),
            p["A"][:nb].float().to(dev),
            p["zg"][:nb].float().to(dev),
        )
        ag0, up = p["ground_agent0"][:nb].float().to(dev), p["ground_u_prev"][:nb].float().to(dev)
        with torch.no_grad():
            u_all = (
                A.view(nb, H, FS, 2) * torch.as_tensor(astd, device=dev, dtype=torch.float32)
                + torch.as_tensor(amu, device=dev, dtype=torch.float32)
            ) * ACTION_SCALE
            goal_b = decode(zg)[:, 2:4]
            term_b = decode(traj[:, -1])[:, 2:4]
            base_err = (term_b - goal_b).norm(dim=1)
            pred = {name: [] for name in models}
            z_prev, pcur, uprev = z0, ag0, up
            for k in range(H):
                fk = torch.cat([block_features(z_prev, traj[:, k], pcur, u_all[:, k]), z_prev], dim=1)
                xin = (fk.cpu() - mu) / sd
                cs0 = unit(decode(z_prev)[:, 4:6])
                for name, model in models.items():
                    cols = slice(0, NF) if name == "geo" else slice(0, X.shape[1])
                    out = model(xin[:, cols].to(dev)) * Ysc.to(dev)  # body frame residual (real - imagined)
                    # back to world frame: rotate by +theta
                    c, s = cs0[:, :1], cs0[:, 1:2]
                    world = torch.cat([out[:, :1] * c - out[:, 1:2] * s, out[:, :1] * s + out[:, 1:2] * c], dim=1)
                    pred[name].append(world)
                pcur, uprev = agent_end(pcur, u_all[:, k], uprev)
                z_prev = traj[:, k]
        log(
            f"draw {draw} decision {kdec}: n={nb} fails={int(fail.sum())} | uncorrected terminal block error to goal fail/succ {base_err[fail].mean():.1f}/{base_err[~fail].mean():.1f} px"  # noqa: E501
        )
        sep("critic energy E (probe)", p["E"][:nb].float().numpy(), fail)
        sep("uncorrected terminal error", base_err.cpu().numpy(), fail)
        for name in models:
            corr = torch.stack(pred[name], dim=1)  # (n, H, 2) world-frame residuals
            fiction = corr.norm(dim=2).sum(1)
            corrected_err = (term_b + corr.sum(1) - goal_b).norm(dim=1)
            sep(f"{name}: predicted fiction sum", fiction.cpu().numpy(), fail)
            sep(f"{name}: corrected terminal error", corrected_err.cpu().numpy(), fail)
            sep(f"{name}: correction - baseline", (corrected_err - base_err).cpu().numpy(), fail)
log(f"done in {time.time() - t0:.0f}s")
