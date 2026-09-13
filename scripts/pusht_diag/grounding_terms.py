"""Score the physics-grounding term on the deployed decisions of an eval run.

POST script for counterstrike_pusht.yaml (needs PROBES=1 on the `rlp` condition
and a grounding-trained actor). For each recorded decision it recomputes the
GroundingPenalty terms from the probe dump (z0, imagined trajectory, plan, agent
position) and joins them with the per-episode outcome, then asks two things:

1. Does the term see the failures? Mean penalty / decoded displacement /
   no-contact probability for failed vs successful episodes, per decision.
2. What would tighter thresholds have seen? The same decisions rescored with
   alternative dead zones and contact margins (no retraining).

Env: D (experiment dir), DRAWS (comma list, default "43"), COND (default "rlp").
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch

from rlp.core.grounding import GroundingPenalty

D = Path(os.environ["D"])
DRAWS = [int(x) for x in os.environ.get("DRAWS", "43").split(",")]
COND = os.environ.get("COND", "rlp")


def log(msg: str) -> None:
    print(f"[gt] {msg}", flush=True)


ck = torch.load(D / "train_checkpoints" / "planner.pt", map_location="cpu", weights_only=False)
payload = ck.get("grounding")
if not payload:
    raise SystemExit("[gt] actor checkpoint carries no grounding")
base = GroundingPenalty.from_export(payload)
log(
    f"trained thresholds: margin {base.margin:.1f} deadzone {base.deadzone:.1f} tau {base.tau:g} weight {base.weight:g}"
)


def successes(draw: int) -> np.ndarray:
    text = (D / f"evallog_{COND}_s{draw}.txt").read_text()
    m = re.findall(r"episode_successes': array\(\[(.*?)\]\)", text, re.S)
    if not m:
        raise SystemExit(f"[gt] no episode_successes in evallog for draw {draw}")
    return np.array([t == "True" for t in re.findall(r"True|False", m[-1])])


def variants() -> list[tuple[str, GroundingPenalty]]:
    out = []
    for dz in (base.deadzone, base.deadzone / 2, 5.0):
        for mg in (base.margin, 5.0, 0.0):
            g = GroundingPenalty.from_export(payload)
            g.deadzone, g.margin, g.weight = float(dz), float(mg), 1.0
            out.append((f"dz{dz:4.1f} mg{mg:4.1f}", g))
    return out


for draw in DRAWS:
    files = sorted((D / f"probes_{COND}_s{draw}").glob("probe_*.pt"))
    if not files:
        log(f"draw {draw}: no probe files")
        continue
    ok = successes(draw)
    log(f"draw {draw}: {len(files)} decisions recorded, success {ok.mean() * 100:.0f}% ({ok.sum()}/{len(ok)})")
    for k, f in enumerate(files):
        p = torch.load(f, map_location="cpu", weights_only=False)
        if "ground_agent0" not in p:
            log("probe dump lacks ground_agent0 (older solver); cannot rescore")
            break
        z0, traj, A = p["z0"].float(), p["z_traj"].float(), p["A"].float()
        ag0, up = p["ground_agent0"].float(), p["ground_u_prev"].float()
        n = min(len(ok), z0.shape[0])
        okk = ok[:n]
        with torch.no_grad():
            t = base.terms(z0[:n], traj[:n], A[:n], ag0[:n], up[:n])
        pen = base.weight * t["penalty"].numpy()
        disp = t["displacement"][:, -1].numpy()
        unsup = t["unsupported"][:, -1].numpy()
        gap = t["gap"].amin(dim=1).numpy()
        fiction = disp > 30.0
        seen = fiction & (unsup > 0.5)
        hidden = fiction & (unsup <= 0.5)
        log(
            f"draw {draw} decision {k}: E {p['E'].float().mean():.2f} | "
            f"penalty fail/succ {pen[~okk].mean():.3f}/{pen[okk].mean():.3f} | "
            f"disp_end fail/succ {disp[~okk].mean():.1f}/{disp[okk].mean():.1f} px | "
            f"no-contact fail/succ {unsup[~okk].mean():.2f}/{unsup[okk].mean():.2f} | "
            f"closest gap fail/succ {np.median(gap[~okk]):.1f}/{np.median(gap[okk]):.1f} px"
        )
        log(
            f"draw {draw} decision {k}: imagined displacement > 30 px on {fiction[~okk].sum()}/{(~okk).sum()} fails "
            f"and {fiction[okk].sum()}/{okk.sum()} successes; of the fails' large-motion plans "
            f"{seen[~okk].sum()} are no-contact (term sees them) "
            f"and {hidden[~okk].sum()} are under contact (term blind)"
        )
        for name, g in variants():
            with torch.no_grad():
                v = g.terms(z0[:n], traj[:n], A[:n], ag0[:n], up[:n])["penalty"].numpy()
            thr = np.quantile(v[okk], 0.9) if okk.any() else float("inf")
            log(
                f"   what-if {name}: penalty fail/succ {v[~okk].mean():.3f}/{v[okk].mean():.3f}; "
                f"fails above the successes' P90 ({thr:.2f}): {(v[~okk] > thr).mean() * 100:.0f}%"
            )
log("done")
