"""Is Hi-LeWM's frozen low level the same base our PushT actors were trained on?

Their checkpoint (arXiv 2607.12547, Zenodo 10.5281/zenodo.21353240) carries the
whole model: the frozen LeWM (``model.encoder`` / ``model.low_predictor`` /
``model.action_encoder`` / ``model.projector`` / ``model.low_pred_proj``) plus
the high-level parts (``model.high_predictor``, ``model.latent_action_encoder``,
``model.macro_to_condition``, ``model.high_pred_proj``).

Their bundled base carries the RAW release key naming (transformers-4 HF ViT:
``encoder.encoder.layer.0.attention.attention.query.weight``), so compare it
against the **raw** ``quentinll/lewm-pusht`` ``weights.pt``, not against our
converted asset — conversion applies the transformers-4 -> 5 rename
(``encoder.layers.0.attention.q_proj.weight``) and the names would not line up.
That rename is 1:1 and deterministic, so identity on the raw download implies
identity of the converted copy our actors were trained against.

Per-group tensor counts on both sides are ``{encoder: 198, predictor: 81,
action_encoder: 6, projector: 9, pred_proj: 9}`` = 303. This script applies the
prefix map and reports, per group, how many tensors are bit-identical and the
worst absolute/relative deviation.

Exit status is 0 when every shared tensor matches within tolerance, 1 otherwise,
so a job can gate on it. Unmatched keys are always listed: a shape-only fallback
match is reported but never counted as agreement.

Usage:
    python scripts/hilewm/compare_base.py --theirs <hi_lewm_weights.ckpt> \
        --ours <converted weights.pt> [--atol 0] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

# their frozen-low-level prefix -> our converted-checkpoint prefix
PREFIX_MAP = {
    "model.encoder.": "encoder.",
    "model.low_predictor.": "predictor.",
    "model.action_encoder.": "action_encoder.",
    "model.projector.": "projector.",
    "model.low_pred_proj.": "pred_proj.",
}

# high-level-only groups: expected to have no counterpart on our side
HIGH_LEVEL_PREFIXES = (
    "model.high_predictor.",
    "model.latent_action_encoder.",
    "model.macro_to_condition.",
    "model.high_pred_proj.",
)


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    blob: Any = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
    if not isinstance(blob, dict):
        raise TypeError(f"{path}: expected a state dict, got {type(blob).__name__}")
    return {k: v for k, v in blob.items() if isinstance(v, torch.Tensor)}


def translate(key: str) -> str | None:
    """Map one of their keys into our naming, or None if it is high-level-only."""
    for src, dst in PREFIX_MAP.items():
        if key.startswith(src):
            return dst + key[len(src) :]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theirs", required=True, help="Hi-LeWM *_weights.ckpt")
    ap.add_argument("--ours", required=True, help="converted quentinll/lewm-pusht weights.pt")
    ap.add_argument("--atol", type=float, default=0.0, help="0 = require bit-identical")
    ap.add_argument("--json", default="", help="also write the report as JSON")
    args = ap.parse_args()

    theirs = load_state_dict(args.theirs)
    ours = load_state_dict(args.ours)

    print(f"theirs: {len(theirs)} tensors  ({args.theirs})")
    print(f"ours:   {len(ours)} tensors  ({args.ours})")

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "identical": 0, "close": 0, "max_abs": 0.0, "max_rel": 0.0, "missing": []}
    )
    high_level = 0
    unmapped: list[str] = []
    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for key, their_tensor in theirs.items():
        if key.startswith(HIGH_LEVEL_PREFIXES):
            high_level += 1
            continue
        our_key = translate(key)
        if our_key is None:
            unmapped.append(key)
            continue
        group = our_key.split(".")[0]
        row = stats[group]
        row["n"] += 1
        if our_key not in ours:
            row["missing"].append(our_key)
            continue
        our_tensor = ours[our_key]
        if tuple(our_tensor.shape) != tuple(their_tensor.shape):
            shape_mismatch.append((key, tuple(their_tensor.shape), tuple(our_tensor.shape)))
            continue
        a = their_tensor.detach().to(torch.float64)
        b = our_tensor.detach().to(torch.float64)
        if torch.equal(their_tensor, our_tensor):
            row["identical"] += 1
        diff = (a - b).abs()
        max_abs = float(diff.max()) if diff.numel() else 0.0
        scale = float(b.abs().max()) if b.numel() else 0.0
        max_rel = max_abs / scale if scale > 0 else 0.0
        row["max_abs"] = max(row["max_abs"], max_abs)
        row["max_rel"] = max(row["max_rel"], max_rel)
        if max_abs <= args.atol:
            row["close"] += 1

    print(f"\nhigh-level-only tensors skipped: {high_level}")
    if unmapped:
        print(f"unmapped keys ({len(unmapped)}): {unmapped[:8]}")

    print(f"\n{'group':18s} {'n':>4s} {'bit-identical':>14s} {'within atol':>12s} {'max_abs':>12s} {'max_rel':>10s}")
    all_ok = True
    for group in sorted(stats):
        row = stats[group]
        ok = row["identical"] == row["n"] or row["close"] == row["n"]
        all_ok &= bool(ok)
        print(
            f"{group:18s} {row['n']:4d} {row['identical']:14d} {row['close']:12d} "
            f"{row['max_abs']:12.3e} {row['max_rel']:10.3e}"
        )
        if row["missing"]:
            all_ok = False
            print(f"  missing on our side ({len(row['missing'])}): {row['missing'][:5]}")

    if shape_mismatch:
        all_ok = False
        print(f"\nshape mismatches ({len(shape_mismatch)}):")
        for key, their_shape, our_shape in shape_mismatch[:10]:
            print(f"  {key}: theirs {their_shape} vs ours {our_shape}")

    our_extra = set(ours) - {translate(k) for k in theirs if translate(k)}
    if our_extra:
        print(f"\nkeys only on our side ({len(our_extra)}): {sorted(our_extra)[:8]}")

    verdict = "IDENTICAL" if all_ok else "DIFFERENT"
    print(f"\nVERDICT: frozen low level is {verdict} (atol={args.atol})")
    if all_ok:
        print("=> their high level and our PushT actors share one frozen base; the")
        print("   head-to-head needs no base-mismatch caveat.")
    else:
        print("=> bases differ. Any Hi-LeWM vs RLP comparison must either re-run RLP")
        print("   on their base or state the mismatch as a confound.")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "verdict": verdict,
                    "atol": args.atol,
                    "high_level_skipped": high_level,
                    "groups": {g: {k: v for k, v in row.items()} for g, row in stats.items()},
                    "shape_mismatch": [
                        {"key": k, "theirs": list(t), "ours": list(o)} for k, t, o in shape_mismatch
                    ],
                },
                indent=2,
            )
        )

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
