"""Restrict a latent cache to an episode range — see ../DATA_SPLIT_POLICY.md.

Used to build the train-split caches for the episode-disjoint Dyna control, so the
TD teacher and the LIP critic never see an episode the evaluation draws from.

Why a prefix range is the right choice: COLLECT = episodes 0..7999 is a prefix of
the 10k expert set, so kept episode ids stay 0..7999 contiguous. Nothing needs
renumbering, `expert_actions.h5` lookups by episode id keep working, and
train_lip_ac.py's ep_off/ep_len bookkeeping stays valid. Do NOT switch to a
non-prefix range without checking those assumptions.

Usage:
    python filter_cache_eprange.py IN.pt OUT.pt --lo 0 --hi 8000
"""

from __future__ import annotations

import argparse

import torch


EP_KEYS = ("episode_idx", "ep_idx", "episodes", "ep")
STEP_KEYS = ("step_idx", "steps", "step")


def _find(d, names):
    for n in names:
        if n in d:
            return n
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--lo", type=int, required=True)
    p.add_argument("--hi", type=int, required=True, help="exclusive")
    a = p.parse_args()

    c = torch.load(a.inp, map_location="cpu")
    if not isinstance(c, dict):
        raise SystemExit(f"unexpected cache type {type(c)}; expected a dict")

    ep_key = _find(c, EP_KEYS)
    if ep_key is None:
        raise SystemExit(
            f"no episode column found in {a.inp}; keys = {sorted(c.keys())}. "
            "Refusing to guess — a wrong mask would silently corrupt the split."
        )
    ep = torch.as_tensor(c[ep_key]).reshape(-1)
    keep = (ep >= a.lo) & (ep < a.hi)
    n_before, n_after = int(ep.numel()), int(keep.sum())
    if n_after == 0:
        raise SystemExit(f"range {a.lo}:{a.hi} keeps 0 rows of {n_before}")

    out = {}
    for k, v in c.items():
        if torch.is_tensor(v) and v.shape and v.shape[0] == n_before:
            out[k] = v[keep]
        elif isinstance(v, list) and len(v) == n_before:
            out[k] = [x for x, m in zip(v, keep.tolist()) if m]
        else:
            out[k] = v  # scalars, meta, per-episode tables handled below

    kept_ep = ep[keep]
    lo_seen, hi_seen = int(kept_ep.min()), int(kept_ep.max())
    # Per-episode side tables (len == n_episodes, not n_rows) must be truncated too.
    n_ep_before = int(ep.max()) + 1
    for k, v in list(out.items()):
        if k == ep_key:
            continue
        if torch.is_tensor(v) and v.shape and v.shape[0] == n_ep_before and n_ep_before != n_before:
            out[k] = v[a.lo:a.hi]
            print(f"  per-episode table {k!r}: {n_ep_before} -> {out[k].shape[0]}")

    meta = out.get("meta")
    if isinstance(meta, dict):
        meta = dict(meta)
        meta["ep_range"] = [a.lo, a.hi]
        meta["split_note"] = "train split for the episode-disjoint Dyna control"
        out["meta"] = meta

    torch.save(out, a.out)
    print(f"{a.inp} -> {a.out}")
    print(f"  rows {n_before} -> {n_after}   episodes kept {lo_seen}..{hi_seen}")
    assert hi_seen < a.hi and lo_seen >= a.lo, "MASK BUG: out-of-range episode survived"
    print("FILTER_CACHE_DONE")


if __name__ == "__main__":
    main()
