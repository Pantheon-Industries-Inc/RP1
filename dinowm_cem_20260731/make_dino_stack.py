"""Create the two private script variants the DINO-WM TD/LIP campaign needs.

1. eval_wm_dino.py  (already created with the interpolate_pos_encoding fix)
   + pool patch-token latents before the metric head.

   `_MetricCost.get_cost` does `p_raw[..., -1, :]` to take the last imagined
   FRAME. That is correct for LeWM, whose `predicted_emb` is (B, cand, T, D).
   PreJEPA/DINO-WM emits (B, cand, T, P, D) -- so the same expression silently
   takes the last PATCH of every frame instead. The TD metric is trained on
   latents mean-pooled over patches (see cache_latents_dino.py), so reduce the
   same way here, gated on `+metric_pool_dim=<D>`.

2. train_metric_sweep.py
   Expose `--lr`, `--weight-decay` and `--polyak-tau`. `TDConfig` already has
   all three (lr 1e-3, weight_decay 1e-4, tau 0.005) but `train_metric.py`
   passes none of them, so an lr sweep is impossible via the stock CLI.
   NB `tau` here is the target-network Polyak rate, NOT the IQL expectile --
   the expectile is the existing `--expectile` flag.

Both are copies so that the shared scripts other sessions are running stay
untouched.
"""

import sys
from pathlib import Path

PLAN = Path('/workspace/code/stable-worldmodel/scripts/plan')


def patch(path_in, path_out, edits, *, expect_unique=True):
    text = Path(path_in).read_text()
    for old, new in edits:
        n = text.count(old)
        if expect_unique and n != 1:
            sys.exit(f'{path_in}: expected 1 occurrence, found {n} of:\n{old[:120]}')
        text = text.replace(old, new, 1)
    Path(path_out).write_text(text)
    print(f'wrote {path_out}')


# ---------------------------------------------------------------- 1. eval hook
POOL_OLD = """                    p_raw = info_dict['predicted_emb']
                    g_raw = info_dict['goal_emb']
"""
POOL_NEW = """                    p_raw = info_dict['predicted_emb']
                    g_raw = info_dict['goal_emb']
                    # Patch-token world models (PreJEPA / DINO-WM) emit
                    # (..., T, P, D). The metric head is trained on latents
                    # mean-pooled over patches, and the `[..., -1, :]` below
                    # assumes the LAST axis pair is (T, D) -- without this
                    # reduction it would index the last PATCH, not the last
                    # imagined frame. Slice to the pixel dims first so the
                    # tiled action embedding never reaches the metric.
                    _pd = int(cfg.get('metric_pool_dim', 0) or 0)
                    if _pd:
                        if p_raw.shape[-1] > _pd:
                            p_raw = p_raw[..., :_pd].mean(dim=-2)
                        if g_raw.shape[-1] > _pd:
                            g_raw = g_raw[..., :_pd].mean(dim=-2)
                        if not self._shapes_printed[0]:
                            print(
                                f'[metric-hook] pooled to pred '
                                f'{tuple(p_raw.shape)} goal {tuple(g_raw.shape)}',
                                flush=True,
                            )
"""

patch(PLAN / 'eval_wm_dino.py', PLAN / 'eval_wm_dino.py', [(POOL_OLD, POOL_NEW)])

# ------------------------------------------------------------- 2. metric sweep
ARG_OLD = """    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
"""
ARG_NEW = """    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    # TDConfig carries these but the stock CLI never passes them, so an lr or
    # target-rate sweep is impossible without the flags.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--polyak-tau", type=float, default=0.005,
                   help="target-network soft-update rate (TDConfig.tau); this "
                        "is NOT the IQL expectile -- see --expectile")
"""

CFG_OLD = """            batch_size=args.batch_size, steps=args.steps, seed=args.seed,
            symmetric=args.symmetric,
        )
        module = learners.td.fit(cache, cfg, device)"""
CFG_NEW = """            batch_size=args.batch_size, steps=args.steps, seed=args.seed,
            symmetric=args.symmetric,
            lr=args.lr, weight_decay=args.weight_decay, tau=args.polyak_tau,
        )
        print(f"[td] lr={args.lr} wd={args.weight_decay} "
              f"polyak_tau={args.polyak_tau} expectile={args.expectile}",
              flush=True)
        module = learners.td.fit(cache, cfg, device)"""

patch(PLAN / 'train_metric.py', PLAN / 'train_metric_sweep.py',
      [(ARG_OLD, ARG_NEW), (CFG_OLD, CFG_NEW)])

# ------------------------------------------------------------------- verify
import subprocess

for f in ('eval_wm_dino.py', 'train_metric_sweep.py'):
    r = subprocess.run([sys.executable, '-m', 'py_compile', str(PLAN / f)],
                       capture_output=True, text=True)
    print(f'{f}: compile {"OK" if r.returncode == 0 else "FAILED " + r.stderr[:200]}')
