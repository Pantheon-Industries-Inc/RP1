"""Stop materializing the stacked window cache; gather windows per batch.

WHY. train_window.py builds the F-frame window as

    Zs = torch.cat([Z[i] for i in idxs], dim=1)     # (N, F*D), oldest -> newest

which costs SEVEN times the base cache, not F times: fancy-indexing Z[i] copies,
so the list comprehension holds F full copies and torch.cat then allocates the
output alongside them. At the 192-d bases this is invisible (576 MB). At
DINO-WM's unpooled 75,264-d latent it is

    base                2.01M x  75,264 x 4B  =   605 GB
    F=3 indexed copies                        = 1,815 GB
    cat output          2.01M x 225,792 x 4B  = 1,815 GB
    peak                                        ~4.2 TB

against ~1.9 TB of RAM. The obvious response -- cap the episode prefix and fit
the value on a fraction of the corpus -- is unnecessary, because nothing ever
needs all N windows resident. NStepGoalSampler's entire use of the cache is

    "z_t": self.z[t_idx], "z_tn": self.z[tn_idx], "z_g": self.z[g_idx]

three gathers of batch_size rows, plus cache.latent_dim for the head width. A
view that gathers on demand therefore costs base + one batch:

    605 GB + 1024 x 225,792 x 4B  =  605 GB + 925 MB

and the value trains on all 2.01M rows. LatentCache is a plain dataclass whose
`z: torch.Tensor` is an annotation rather than a check, and episodes() reads only
episode_idx/step_idx, so the view substitutes directly.

Behaviour is identical to the materialized path: same rows, same concatenation
order, same dtype. Only peak memory changes. The tiled-goal diagnostic indexes
Zs with a 4096-row selection and works unchanged.

Replaces the WHOLE line rather than a substring -- a first attempt anchored on
"...dim=1)  # (N, F*D)" and left the line's trailing ", oldest -> newest"
dangling after the inserted block, which parsed as a syntax error.
"""

import ast

P = "/workspace/train_window.py"
NEW = '''class _WindowView:
    """Lazy (N, F*D) window over a (N, D) cache: gathers only what is indexed.

    Replaces torch.cat([Z[i] for i in idxs], dim=1), which needs ~7x the base
    cache (F fancy-indexed copies held alongside the output). The TD sampler
    only ever gathers batch_size rows, so materializing all N windows is waste
    -- at DINO-WM width, 4.2 TB against 1.9 TB of RAM.
    """

    def __init__(self, base, index_list):
        self._Z = base
        self._idxs = index_list
        self.shape = (base.shape[0], base.shape[1] * len(index_list))
        self.dtype = base.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sel):
        # window row k of cache row r is self._idxs[k][r]; oldest -> newest
        return torch.cat([self._Z[i[sel]] for i in self._idxs], dim=-1)


Zs = _WindowView(Z, idxs)  # (N, F*D), oldest -> newest, gathered on demand
logging.info(
    f"lazy window: peak {Z.shape[0] * Z.shape[1] * 4 / 2**30:.0f} GB (base) vs "
    f"~{7 * Z.shape[0] * Z.shape[1] * 4 / 2**30:.0f} GB if materialized")'''

src = open(P).read()
if "_WindowView" in src:
    print("already patched")
    raise SystemExit

lines = src.split("\n")
hit = [i for i, l in enumerate(lines) if l.startswith("Zs = torch.cat(")]
assert len(hit) == 1, f"expected one stacking line, found {hit}"
print("replacing:", repr(lines[hit[0]]))
lines[hit[0]] = NEW
open(P, "w").write("\n".join(lines))
ast.parse(open(P).read())
print("train_window.py: window gathered per batch; syntax OK")
