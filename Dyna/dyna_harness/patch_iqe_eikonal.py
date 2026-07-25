"""Add (1) an IQE quasimetric head and (2) an Eikonal unit-gradient regularizer
to the TRM value learner. Idempotent; anchored replacements fail loudly.

(1) IQE — Interval Quasimetric Embedding (Wang & Isola, arXiv 2211.15120), the
    head used by QRL. Embeddings are reshaped to (k components x d dims); the
    component distance is the Lebesgue measure of the union of intervals
    [u_ij, v_ij] (empty when u >= v), aggregated over components by
    maxmean = a*max + (1-a)*mean with learnable a. Unlike MRN it has no
    symmetric Euclidean term, so it does not inherit high-dim distance
    concentration -- the suspected cause of our far-range compression.

(2) Eikonal regularizer (Eik-HIQL, NeurIPS 2025, arXiv 2509.06782): the Eikonal
    PDE ||grad_s V(s,g)|| = 1 says the value must change at unit rate per unit of
    real motion. Our latents are not unit-speed, so we scale by the dataset's
    mean per-step latent displacement ds: penalise (||grad_z V|| * ds - 1)^2,
    i.e. "V changes by ~1 per ENV STEP". Unlike the TD target (which only
    constrains V on logged transitions) the gradient penalty also shapes V
    off-manifold -- where the planner actually goes.
"""
from pathlib import Path

TRM = Path("/workspace/code/stable-worldmodel/stable_worldmodel/trm")
SCRIPTS = Path("/workspace/code/stable-worldmodel/scripts/plan")

# ---------------------------------------------------------------- 1. head.py
head = TRM / "head.py"
t = head.read_text()
if "class IQEHead" in t:
    print("head.py: already patched")
else:
    anchor = '__all__ = ["PairwiseMetricHead", "QuasimetricHead", "pair_features"]'
    assert t.count(anchor) == 1, "head.py __all__ anchor"
    iqe_src = '''

def _interval_union_length(s: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Lebesgue measure of the union of intervals ``[s_j, e_j]`` along the last
    dim (an interval is empty when ``e_j <= s_j``).

    Sort by start, then sweep with an exclusive cumulative max of the ends: each
    interval contributes only the part beyond every earlier interval's reach.
    Fully vectorised and differentiable (argsort is a permutation; gradients
    flow through gather/cummax to the right elements).
    """
    order = s.argsort(dim=-1)
    s_ = s.gather(-1, order)
    e_ = e.gather(-1, order)
    cm = e_.cummax(dim=-1).values                     # inclusive
    neg_inf = torch.full_like(cm[..., :1], float("-inf"))
    prev = torch.cat([neg_inf, cm[..., :-1]], dim=-1)  # exclusive
    lo = torch.maximum(s_, prev)
    return (e_ - lo).clamp_min(0).sum(-1)


class IQEHead(nn.Module):
    """Interval Quasimetric Embedding head (Wang & Isola, 2022).

        d(z_i -> z_j) = scale * maxmean_k | union_j [ u(z_i)_kj , u(z_j)_kj ] |

    ``num_components`` k groups of ``embed_dim // k`` dims each. Asymmetric by
    construction (intervals only count where the target coordinate is larger),
    zero on the diagonal, and satisfies the triangle inequality -- so it still
    stitches, but without MRN's symmetric Euclidean term.
    """

    def __init__(self, latent_dim, hidden_dim=256, embed_dim=128, depth=2,
                 num_components=8, alpha_init=0.75):
        super().__init__()
        assert embed_dim % num_components == 0, "embed_dim must divide by num_components"
        self.latent_dim = latent_dim
        self.k = num_components
        self.d = embed_dim // num_components
        layers = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, embed_dim)]
        self.enc = nn.Sequential(*layers)
        a = float(min(max(alpha_init, 1e-4), 1 - 1e-4))
        self.alpha_raw = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(a)))))
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, z_i, z_j):
        ei = self.enc(z_i).unflatten(-1, (self.k, self.d))
        ej = self.enc(z_j).unflatten(-1, (self.k, self.d))
        comp = _interval_union_length(ei, ej)          # (..., k)
        a = torch.sigmoid(self.alpha_raw)
        agg = a * comp.max(dim=-1).values + (1.0 - a) * comp.mean(dim=-1)
        return self.log_scale.exp() * agg

    @torch.no_grad()
    def cost(self, z_pred, z_goal):
        return self.forward(z_pred, z_goal)

'''
    t = t.replace(anchor, iqe_src.rstrip("\n") + "\n\n\n" +
                  '__all__ = ["PairwiseMetricHead", "QuasimetricHead", "IQEHead", "pair_features"]')
    head.write_text(t)
    print("head.py: IQEHead added")

# ------------------------------------------------------------------ 2. io.py
io = TRM / "io.py"
t = io.read_text()
if "IQEHead" in t:
    print("io.py: already patched")
else:
    t = t.replace("from .head import PairwiseMetricHead, QuasimetricHead",
                  "from .head import IQEHead, PairwiseMetricHead, QuasimetricHead", 1)
    a = '        if arch.get("head") == "quasimetric":'
    assert t.count(a) == 1, "io.py head branch anchor"
    t = t.replace(a, '''        if arch.get("head") == "iqe":
            return IQEHead(
                latent_dim,
                hidden_dim=arch.get("hidden_dim", 256),
                embed_dim=arch.get("embed_dim", 128),
                depth=arch.get("depth", 2),
                num_components=arch.get("num_components", 8),
            )
        if arch.get("head") == "quasimetric":''')
    io.write_text(t)
    print("io.py: iqe branch added")

# ------------------------------------------------------------ 3. learners/td.py
td = TRM / "learners" / "td.py"
t = td.read_text()
if "eikonal_weight" in t:
    print("td.py: already patched")
else:
    t = t.replace("from ..head import PairwiseMetricHead, QuasimetricHead",
                  "from ..head import IQEHead, PairwiseMetricHead, QuasimetricHead", 1)
    a = "    huber_beta: float = 1.0"
    assert t.count(a) == 1, "td.py config anchor"
    t = t.replace(a, "    huber_beta: float = 1.0\n"
                     "    eikonal_weight: float = 0.0   # Eik-HIQL unit-gradient penalty (0 = off)\n"
                     "    num_components: int = 8       # iqe head only")

    a = '''def _make_head(cfg, latent_dim):
    if cfg.head == "quasimetric":'''
    assert t.count(a) == 1, "td.py _make_head anchor"
    t = t.replace(a, '''def _make_head(cfg, latent_dim):
    if cfg.head == "iqe":
        return IQEHead(latent_dim, hidden_dim=cfg.hidden_dim, embed_dim=cfg.embed_dim,
                       depth=cfg.depth, num_components=cfg.num_components)
    if cfg.head == "quasimetric":''')

    # mean per-step latent displacement, for scaling the eikonal target
    a = "    g = cfg.gamma\n    value.train()"
    assert t.count(a) == 1, "td.py pre-loop anchor"
    t = t.replace(a, '''    g = cfg.gamma

    step_norm = 1.0
    if cfg.eikonal_weight > 0:
        # mean ||z_{t+1} - z_t|| over consecutive rows within episodes: converts
        # the Eikonal target from "per unit latent" to "per ENV STEP".
        _eps = cache.episodes()
        _d, _n = 0.0, 0
        for _rws in list(_eps.values())[:200]:
            _r = torch.as_tensor(list(_rws))
            _zz = cache.z[_r].float()
            _d += (_zz[1:] - _zz[:-1]).norm(dim=-1).sum().item()
            _n += len(_r) - 1
        step_norm = max(_d / max(_n, 1), 1e-6)
        logging.info(f"eikonal: mean per-step latent displacement = {step_norm:.4f}")

    value.train()''')

    a = """        pred = value(z_t, z_g)
        loss = _expectile_loss(pred - tgt, cfg.expectile, cfg.huber_beta)"""
    assert t.count(a) == 1, "td.py loss anchor"
    t = t.replace(a, """        pred = value(z_t, z_g)
        loss = _expectile_loss(pred - tgt, cfg.expectile, cfg.huber_beta)
        if cfg.eikonal_weight > 0:
            # ||grad_z V(z, g)|| * step_norm == 1  <=>  V changes by ~1 per env step
            z_in = z_t.detach().requires_grad_(True)
            v_in = value(z_in, z_g)
            (grad_z,) = torch.autograd.grad(v_in.sum(), z_in, create_graph=True)
            gnorm = grad_z.norm(dim=-1)
            eik = ((gnorm * step_norm - 1.0) ** 2).mean()
            loss = loss + cfg.eikonal_weight * eik""")
    td.write_text(t)
    print("td.py: eikonal + iqe wired")

# -------------------------------------------------------- 4. train_metric.py
tm = SCRIPTS / "train_metric.py"
t = tm.read_text()
if "eikonal-weight" in t:
    print("train_metric.py: already patched")
else:
    a = 'p.add_argument("--head", choices=["mlp", "quasimetric"], default="quasimetric")'
    assert t.count(a) == 1, "train_metric head-choices anchor"
    t = t.replace(a, 'p.add_argument("--head", choices=["mlp", "quasimetric", "iqe"], default="quasimetric")\n'
                     '    p.add_argument("--eikonal-weight", type=float, default=0.0,\n'
                     '                   help="Eik-HIQL unit-gradient penalty weight (0 = off)")\n'
                     '    p.add_argument("--num-components", type=int, default=8, help="iqe head only")')
    # forward the two new knobs into TDConfig(...)
    a = "            head=args.head, hidden_dim=args.hidden_dim, depth=args.depth, embed_dim=args.embed_dim,"
    assert t.count(a) == 1, "train_metric TDConfig anchor"
    t = t.replace(a, a + "\n            eikonal_weight=args.eikonal_weight, num_components=args.num_components,")
    # persist num_components so load_metric can rebuild an iqe head
    a = '''        arch = {"head": args.head, "hidden_dim": args.hidden_dim, "depth": args.depth,
                "embed_dim": args.embed_dim, "softplus": True, "symmetric": args.symmetric}'''
    assert t.count(a) == 1, "train_metric arch anchor"
    t = t.replace(a, '''        arch = {"head": args.head, "hidden_dim": args.hidden_dim, "depth": args.depth,
                "embed_dim": args.embed_dim, "softplus": True, "symmetric": args.symmetric,
                "num_components": args.num_components}''')
    tm.write_text(t)
    print("train_metric.py: --head iqe / --eikonal-weight added")

print("PATCH_OK")
