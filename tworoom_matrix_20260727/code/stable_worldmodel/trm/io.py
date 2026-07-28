"""Save / load trained TRM metric modules (regression / TD / contrastive).

Also hosts the **latent compressor** used to make a learned value trainable on
DINO-style *flat patch-token* bases.

Why a compressor at all
-----------------------
``DinoWMTokens`` hands the planner all 196 patch tokens flat (196 x 384 =
75264-d). A full-dataset latent cache at that width would be ~600 GB, so the
value has to be trained on a capped cache (200k rows of 75264-d) -- a hopeless
data/dim ratio, and empirically the TD value collapses (TD+CEM 22.0) even though
the same world model plans fine with a plain L2 cost (CEM 72.0/94.0).

Compressing the latent *before* the value head fixes the ratio: 2M rows at
1024-d is ~8 GB. Which compressor matters:

* ``mean``      -- average the 196 patches -> 384-d. Cheap, but largely
  **position-blind**: the same arm appearance at a different image location
  averages to nearly the same vector, and position is exactly what a reacher
  value needs.
* ``rp<D>``     -- fixed Gaussian **random projection** 75264 -> D with entries
  ~ N(0, 1/D). By Johnson-Lindenstrauss this approximately *preserves the L2
  geometry of the full flat vector* -- i.e. the very distances plain CEM is
  already using successfully -- while being fitting-free, linear (so LIP's
  value-gradient still flows) and cheap to store.
* ``spatial<k>`` -- mean-pool the 14x14 patch grid into a kxk grid ->
  ``k*k*384`` dims. Keeps coarse position; the structural middle ground.

All modes are wrapped by :class:`CompressedMetric`, which **dispatches on the
last dim**: training rows arrive already compressed, the planner passes full
flat tokens. Every mode is differentiable, so value-gradient methods still work.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import torch
from torch import nn

from .head import PairwiseMetricHead, QuasimetricHead
from .learners.contrastive import ContrastiveCritic

DEFAULT_TOKEN_DIM = 384
DEFAULT_PATCHES = 196


def parse_compress(spec: str) -> tuple[str, int | None]:
    """``'mean' -> ('mean', None)``, ``'rp1024' -> ('rp', 1024)``, ``'spatial2' -> ('spatial', 2)``."""
    spec = str(spec).strip().lower()
    if spec in ("mean", "pool", "meanpool"):
        return "mean", None
    m = re.fullmatch(r"rp(\d+)", spec)
    if m:
        return "rp", int(m.group(1))
    m = re.fullmatch(r"spatial(\d+)", spec)
    if m:
        return "spatial", int(m.group(1))
    raise ValueError(f"unknown --compress spec '{spec}' (want mean | rp<D> | spatial<k>)")


def compressed_dim(spec: str, token_dim: int = DEFAULT_TOKEN_DIM) -> int:
    """Output width of a compressor spec."""
    mode, param = parse_compress(spec)
    if mode == "mean":
        return int(token_dim)
    if mode == "rp":
        return int(param)
    return int(param) * int(param) * int(token_dim)


def make_projection(full_dim: int, out_dim: int, seed: int = 0) -> torch.Tensor:
    """Fixed Gaussian random-projection matrix ``(full_dim, out_dim)``.

    Entries ~ N(0, 1/out_dim) (i.e. standard normal / sqrt(out_dim)), so
    ``||x P - y P|| ~= ||x - y||`` in expectation (Johnson-Lindenstrauss).
    Generated from an explicit CPU generator so cache-time and inference-time
    matrices are bit-identical; the matrix is *also* persisted in the metric blob
    (registered buffer) so loading never depends on RNG reproducibility.
    """
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(int(full_dim), int(out_dim), generator=g, dtype=torch.float32) / math.sqrt(int(out_dim))


def projection_sha256(mat: torch.Tensor) -> str:
    return hashlib.sha256(mat.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class CompressedMetric(nn.Module):
    """Compress a flat patch-token latent ``(..., full_dim)`` before the inner metric.

    Dispatches on the last dim: already-compressed input (``out_dim``, the
    training cache) passes straight through; full flat tokens (``full_dim``, the
    planner) are compressed first. All compressors are differentiable.

    Args:
        inner: metric head consuming the compressed ``out_dim`` latent.
        compress: spec string -- ``mean`` | ``rp<D>`` | ``spatial<k>``.
        full_dim: uncompressed latent width (e.g. 75264).
        out_dim: compressed width (must equal the inner head's input dim).
        patches: number of patch tokens ``P`` (mean/spatial).
        token_dim: per-token width ``d`` (mean/spatial).
        seed: RNG seed for ``rp`` (recorded so the matrix is reproducible).
    """

    def __init__(
        self,
        inner: nn.Module,
        compress: str,
        full_dim: int,
        out_dim: int,
        patches: int | None = DEFAULT_PATCHES,
        token_dim: int | None = DEFAULT_TOKEN_DIM,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.compress = str(compress)
        self.mode, self.param = parse_compress(compress)
        self.full_dim = int(full_dim)
        self.out_dim = int(out_dim)
        self.patches = int(patches) if patches else None
        self.token_dim = int(token_dim) if token_dim else None
        self.seed = int(seed)
        assert self.out_dim != self.full_dim, "compressor must change the width (dispatch is by dim)"
        if self.mode == "rp":
            self.register_buffer("proj", make_projection(self.full_dim, self.out_dim, self.seed))
        if self.mode in ("mean", "spatial"):
            assert self.patches and self.token_dim, f"{self.mode} needs patches/token_dim"
            assert self.patches * self.token_dim == self.full_dim, (
                f"patches*token_dim ({self.patches}*{self.token_dim}) != full_dim {self.full_dim}"
            )

    # -- compression ------------------------------------------------------
    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.out_dim:  # already compressed (training cache)
            return x
        assert x.shape[-1] == self.full_dim, (
            f"CompressedMetric[{self.compress}] expects last dim {self.out_dim} "
            f"(compressed) or {self.full_dim} (flat tokens), got {tuple(x.shape)}"
        )
        if self.mode == "rp":
            return x @ self.proj.to(dtype=x.dtype)
        P, d = self.patches, self.token_dim
        if self.mode == "mean":
            return x.reshape(*x.shape[:-1], P, d).mean(-2)
        # spatial<k>: 14x14 patch grid -> kxk grid of block means
        g = int(round(math.sqrt(P)))
        assert g * g == P, f"{P} patches is not a square grid"
        k = self.param
        assert g % k == 0, f"grid {g} not divisible by k={k}"
        b = g // k
        t = x.reshape(*x.shape[:-1], k, b, k, b, d).mean(dim=(-4, -2))  # (..., k, k, d)
        return t.reshape(*x.shape[:-1], k * k * d)

    # -- metric API (transparent over the inner head) ---------------------
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.inner(self._compress(x), self._compress(y))

    def cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.inner.cost(self._compress(x), self._compress(y))

    def __getattr__(self, name):
        # nn.Module.__getattr__ resolves params/buffers/submodules; anything else
        # a caller expects (``latent_dim``, ``enc``, ``sym_dim``, ...) is
        # delegated to the wrapped head so the wrapper is API-transparent.
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("_"):
                raise
            inner = self.__dict__.get("_modules", {}).get("inner")
            if inner is None:
                raise
            return getattr(inner, name)

    def extra_repr(self) -> str:
        return f"compress={self.compress}, {self.full_dim} -> {self.out_dim}"


# Backwards-compatible name: the original mean-pool-only wrapper.
PatchPooledMetric = CompressedMetric


def compress_arch(spec: str, full_dim: int, out_dim: int,
                  patches: int | None = DEFAULT_PATCHES,
                  token_dim: int | None = DEFAULT_TOKEN_DIM,
                  seed: int = 0) -> dict:
    """Arch-dict fragment that lets :func:`build_metric` rebuild the wrapper."""
    return {"compress": str(spec), "compress_full_dim": int(full_dim),
            "compress_out_dim": int(out_dim),
            "compress_patches": int(patches) if patches else None,
            "compress_token_dim": int(token_dim) if token_dim else None,
            "compress_seed": int(seed)}


def save_metric(module: nn.Module, learner: str, latent_dim: int, arch: dict, path: str | Path) -> None:
    """Persist a metric module with the metadata needed to rebuild it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"learner": learner, "latent_dim": latent_dim, "arch": arch,
         "state_dict": module.state_dict()},
        path,
    )


def _build_inner(learner: str, latent_dim: int, arch: dict) -> nn.Module:
    if learner == "dwell":
        # discounted-dwell value: unconstrained scalar + a lower-is-better
        # cost adapter, not a metric (a discounted return has no triangle
        # inequality, so it must not be rebuilt as a quasimetric head)
        from .learners.dwell import DwellValue

        return DwellValue(
            latent_dim,
            hidden_dim=arch.get("hidden_dim", 256),
            depth=arch.get("depth", 2),
            gamma=arch.get("gamma", 0.98),
        )
    if learner in ("regression", "td", "shuffled"):
        if arch.get("head") == "quasimetric":
            return QuasimetricHead(
                latent_dim,
                hidden_dim=arch.get("hidden_dim", 256),
                embed_dim=arch.get("embed_dim", 128),
                depth=arch.get("depth", 2),
            )
        return PairwiseMetricHead(
            latent_dim,
            hidden_dim=arch.get("hidden_dim", 256),
            depth=arch.get("depth", 2),
            softplus=arch.get("softplus", True),
            symmetric=arch.get("symmetric", False),
        )
    if learner == "contrastive":
        return ContrastiveCritic(
            latent_dim,
            hidden_dim=arch.get("hidden_dim", 256),
            rep_dim=arch.get("rep_dim", 64),
            depth=arch.get("depth", 2),
        )
    raise ValueError(f"unknown learner '{learner}'")


def build_metric(learner: str, latent_dim: int, arch: dict) -> nn.Module:
    """Construct an (untrained) metric module from architecture metadata.

    ``latent_dim`` is always the width the *head* consumes. With a compressor
    recorded in ``arch`` the head is wrapped in :class:`CompressedMetric` so it
    also accepts full flat patch-token latents at plan time.
    """
    inner = _build_inner(learner, latent_dim, arch)
    if not isinstance(arch, dict):
        return inner
    spec = arch.get("compress")
    if spec:
        return CompressedMetric(
            inner, spec,
            full_dim=arch["compress_full_dim"],
            out_dim=arch.get("compress_out_dim", latent_dim),
            patches=arch.get("compress_patches", DEFAULT_PATCHES),
            token_dim=arch.get("compress_token_dim", DEFAULT_TOKEN_DIM),
            seed=arch.get("compress_seed", 0),
        )
    pool_patches = arch.get("pool_patches")  # legacy mean-pool blobs
    if pool_patches:
        return CompressedMetric(
            inner, "mean", full_dim=int(pool_patches) * latent_dim, out_dim=latent_dim,
            patches=int(pool_patches), token_dim=latent_dim,
        )
    return inner


def load_metric(path: str | Path, device: str = "cpu") -> nn.Module:
    """Load a trained metric module saved by :func:`save_metric`."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    module = build_metric(blob["learner"], blob["latent_dim"], blob["arch"])
    try:
        module.load_state_dict(blob["state_dict"])
    except RuntimeError:
        # tolerate blobs whose state_dict was saved from the bare inner head
        # before it was wrapped (no 'inner.' prefix)
        if not isinstance(module, CompressedMetric):
            raise
        module.inner.load_state_dict(blob["state_dict"])
    return module.to(device).eval()


__all__ = ["save_metric", "build_metric", "load_metric", "CompressedMetric",
           "PatchPooledMetric", "parse_compress", "compressed_dim",
           "make_projection", "projection_sha256", "compress_arch"]
