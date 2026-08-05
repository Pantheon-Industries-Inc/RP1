"""Load latent caches lazily (mmap) instead of reading them fully into RAM.

WHY. The DINO-WM cache is 2,010,000 x 75,264 fp32 = 605 GB. Every consumer
currently pays a full torch.load of it before a single training step:

    LatentCache.load  ->  torch.load(path, map_location="cpu")

On this volume that is ~50 min per process, and there are several consumers
(the value trainer, then each RLP seed, then PWM). mmap=True makes the tensor
file-backed, so pages are faulted in only as rows are actually gathered. With
2 TB of RAM the page cache then holds the whole 605 GB after the first pass, so
steady-state speed matches the in-RAM case while the up-front cost disappears
and no process ever allocates 605 GB of its own.

THE HAZARD THIS CHECKS. The loader does d["z"].float(). Tensor.float() is
self.to(torch.float32), which returns *self* when the dtype already matches but
COPIES otherwise -- and a copy would silently defeat mmap, reintroducing the
full 605 GB allocation with no error. So the patch verifies at load time that
the returned tensor is still the file-backed one, and says so, rather than
assuming the no-op.

Read-only is safe here: train_window indexes rows, train_lip_ac's _wrow/_wpair
gather rows, and the samplers index z. Nothing mutates the cache in place.

Falls back to a normal load if mmap is unavailable (e.g. a cache written with
the legacy non-zipfile serialization), so older caches keep working.
"""

import ast

P = "/workspace/swm_cem/stable_worldmodel/trm/latent_cache.py"
s = open(P).read()

if "mmap=True" in s:
    print("already patched")
    raise SystemExit

OLD = '''    @classmethod
    def load(cls, path: str | Path) -> "LatentCache":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls('''

NEW = '''    @classmethod
    def load(cls, path: str | Path, mmap: bool = True) -> "LatentCache":
        # mmap: page the latents in on demand instead of reading the whole file.
        # At DINO-WM width the cache is 605 GB and every consumer would otherwise
        # pay a ~50 min read before its first step. With enough RAM the page
        # cache holds it after one pass, so steady-state speed is unchanged.
        try:
            d = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
        except (TypeError, RuntimeError):
            # legacy (non-zipfile) checkpoint, or a torch without mmap support
            d = torch.load(path, map_location="cpu", weights_only=False)
        _z = d["z"]
        _zf = _z.float()
        if mmap and _zf.data_ptr() != _z.data_ptr():
            # .float() copied, so the mmap was defeated and the full tensor is
            # now resident. Report it -- silently allocating 605 GB is exactly
            # the failure this patch exists to prevent.
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "LatentCache.load: stored dtype %s != float32, so .float() "
                "materialized a %.0f GB copy and mmap had no effect",
                _z.dtype, _z.numel() * 4 / 2**30)
        return cls('''

assert s.count(OLD) == 1, f"load anchor x{s.count(OLD)}"
s = s.replace(OLD, NEW)
s = s.replace('            z=d["z"].float(),', '            z=_zf,', 1)

open(P, "w").write(s)
ast.parse(open(P).read())
print("LatentCache.load: mmap=True by default, with copy detection; syntax OK")
