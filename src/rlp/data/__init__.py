"""Dataset-side artifacts shared by the RL core, trainers, and probes.

Currently just the frozen-latent cache: encode a logged dataset once with a
frozen world model, persist ``(z, episode_idx, step_idx[, state])``, and iterate
it cheaply thereafter. Deliberately independent of :mod:`rlp.core` — a probe
that only wants to read a cache should not import the core package.

Public surface:
    * ``LatentCache``    -- the cache container (``save``/``load``/``episodes``).
    * ``encode_dataset`` -- build one from a stable-worldmodel dataset.
"""

from .latent_cache import LatentCache, encode_dataset
from .registry import DATASETS, DatasetSpec, data_home, dataset_path, get_dataset_spec

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "LatentCache",
    "data_home",
    "dataset_path",
    "encode_dataset",
    "get_dataset_spec",
]
