"""RLP -- learned iterative planning over latent world models.

Layout:
    * :mod:`rlp.core`    -- the RL loop: policy and planning (``solver``,
      ``planner``), value functions (``value``), world models
      (``world_model``), and the shared unroll (``rollout``).
    * :mod:`rlp.data`     -- frozen-latent caching.
    * :mod:`rlp.train`    -- model, metric, and planner training programs.
    * :mod:`rlp.eval`     -- evaluation programs.
    * :mod:`rlp.tools`    -- one-off data builders and diagnostic probes.

The framework this builds on (environments, datasets, the CEM/MPPI solver family,
the LeWM/PLDM world models) is the pinned, installed ``stable_worldmodel`` package.
Campaign-specific behavior lives beside the RLP subsystem that owns it.
"""

__version__ = "0.1.0"
