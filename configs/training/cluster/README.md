# Cluster specs

SkyPilot task specs rather than Hydra configs; the one exception to `configs/` mirroring `src/`.
Launch them from the `cluster` environment:

```bash
pixi run -e cluster sky jobs launch configs/training/cluster/replicate_cube.yaml --env BASE=lewm -y
```

Before the first launch, point the spec at your infrastructure:

- `resources` names the accelerator and container image; adjust them to what your cloud offers.
- `volumes` mounts a persistent volume at `VOLUME_ROOT` for caches and checkpoints. Create it with
  `sky volumes apply`, or drop the mount and set `VOLUME_ROOT` to a local path when jobs need not
  survive preemption.
- The `cluster` environment installs SkyPilot's Kubernetes support; add the extra for your cloud
  (`skypilot[aws]`, `skypilot[gcp]`, ...) to the `cluster` feature in `pixi.toml`.
