# Stable-WorldModel compatibility

rp1 depends on the published `stable-worldmodel==0.1.1` package. The version is
pinned because rp1's planning and checkpoint adapters target that release's
public module layout and runtime contracts.

The vendored Stable-WM fork has been removed. rp1-owned behavior now lives in
small adapters instead:

- `rp1.environment.World` preserves dataset-backed resets, goal snapshots,
  recording, and image resizing used by the evaluation campaigns.
- `rp1.core.value.LatentGoalCost` corrects the candidate-axis broadcasting in
  the 0.1.1 LeWM/PLDM terminal cost.
- `rp1.core.world_model.DinoWMTokens` preserves the released patch-token DINO
  checkpoint contract.
- rp1's rp1, Dyna, and TRM implementations remain local research algorithms;
  they are not copies of published Stable-WM modules.

## Why upstream `main` is not a drop-in upgrade

The comparison baseline was upstream commit
`73dade035ff789e007194971ca5a59b3c3f77e6b` (2026-07-29). It contains useful
changes, but also incompatible API and behavior changes:

- solver modules moved under `stable_worldmodel.planning` and constructors now
  receive a cost object rather than a model;
- `World` step callbacks and dataset reset extraction changed shape/masking
  behavior;
- LeWM prediction/action-history semantics changed to future-only predictions;
- checkpoint saving no longer needs the wrapper used by 0.1.1; and
- its training stack requires `stable-pretraining>=0.1.8`, while the released
  checkpoints and this project are validated with 0.1.7.

An upgrade should therefore be treated as a migration: update the planning
adapter, revalidate dataset reset and callback behavior, migrate checkpoint
loading, and rerun latent/rollout parity tests before changing the pin.
