# OGBench Cube Full World Model Handoff

This bundle contains six trained Stable World Model checkpoints for the OGBench visual cube tasks.

Run stamp: `20260701T080809Z`  
Source repo commit: `38eb3a5a77d443dd421f8297f6f9926ee0f78dea`  
Project: `yash21-dagade-duke-university/pantheon-swm`

This is the full sharing bundle. It includes:

- final model weights,
- model `config.json` files required by the loader,
- original training `config.yaml` files,
- a copy of the exact `stable-worldmodel` source tree used for this handoff,
- local source changes as `code/LOCAL_CHANGES.patch`,
- Python environment metadata,
- eval summaries,
- a smoke-test script,
- checksums.

## What To Give Someone

Give them each model folder, not only the `.pt` file.

The `.pt` file contains learned weights. The sibling `config.json` is needed by
`stable_worldmodel.wm.utils.load_pretrained(...)` to reconstruct the model architecture.
The `training_config.yaml` is not required for loading weights, but it records the dataset,
trainer settings, run name, and other provenance needed to reproduce the run.

## Contents

```text
models/
  ogbench_cube_single_dino/
    weights_step_100000.pt
    config.json
    training_config.yaml
  ogbench_cube_double_dino/
    weights_step_100000.pt
    config.json
    training_config.yaml
  ogbench_cube_triple_dino/
    weights_step_100000.pt
    config.json
    training_config.yaml
  ogbench_cube_single_lewm/
    weights_step_40000.pt
    config.json
    training_config.yaml
  ogbench_cube_double_lewm/
    weights_step_40000.pt
    config.json
    training_config.yaml
  ogbench_cube_triple_lewm/
    weights_step_40000.pt
    config.json
    training_config.yaml
```

Additional files:

- `EVAL_SUMMARY.md`: planning-eval summary available at packaging time.
- `eval_results.csv`: machine-readable eval summary available at packaging time.
- `SHA256SUMS`: checksums for all files in this bundle.
- `code/stable-worldmodel/`: source snapshot for loading, inference, planning eval, or fine-tuning.
- `code/PIP_FREEZE.txt`: Python packages from the environment used to package/test this.
- `scripts/load_all_models_smoke_test.py`: imports the codebase and verifies all six checkpoints load.

## Recommended First Test

From the extracted bundle root:

```bash
export PYTHONPATH="$PWD/code/stable-worldmodel:${PYTHONPATH:-}"
python scripts/load_all_models_smoke_test.py
```

Expected behavior: it prints one `OK ...` line for each of the six checkpoints.

If dependencies are missing, create an environment from `code/stable-worldmodel/pyproject.toml`
and use `code/PIP_FREEZE.txt` as the exact package reference from this run.

## Model Table

| Task | Model | Checkpoint |
| --- | --- | --- |
| cube single | DINO / PreJEPA-style world model | `models/ogbench_cube_single_dino/weights_step_100000.pt` |
| cube double | DINO / PreJEPA-style world model | `models/ogbench_cube_double_dino/weights_step_100000.pt` |
| cube triple | DINO / PreJEPA-style world model | `models/ogbench_cube_triple_dino/weights_step_100000.pt` |
| cube single | LeWM | `models/ogbench_cube_single_lewm/weights_step_40000.pt` |
| cube double | LeWM | `models/ogbench_cube_double_lewm/weights_step_40000.pt` |
| cube triple | LeWM | `models/ogbench_cube_triple_lewm/weights_step_40000.pt` |

## Loading A Model

Use the bundled `code/stable-worldmodel` codebase or an equivalent compatible checkout.

The loader supports either a path to a `.pt` file or a folder containing one `.pt`
file plus `config.json`.

```python
import torch
import stable_worldmodel as swm

ckpt = "/path/to/ogbench_cube_model_handoff_20260701T080809Z/models/ogbench_cube_single_dino/weights_step_100000.pt"

model = swm.wm.utils.load_pretrained(ckpt)
model = model.to("cuda").eval()
model.requires_grad_(False)
```

Folder loading is also valid:

```python
model = swm.wm.utils.load_pretrained(
    "/path/to/ogbench_cube_model_handoff_20260701T080809Z/models/ogbench_cube_single_lewm"
)
```

## Running Planning Eval

For the original evaluation path, use `scripts/plan/eval_wm.py` from the
Stable World Model repo and pass the checkpoint with `policy=...`.

Example shape:

```bash
cd /path/to/stable-worldmodel
export MUJOCO_GL=osmesa

python scripts/plan/eval_wm.py \
  --config-dir /path/to/eval/configs \
  --config-name cube_single \
  policy=/path/to/ogbench_cube_model_handoff_20260701T080809Z/models/ogbench_cube_single_dino/weights_step_100000.pt \
  solver=cem \
  output.filename=single_dino_cem_eval.txt \
  seed=42
```

The eval config must match the task variant:

- `cube_single` for `ogbench_cube_single_*`
- `cube_double` for `ogbench_cube_double_*`
- `cube_triple` for `ogbench_cube_triple_*`

The solver names used in this repo are:

- `cem`: CEM planning
- `adam`: gradient-descent-style planning via `stable_worldmodel.solver.GradientSolver`

## Practical Notes

- Keep `config.json` next to the checkpoint file. Moving only the `.pt` file will break the default loader.
- DINO checkpoints here are final `weights_step_100000.pt`.
- LeWM checkpoints here are final `weights_step_40000.pt`.
- These are world models for planning/inference, not standalone behavior policies. To act in an environment, instantiate a planner/policy around the world model, as done by `scripts/plan/eval_wm.py`.
- The OGBench cube environment, MuJoCo, Stable World Model code, and matching Python dependencies are required to reproduce planning behavior.
- A PyTorch state dict is not completely codebase-agnostic. The safest path is to use the bundled code snapshot or port the model classes/configs into another codebase deliberately.
- Fine-tuning requires the matching datasets and training scripts/configs. This bundle includes the model side and training configs, but not the original OGBench HDF5 datasets.

## Verify Bundle Integrity

From this folder:

```bash
shasum -a 256 -c SHA256SUMS
```
