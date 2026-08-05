#!/usr/bin/env python
"""Verify the OGBench cube checkpoints in the handoff bundle (single source of truth).

Canonical location (only copy of the weights):
    checkpoints/ogbench_cube_full_handoff_20260701T080809Z/models/<name>/

Loads every model through the bundle's OWN code snapshot (as the handoff README
prescribes) — no config patching needed, bundle stays pristine/checksummed.

Run:  stable-worldmodel/.venv/bin/python stable-worldmodel/checkpoints/register.py

NOTE for loading in experiments:
- via bundled code:  PYTHONPATH=<bundle>/code/stable-worldmodel  ->  load_pretrained(<abs model folder>)
- via OUR repo:      LeWM loads directly; PreJEPA/DINO needs the config target
  'stable_worldmodel.wm.prejepa.CausalPredictor' -> '...prejepa.module.CausalPredictor'
  (pass via a patched copy of config.json or extra_args; do NOT edit the bundle in place).
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "ogbench_cube_full_handoff_20260701T080809Z")
SNAP = os.path.join(BUNDLE, "code", "stable-worldmodel")
SMOKE = os.path.join(BUNDLE, "scripts", "load_all_models_smoke_test.py")

env = dict(os.environ, PYTHONPATH=SNAP + os.pathsep + os.environ.get("PYTHONPATH", ""))
print(f"bundle : {BUNDLE}\nusing bundled code snapshot on PYTHONPATH\n")
r = subprocess.run([sys.executable, SMOKE], cwd=BUNDLE, env=env)
sys.exit(r.returncode)
