#!/usr/bin/env python3
"""Flatten a recipe yaml into shell exports: section.key -> UNI_<KEY>=value.

    eval "$(python3 scripts/sky/unigamma/recipes/recipe_env.py scripts/sky/unigamma/recipes/uniJ.yaml)"
Keys are unique across sections by construction (asserted). Numbers are printed as
written in the yaml where possible (3e-4 stays 3e-4) so the task yamls see the same
literals a human would type.
"""
import re, shlex, sys
import yaml

path = sys.argv[1]
raw = open(path).read()
doc = yaml.safe_load(raw)
seen = {}
for section, body in doc.items():
    if not isinstance(body, dict):
        continue
    for key, val in body.items():
        name = "UNI_" + key.upper()
        assert name not in seen, f"duplicate recipe key {key} in {section} and {seen[name]}"
        seen[name] = section
        # keep the literal spelling from the file (e.g. 3e-4, 1e-3) rather than python's repr
        m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(\"[^\"]*\"|'[^']*'|[^#\n]*?)\s*(#.*)?$", raw, flags=re.M)
        lit = m.group(1).strip().strip("\"'") if m else str(val)
        print(f"export {name}={shlex.quote(lit)}")
