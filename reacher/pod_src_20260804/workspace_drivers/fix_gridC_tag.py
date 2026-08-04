"""Fix run_gridC.sh's tag(): the LR component was dropping out, collapsing 36
configs onto 18 filenames so the 1e-4 and 3e-4 arms overwrote each other.

Cause: `$(echo $3 | tr -d 'e-')` inside the function tripped `set -u` ("$3:
unbound variable") and expanded to nothing. Replaced with pure parameter
expansion, which needs no subshell and cannot lose the positional args.
"""

import ast  # noqa: F401  (kept for symmetry with the other patch scripts)

P = "/workspace/run_gridC.sh"
s = open(P).read()
OLD = '''tag(){ # sched steps lr
  local sc=$1; case $sc in uniform) sc=u;; geom-early) sc=ge;; geom-late) sc=gl;; esac
  echo "${sc}st$(( $2 / 1000 ))k$(echo $3 | tr -d 'e-')"
}'''
NEW = '''tag(){ # sched steps lr  -- pure parameter expansion: no subshell, no set -u trap
  local sc="$1" st="$2" lr="$3"
  case "$sc" in uniform) sc=u ;; geom-early) sc=ge ;; geom-late) sc=gl ;; esac
  local lrs="${lr//e-/e}"          # 1e-4 -> 1e4, keeps the arms distinct
  echo "${sc}st$(( st / 1000 ))k${lrs}"
}'''
assert s.count(OLD) == 1, f"tag anchor x{s.count(OLD)}"
open(P, "w").write(s.replace(OLD, NEW))
print("fixed tag(); distinct tags now:")

# show what the 12 tags will be
import subprocess

for sc in ("uniform", "geom-early", "geom-late"):
    for st in (4000, 8000):
        for lr in ("1e-4", "3e-4"):
            out = subprocess.run(
                ["bash", "-c", f'source <(sed -n "/^tag(){{/,/^}}/p" {P}); tag {sc} {st} {lr}'],
                capture_output=True, text=True).stdout.strip()
            print(f"  {sc:<11} {st:<5} {lr:<5} -> {out}")
