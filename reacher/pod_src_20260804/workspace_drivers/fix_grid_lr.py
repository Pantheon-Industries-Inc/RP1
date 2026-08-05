"""Fix run_gridB.sh's actor-lr-final computation.

The original line nested single quotes inside a double-quoted bash command
substitution, so the shell consumed them and every training got
`--actor-lr-final ''` -> argparse rejected it and all 56 grid trainings died
within a second. Replaced with a plain case statement (the three sweep values
are known) plus an awk fallback.
"""

P = "/workspace/run_gridB.sh"
s = open(P).read()
if "case \"$lr\" in" in s:
    print("already fixed")
    raise SystemExit

import re

m = re.search(r"  local lrf\n  lrf=\$\(python3 -c [^\n]*\n", s)
assert m, "lrf anchor not found"
NEW = (
    "  local lrf\n"
    "  case \"$lr\" in\n"
    "    1e-4) lrf=1e-5 ;;\n"
    "    3e-4) lrf=3e-5 ;;\n"
    "    1e-3) lrf=1e-4 ;;\n"
    "    *) lrf=$(awk \"BEGIN{printf \\\"%.0e\\\", $lr/10}\") ;;\n"
    "  esac\n"
)
s = s[: m.start()] + NEW + s[m.end():]
open(P, "w").write(s)
print("fixed actor-lr-final computation")
