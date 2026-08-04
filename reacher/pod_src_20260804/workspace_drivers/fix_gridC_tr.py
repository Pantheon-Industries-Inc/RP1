"""Fix run_gridC.sh: the training helper was named `tr`, shadowing the /usr/bin/tr
that tag() used internally.

`tag()` built its label with `$(echo $3 | tr -d 'e-')`. Because a shell function
named `tr` was in scope, that call re-entered the TRAINING function with args
`-d` and `e-`, which under `set -u` died on `$3: unbound variable` and returned
an empty string. Result: the learning-rate component vanished from every tag, so
the 1e-4 and 3e-4 arms shared filenames -- 36 trainings writing to 18 paths and
overwriting each other.

Two independent fixes so this cannot recur:
  1. rename the helper `tr` -> `train_cell`;
  2. build the label with parameter expansion instead of the `tr` utility.
"""

import re

P = "/workspace/run_gridC.sh"
s = open(P).read()

# 1. label without calling any external command
OLD_TAG = re.search(r"tag\(\)\{.*?\n\}", s, re.S)
assert OLD_TAG, "tag() not found"
NEW_TAG = '''tag(){ # sched steps lr  -- pure parameter expansion: no subshell, no `tr`
  local sc="$1" st="$2" lr="$3"
  case "$sc" in uniform) sc=u ;; geom-early) sc=ge ;; geom-late) sc=gl ;; esac
  local lrs="${lr//e-/e}"          # 1e-4 -> 1e4, keeps the arms distinct
  echo "${sc}st$(( st / 1000 ))k${lrs}"
}'''
s = s[: OLD_TAG.start()] + NEW_TAG + s[OLD_TAG.end():]

# 2. rename the helper so it can never shadow a utility again
s = re.sub(r"^tr\(\)\{", "train_cell(){", s, flags=re.M)
s = re.sub(r"^(\s*)tr (\"\$sc\")", r"\1train_cell \2", s, flags=re.M)
s = s.replace('log "train ${t} s${seed}: exists"', 'log "train ${t} s${seed}: exists"')

open(P, "w").write(s)
n_tr = len(re.findall(r"(?<![a-z_])tr ", s))
print(f"patched: tag() uses parameter expansion; helper renamed to train_cell "
      f"(bare `tr ` occurrences left: {n_tr})")
