"""Pick the tasks worth exporting from a paired PRE/POST eval.

Prints: gains (PRE fail -> POST success), regressions, and two controls
(both-succeed, both-fail) so the folder is not cherry-picked.
"""
import re, sys
from pathlib import Path

def succ(p):
    m = re.search(r"episode_successes.: array\(\[(.*?)\]\)", Path(p).read_text(), re.S)
    return [x.strip() == "True" for x in m.group(1).replace("\n", " ").split(",")]

pre, post = succ(sys.argv[1]), succ(sys.argv[2])
n = min(len(pre), len(post))
gain = [i for i in range(n) if not pre[i] and post[i]]
lose = [i for i in range(n) if pre[i] and not post[i]]
both_ok = [i for i in range(n) if pre[i] and post[i]]
both_no = [i for i in range(n) if not pre[i] and not post[i]]
print("PRE  %.1f  POST %.1f  n=%d" % (100*sum(pre)/n, 100*sum(post)/n, n))
print("gain=" + ",".join(map(str, gain)))
print("lose=" + ",".join(map(str, lose)))
print("both_ok=" + ",".join(map(str, both_ok[:2])))
print("both_fail=" + ",".join(map(str, both_no[:2])))
print("EXPORT=" + ",".join(map(str, gain + lose + both_ok[:2] + both_no[:2])))
