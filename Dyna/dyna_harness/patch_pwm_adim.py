"""Patch train_pwm_ac.py's action-dim derivation, into a COPY.

THE BUG. train_pwm_ac.py line ~163 reads

    a_dim = int(getattr(c_act, 'action', torch.zeros(1, 10)).shape[-1])

i.e. it takes the action dim off the latent cache and falls back to a
HARDCODED 10 when the cache has no `action` field. The fs5 caches this
campaign builds have no such field -- LIP gets actions from a separate
--h5 -- so every cube PWM run silently built a 10-dim actor and died in the
world model's action encoder:

    Conv1d(25, 10, kernel_size=(1,))
    RuntimeError: expected input[128, 10, 3] to have 25 channels, got 10

25 is correct: the WM consumes an action BLOCK, and LIP derives it as
    fs = 5 ; a_dim = act.shape[-1] * fs          (train_lip_ac.py:265-266)
with the cube h5 holding raw 5-dim actions -> 5 * 5 = 25. PWM never applied
the frameskip and never had a real action source. It went unnoticed because
the PWM stack was validated on tworoom, and this is its first cube run.

THE PATCH. Add an optional --h5 and derive a_dim exactly as LIP does, keeping
the old expression as the fallback so existing callers are unaffected.

WHY A COPY. train_pwm_ac.py is a parallel session's committed file on a shared
branch. Editing it in place would collide with their work, so the patched
version lands beside it as train_pwm_ac_cube.py and the original is untouched.

Gates: both edits must apply exactly once, the result must compile, and
--help must run.
"""

import py_compile
import shutil
import subprocess
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/code/stable-worldmodel/scripts/plan/train_pwm_ac.py"
DST = sys.argv[2] if len(sys.argv) > 2 else \
    "/workspace/code/stable-worldmodel/scripts/plan/train_pwm_ac_cube.py"

OLD_ARG = "    p.add_argument('--cache-td', default='', help='fs1 cache for the critic (default: --cache)')"
NEW_ARG = OLD_ARG + (
    "\n    p.add_argument('--h5', default='', help='expert action h5. When given, "
    "a_dim = action_dim * fs (fs=5), the LIP convention -- the cache carries no "
    "action field, and the WM consumes an action BLOCK not a single action.')"
)

OLD_DIM = "    a_dim = int(getattr(c_act, 'action', torch.zeros(1, 10)).shape[-1])"
NEW_DIM = """    if a.h5:
        import h5py as _h5py
        with _h5py.File(a.h5, 'r') as _h:
            a_dim = int(_h['action'].shape[-1]) * 5   # fs=5 primitive steps/block
        print(f'[a_dim] from {a.h5}: raw x fs = {a_dim}', flush=True)
    else:
        a_dim = int(getattr(c_act, 'action', torch.zeros(1, 10)).shape[-1])"""


def main():
    src = open(SRC).read()
    for name, old in (("argparse --h5", OLD_ARG), ("a_dim derivation", OLD_DIM)):
        n = src.count(old)
        if n != 1:
            sys.exit(f"FATAL: '{name}' anchor found {n} times, expected exactly 1 "
                     f"-- the upstream file changed, re-read it before patching")
    out = src.replace(OLD_ARG, NEW_ARG).replace(OLD_DIM, NEW_DIM)
    assert out != src
    shutil.copymode(SRC, DST) if False else None
    with open(DST, "w") as f:
        f.write(out)
    py_compile.compile(DST, doraise=True)
    print(f"patched -> {DST}")
    r = subprocess.run([sys.executable, DST, "--help"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("FATAL: --help failed\n" + r.stderr[-1500:])
    if "--h5" not in r.stdout:
        sys.exit("FATAL: --h5 not in --help output")
    print("compile OK, --help OK, --h5 present")
    print("PATCH_PWM_OK")


if __name__ == "__main__":
    main()
