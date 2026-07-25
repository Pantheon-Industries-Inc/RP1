"""Patch LIPSolver._proposal_lip to stop recomputing a rollout it already has.

At solver/lip.py:549-554 the inference path does:

    A_in  = A.detach().requires_grad_(True)
    traj  = rollout_traj(wm, zh_r, a_hist, A_in)          # rollout #1 (grad)
    (gA,) = autograd.grad(value(traj[:, -1], zg_r).sum(), A_in)
    with torch.no_grad():
        traj_f = rollout_traj(wm, zh_r, a_hist, A)        # rollout #2 -- SAME VALUES
        E      = value(traj_f[:, -1], zg_r)               # == the scalar summed above

`A` is not mutated between those lines and `A_in` holds `A`'s values, so
traj_f == traj.detach() exactly (wm is .eval(): dropout off, BatchNorm on running
stats). That makes 40 of the solver's 85 forward predict() calls redundant.

The TRAINER already does exactly this -- scripts/plan/train_lip_ac.py:407-409:
    # A_in holds A's values, so the grad pass's trajectory IS the feature
    # trajectory -- reuse it instead of a third WM rollout (value-identical)
    traj_f = traj.detach()
The solver never got the same treatment.

Env-gated so the default path is untouched:
    LIP_REUSE_TRAJ=0 / unset  -> original behaviour
    LIP_REUSE_TRAJ=1          -> reuse traj.detach() (the optimization)
    LIP_REUSE_TRAJ=verify     -> compute BOTH and print max|diff| per iteration

Usage:  python3 patch_lip_reuse_traj.py /path/to/stable_worldmodel/solver/lip.py
Writes a .bak alongside. Idempotent.
"""
import shutil
import sys
from pathlib import Path

OLD = """            with torch.no_grad():
                traj_f = rollout_traj(wm, zh_r, a_hist, A)
                E = self.lip_value(traj_f[:, -1], zg_r)"""

NEW = '''            with torch.no_grad():
                # [patched] traj was rolled from A_in, which holds A's values, so
                # traj.detach() IS traj_f -- see train_lip_ac.py:407-409. Skipping
                # the recompute removes H of the 2H forward WM steps per iteration.
                _reuse = os.environ.get("LIP_REUSE_TRAJ", "")
                if _reuse == "verify":
                    traj_f = rollout_traj(wm, zh_r, a_hist, A)
                    _d = (traj_f - traj.detach()).abs().max().item()
                    print(f"[reuse-verify] k={k_it} max|traj_f - traj.detach()| = {_d:.3e}",
                          flush=True)
                elif _reuse and _reuse != "0":
                    traj_f = traj.detach()
                else:
                    traj_f = rollout_traj(wm, zh_r, a_hist, A)
                E = self.lip_value(traj_f[:, -1], zg_r)'''


def main():
    p = Path(sys.argv[1])
    src = p.read_text()
    if "[patched] traj was rolled from A_in" in src:
        print(f"already patched: {p}")
        return
    if OLD not in src:
        sys.exit(f"anchor not found in {p} -- solver/lip.py may have changed")
    if "\nimport os\n" not in src:
        sys.exit(f"expected `import os` in {p}")
    shutil.copy2(p, str(p) + ".bak")
    p.write_text(src.replace(OLD, NEW, 1))
    print(f"patched {p} (backup at {p}.bak)")


if __name__ == "__main__":
    main()
