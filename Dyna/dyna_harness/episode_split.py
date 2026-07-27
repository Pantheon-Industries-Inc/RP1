"""Canonical episode split for the Dyna loop — see ../DATA_SPLIT_POLICY.md.

RULE: episodes used for on-policy collection (and therefore for the WM fine-tune)
MUST be disjoint from the episodes evaluation draws its tasks from. Round-1 and
round-2 results in this repo were produced WITHOUT this split; they are upper
bounds, not generalization estimates.

Import the ranges rather than hard-coding them, so every script agrees:

    from episode_split import COLLECT, EVAL, assert_disjoint

CLI — measure the leak in an existing run (no pod state needed beyond the lances):

    python episode_split.py overlap \
        --collect /workspace/dyna_data/onpolicy_r1_a0.lance \
                  /workspace/dyna_data/onpolicy_r1_a1.lance \
                  /workspace/dyna_data/onpolicy_r1_a2.lance \
        --eval-dataset /workspace/datasets/ogb_cube_single/ogb_cube_single.lance \
        --eval-seeds 42 43 44 --num-eval 50 --goal-offset 25

Reports how many EVAL episodes were also collected from — i.e. how much of the
Dyna fine-tune saw the states it is later scored on.
"""

from __future__ import annotations

N_EPISODES = 10_000          # quentinll/lewm-cube expert set
COLLECT = range(0, 8_000)    # on-policy collection + fine-tune data
EVAL = range(8_000, 10_000)  # evaluation tasks ONLY


def assert_disjoint(collect_eps, eval_eps, *, label: str = "") -> None:
    """Raise if any episode is used for both collection/fine-tuning and eval."""
    shared = set(map(int, collect_eps)) & set(map(int, eval_eps))
    if shared:
        raise AssertionError(
            f"DATA SPLIT VIOLATION{' (' + label + ')' if label else ''}: "
            f"{len(shared)} episode(s) appear in BOTH the collection/fine-tune set "
            f"and the eval set, e.g. {sorted(shared)[:10]}. "
            f"See Dyna/DATA_SPLIT_POLICY.md — the Dyna gain would be partly "
            f"memorisation of the eval tasks."
        )


def in_collect(ep: int) -> bool:
    return int(ep) in COLLECT


def in_eval(ep: int) -> bool:
    return int(ep) in EVAL


# --------------------------------------------------------------------------- CLI
def _episodes_in_lance(path):
    import lance
    import numpy as np
    ds = lance.dataset(path)
    col = "episode_idx" if "episode_idx" in [f.name for f in ds.schema] else "ep_idx"
    arr = np.asarray(ds.take(list(range(ds.count_rows())), columns=[col]).to_pydict()[col])
    return set(int(x) for x in arr.reshape(-1))


def _eval_episodes(dataset_path, seed, num_eval, goal_offset):
    """Reproduce eval_wm.py's task draw: rng(seed).choice over valid start rows."""
    import lance
    import numpy as np
    ds = lance.dataset(dataset_path)
    names = [f.name for f in ds.schema]
    col = "episode_idx" if "episode_idx" in names else "ep_idx"
    n = ds.count_rows()
    t = ds.take(list(range(n)), columns=[col, "step_idx"]).to_pydict()
    epi = np.asarray(t[col]).reshape(-1).astype(np.int64)
    step = np.asarray(t["step_idx"]).reshape(-1).astype(np.int64)
    # a start row is valid if the goal frame exists inside the same episode
    ep_len = {}
    for e in np.unique(epi):
        ep_len[int(e)] = int((epi == e).sum())
    valid = np.array([i for i in range(n) if step[i] + goal_offset < ep_len[int(epi[i])]])
    g = np.random.default_rng(seed)
    idx = g.choice(len(valid) - 1, size=num_eval, replace=False)
    rows = np.sort(valid[idx])
    return set(int(e) for e in epi[rows]), [(int(epi[r]), int(step[r])) for r in rows]


def main():
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overlap", help="measure collection/eval episode leakage")
    o.add_argument("--collect", nargs="+", required=True, help="on-policy lance(s)")
    o.add_argument("--eval-dataset", required=True)
    o.add_argument("--eval-seeds", nargs="+", type=int, default=[42, 43, 44])
    o.add_argument("--num-eval", type=int, default=50)
    o.add_argument("--goal-offset", type=int, default=25)
    a = p.parse_args()

    collected = set()
    for c in a.collect:
        s = _episodes_in_lance(c)
        print(f"  {c}: {len(s)} distinct episodes")
        collected |= s
    print(f"collection touched {len(collected)} distinct episodes total")

    all_eval_eps, all_tasks = set(), []
    for sd in a.eval_seeds:
        eps, tasks = _eval_episodes(a.eval_dataset, sd, a.num_eval, a.goal_offset)
        shared = eps & collected
        print(f"  eval seed {sd}: {len(eps)} episodes, {len(shared)} also collected "
              f"({100*len(shared)/max(len(eps),1):.0f}%)")
        all_eval_eps |= eps
        all_tasks += tasks

    shared = all_eval_eps & collected
    exact = [t for t in all_tasks if t[0] in collected]
    print(f"\nEPISODE-level overlap: {len(shared)}/{len(all_eval_eps)} eval episodes "
          f"were also collected from ({100*len(shared)/max(len(all_eval_eps),1):.0f}%)")
    print(f"eval (episode,start) tasks whose episode was collected: "
          f"{len(exact)}/{len(all_tasks)}")
    print("\nNOTE: collection draws a DIFFERENT start step within an episode, so this is "
          "episode-level leakage, not identical tasks. It is still leakage: the fine-tune "
          "saw rollouts from the same demonstrations the eval scores on.")
    print("See Dyna/DATA_SPLIT_POLICY.md for the required split "
          f"(collect {COLLECT.start}-{COLLECT.stop-1}, eval {EVAL.start}-{EVAL.stop-1}).")


if __name__ == "__main__":
    main()
