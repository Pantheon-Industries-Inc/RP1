"""Apply ONLY the ep_range support + its KeyError fix to a pod's eval_wm.py.

Deliberately surgical rather than copying the whole file over: the local
eval_wm.py also carries unrelated changes from other campaigns (a TD+Adam
cost()->forward() fix, a tworoom state-column convention) that must not be
imported into cube's eval path on a shared pod.

Idempotent — running twice is a no-op. Verifies with ast.parse before writing.

Usage: python patch_pod_ep_range.py /workspace/code/stable-worldmodel/scripts/plan/eval_wm.py
"""

import ast
import sys

EP_ANCHOR = """    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )
"""

EP_BLOCK = """
    # Optional episode-range restriction, e.g. `+eval.ep_range=0:8000`.
    # Tasks are drawn by sampling episodes, so a Dyna loop that COLLECTS
    # on-policy rollouts and later EVALUATES on the same dataset trains on the
    # very (start, goal) pairs it is scored on -- different seeds still draw from
    # the same pool. Give collection and evaluation disjoint ranges over the
    # same file to keep the eval tasks genuinely held out.
    _rng_spec = cfg.eval.get('ep_range', None)
    if _rng_spec:
        _lo, _hi = (int(x) for x in str(_rng_spec).split(':'))
        _keep = ep_indices[(ep_indices >= _lo) & (ep_indices < _hi)]
        assert len(_keep) >= cfg.eval.num_eval, (
            f'ep_range {_rng_spec} leaves {len(_keep)} episodes, '
            f'need >= {cfg.eval.num_eval}'
        )
        print(
            f'[eval] ep_range {_lo}:{_hi} -> {len(_keep)}/{len(ep_indices)} '
            'episodes eligible for the task draw'
        )
        ep_indices = _keep
"""

MASK_OLD = """    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in _row_epi]
    )
"""

MASK_NEW = """    # `ep_range` shrinks ep_indices but NOT the dataset, so rows belonging to
    # excluded episodes have no entry in max_start_idx_dict -- looking them up
    # unconditionally raises KeyError and made ep_range unusable. Mask them out
    # instead. With no ep_range every row is in range, so this is a no-op.
    _row_in_range = np.isin(_row_epi, ep_indices)
    max_start_per_row = np.full(len(_row_epi), -1, dtype=np.int64)
    max_start_per_row[_row_in_range] = [
        max_start_idx_dict[ep_id] for ep_id in _row_epi[_row_in_range]
    ]
"""

VALID_OLD = "    valid_mask = _row_step <= max_start_per_row\n"
VALID_NEW = "    valid_mask = _row_in_range & (_row_step <= max_start_per_row)\n"


def main():
    path = sys.argv[1]
    src = open(path).read()
    orig = src
    done = []

    if "_rng_spec = cfg.eval.get('ep_range'" in src:
        done.append("ep_range block already present")
    else:
        if src.count(EP_ANCHOR) != 1:
            sys.exit(f"FAIL: ep_indices anchor found {src.count(EP_ANCHOR)}x, expected 1")
        src = src.replace(EP_ANCHOR, EP_ANCHOR + EP_BLOCK)
        done.append("added ep_range block")

    if "_row_in_range" in src:
        done.append("mask fix already present")
    else:
        if src.count(MASK_OLD) != 1:
            sys.exit(f"FAIL: max_start_per_row anchor found {src.count(MASK_OLD)}x, expected 1")
        if src.count(VALID_OLD) != 1:
            sys.exit(f"FAIL: valid_mask anchor found {src.count(VALID_OLD)}x, expected 1")
        src = src.replace(MASK_OLD, MASK_NEW).replace(VALID_OLD, VALID_NEW)
        done.append("added KeyError mask fix")

    if src == orig:
        print("no changes needed: " + "; ".join(done))
        print("PATCH_EP_RANGE_DONE")
        return

    ast.parse(src)  # refuse to write a file that will not import
    open(path + ".bak_ep_range", "w").write(orig)
    open(path, "w").write(src)
    print("; ".join(done))
    print(f"backup at {path}.bak_ep_range")
    print("PATCH_EP_RANGE_DONE")


if __name__ == "__main__":
    main()
