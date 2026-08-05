"""Run an RLP data tool selected by a root Hydra config."""

from rlp.config import dispatch, run_hydra


def run() -> object:
    return run_hydra(dispatch, config_name="tools/data/collect_tworoom_mixed", selector=("tool", "tools/data"))


if __name__ == "__main__":
    run()
