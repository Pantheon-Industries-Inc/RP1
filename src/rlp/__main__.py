"""Run an RLP training job selected by a root Hydra config."""

from rlp.config import dispatch, run_hydra


def run() -> object:
    """Run LeWM by default; ``model=<name>`` selects another training job."""
    return run_hydra(dispatch, config_name="train/lewm", selector=("model", "train"))


if __name__ == "__main__":
    run()
