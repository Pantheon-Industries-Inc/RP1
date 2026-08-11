"""Run an RLP evaluation selected by a root Hydra config."""

from rlp.config import dispatch, run_hydra


def run() -> object:
    return run_hydra(dispatch, config_name="eval/lewm", selector=("model", "eval"))


if __name__ == "__main__":
    run()
