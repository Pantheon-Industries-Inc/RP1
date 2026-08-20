"""Run an RLP evaluation selected by a root Hydra config."""

from rlp.utils.config import dispatch, run_hydra


def run() -> object:
    return run_hydra(dispatch, config_name="inference/benchmark/lewm", selector=("benchmark", "inference/benchmark"))


if __name__ == "__main__":
    run()
