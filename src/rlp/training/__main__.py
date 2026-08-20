from rlp.utils.config import dispatch, run_hydra


def run() -> object:
    return run_hydra(dispatch, config_name="training/lewm", selector=("model", "training"))


if __name__ == "__main__":
    run()
