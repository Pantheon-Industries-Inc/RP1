from rlp.utils.config import dispatch, run_hydra


def run() -> object:
    return run_hydra(
        dispatch,
        config_name="training/data/preparation/collect_tworoom_mixed",
        selector=("job", "training/data/preparation"),
    )


if __name__ == "__main__":
    run()
