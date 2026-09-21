from rlp.utils.config import dispatch, run_hydra


def run() -> object:
    return run_hydra(
        dispatch,
        config_name="training/data/job/collect_tworoom_mixed",
        selector=("job", "training/data/job"),
    )


if __name__ == "__main__":
    run()
