"""Composable, in-process TRM reproduction pipeline."""

from pathlib import Path
from typing import TypedDict

from hydra import compose
from omegaconf import DictConfig, OmegaConf, open_dict

from rlp.config import dispatch, run_hydra
from rlp.logging import logger
from rlp.run import save_stage_config


class TaskSettings(TypedDict):
    env: str
    dataset: str
    obs_key: str
    wm: str
    scale: int


TASKS: dict[str, TaskSettings] = {
    "tworoom": {
        "env": "swm/TwoRoom-v1",
        "dataset": "tworoom_expert.lance",
        "obs_key": "proprio",
        "wm": "statewm_tworoom",
        "scale": 60,
    },
}


def _stage(parent: DictConfig, index: int, name: str, config_name: str, **values: object) -> object:
    """Compose a stage and execute it inside the pipeline's parent run."""
    stage = compose(config_name=config_name)
    with open_dict(stage):
        stage.run = OmegaConf.create(OmegaConf.to_container(parent.run, resolve=True))
    for key, value in values.items():
        OmegaConf.update(stage, key, value, merge=False)
    stage_directory = Path(parent.run.stages) / f"{index:02d}-{name}"
    save_stage_config(stage, stage_directory)
    logger.info(f"Pipeline stage started: {name}")
    result = dispatch(stage)
    logger.info(f"Pipeline stage completed: {name}")
    return result


def _run(cfg: DictConfig) -> None:
    task = TASKS[cfg.task]
    skip = {name.strip() for name in cfg.skip.split(",") if name.strip()}
    dataset = str(Path(cfg.data_directory) / task["dataset"])
    cache = str(Path(cfg.cache_directory) / f"{cfg.task}_state.pt")
    checkpoint_directory = Path(cfg.run.checkpoints)
    world_model = checkpoint_directory / task["wm"]
    stage_index = 0

    def run_stage(name: str, config_name: str, **values: object) -> object:
        nonlocal stage_index
        stage_index += 1
        return _stage(cfg, stage_index, name, config_name, **values)

    if "collect" not in skip:
        run_stage(
            "collect",
            "tools/collect_tworoom_mixed",
            expert=400,
            random=0,
            num_envs=10,
            max_steps=100,
            out=dataset,
        )
    if "wm" not in skip:
        run_stage(
            "world-model",
            "train/state",
            dataset=dataset,
            **{"output.model_name": task["wm"]},
            obs_key=task["obs_key"],
            steps=cfg.wm_steps,
            device=cfg.device,
        )
    if "cache" not in skip:
        run_stage(
            "cache-latents",
            "tools/cache_latents",
            wm=str(world_model),
            dataset=dataset,
            out=cache,
            device=cfg.device,
        )
    if "metrics" not in skip:
        for learner in ("regression", "td", "contrastive"):
            run_stage(
                f"metric-{learner}",
                "train/metric",
                cache=cache,
                learner=learner,
                **{"output.checkpoint": f"{cfg.task}_{learner}"},
                steps=cfg.metric_steps,
                scale=task["scale"],
                device=cfg.device,
            )
        run_stage(
            "metric-shuffled",
            "train/metric",
            cache=cache,
            learner="regression",
            labels="shuffled",
            **{"output.checkpoint": f"{cfg.task}_shuffled"},
            steps=cfg.metric_steps,
            scale=task["scale"],
            device=cfg.device,
        )
    if "eval" not in skip:
        conditions = [
            "latent",
            f"trm_regression={checkpoint_directory}/{cfg.task}_regression",
            f"trm_td={checkpoint_directory}/{cfg.task}_td",
            f"trm_contrastive={checkpoint_directory}/{cfg.task}_contrastive",
            f"hybrid={checkpoint_directory}/{cfg.task}_regression",
            f"shuffled={checkpoint_directory}/{cfg.task}_shuffled",
            "oracle",
        ]
        run_stage(
            "evaluate",
            "eval/trm",
            env=task["env"],
            wm=str(world_model),
            dataset=dataset,
            num_eval=cfg.num_eval,
            goal_offset=cfg.goal_offset,
            eval_budget=cfg.eval_budget,
            horizon=cfg.horizon,
            receding=cfg.receding,
            num_samples=cfg.num_samples,
            cem_steps=cfg.cem_steps,
            conditions=conditions,
            scsa=True,
            cross_wall=cfg.cross_wall,
            device=cfg.device,
            **{"output.filename": f"{cfg.task}_sweep.txt"},
        )


def main() -> object:
    return run_hydra(dispatch, config_name="train/pipeline")


if __name__ == "__main__":
    main()
