from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlp.utils.config import dispatch, get_config_root, run_hydra, validate_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs"


def _public_configs(group: str) -> list[str]:
    return [
        str(path.relative_to(CONFIG_ROOT).with_suffix(""))
        for path in sorted((CONFIG_ROOT / group).glob("*.yaml"))
        if not path.stem.startswith("_")
    ]


@pytest.mark.parametrize(
    "config_name",
    _public_configs("training") + _public_configs("inference/benchmark") + _public_configs("training/data/job"),
)
def test_every_public_job_config_composes(config_name: str) -> None:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name=config_name)
    assert config.entrypoint._target_.startswith("rlp.")


@pytest.mark.parametrize(
    "config_name",
    ["inference/benchmark/lewm", "inference/benchmark/reacher", "inference/benchmark/tworoom_pixels"],
)
def test_every_default_benchmark_config_composes(config_name: str) -> None:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name=config_name)
    assert config.environment.env_name
    assert not OmegaConf.missing_keys(config)


def test_default_training_config_composes() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="training/lewm")
    assert config.entrypoint._target_.startswith("rlp.")
    assert not OmegaConf.missing_keys(config)


def test_config_root_can_be_overridden(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RLP_CONFIG_DIR", str(tmp_path))
    assert get_config_root() == tmp_path


def test_run_hydra_composes_the_selected_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rlp", "model=lewm", "runtime.seed=7"])
    config = run_hydra(lambda cfg: cfg, config_name="training/lewm", selector=("model", "training"))
    assert config.core.world_model.name == "lewm"
    assert config.runtime.seed == 7
    run_directory = Path(config.run.directory)
    assert run_directory.parent.parent == tmp_path / "logs"
    assert (run_directory / "config.yaml").is_file()
    assert (run_directory / "metadata.json").is_file()
    assert (run_directory / "run.log").is_file()


def test_run_hydra_composes_the_selected_preparation_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "job=cache_latents",
            "preparation.wm=model.pt",
            "preparation.dataset=data.lance",
            "preparation.out=cache.pt",
        ],
    )
    config = run_hydra(
        lambda cfg: cfg,
        config_name="training/data/job/collect_tworoom_mixed",
        selector=("job", "training/data/job"),
    )
    assert config.entrypoint._target_ == "rlp.training.data.job.cache_latents._run"
    assert config.preparation.wm == "model.pt"
    assert config.preparation.dataset == "data.lance"


def test_run_hydra_records_validation_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rlp", "model=metric"])
    with pytest.raises(SystemExit) as failure:
        run_hydra(dispatch, config_name="training/lewm", selector=("model", "training"))

    assert failure.value.code == 1
    run_directory = next((tmp_path / "logs").glob("*/*"))
    metadata = OmegaConf.load(run_directory / "metadata.json")
    assert metadata.status == "failed"
    assert "Missing required configuration" in metadata.error


def test_pipeline_caches_default_outside_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RLP_DATA_HOME", str(tmp_path))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="training/rlp")
    assert config.training.cache_directory == str(tmp_path / "caches")


def test_config_validation_rejects_invalid_cross_field_values() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="inference/benchmark/lewm", overrides=["planning.budget=4"])
    with pytest.raises(ValueError, match="receding_horizon"):
        validate_config(config)


def test_wandb_mode_is_validated() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        offline = compose(config_name="training/lewm", overrides=["logging.wandb.mode=offline"])
        invalid = compose(config_name="training/lewm", overrides=["logging.wandb.mode=invalid"])
    validate_config(offline)
    with pytest.raises(ValueError, match="logging.wandb.mode"):
        validate_config(invalid)


def test_repository_trees_share_the_subsystem_skeleton() -> None:
    paths = (
        "core",
        "data",
        "environment",
        "inference",
        "training",
        "training/data",
        "training/data/job",
        "utils",
    )
    for relative in paths:
        assert (ROOT / "src" / "rlp" / relative).is_dir(), relative
        assert (ROOT / "configs" / relative).is_dir(), relative
        assert (ROOT / "tests" / relative).is_dir(), relative


def test_core_model_import_does_not_load_training() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rlp.core.world_model; "
            "assert not any(name == 'rlp.training' or name.startswith('rlp.training.') for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
