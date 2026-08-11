"""Repository-level contracts for Hydra configuration and retained tools."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import rlp.tools
from rlp.config import dispatch, get_config_root, run_hydra, validate_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs"


def _public_configs(group: str) -> list[str]:
    return [
        str(path.relative_to(CONFIG_ROOT).with_suffix(""))
        for path in sorted((CONFIG_ROOT / group).rglob("*.yaml"))
        if not path.stem.startswith("_")
    ]


@pytest.mark.parametrize(
    "config_name",
    _public_configs("train") + _public_configs("eval") + _public_configs("tools"),
)
def test_every_public_job_config_composes(config_name: str) -> None:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name=config_name)
    assert config.entrypoint._target_.startswith("rlp.")


@pytest.mark.parametrize(
    "config_name",
    ["eval/lewm", "eval/pusht", "eval/reacher", "eval/tworoom_pixels", "eval/tworoom_state"],
)
def test_every_default_eval_config_composes(config_name: str) -> None:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name=config_name)
    assert config.environment.env_name
    assert not OmegaConf.missing_keys(config)


@pytest.mark.parametrize("config_name", ["train/lewm", "train/prejepa", "train/pipeline"])
def test_every_train_config_composes(config_name: str) -> None:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name=config_name)
    assert config.entrypoint._target_.startswith("rlp.")
    assert not OmegaConf.missing_keys(config)


def test_config_root_can_be_overridden(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setenv("RLP_CONFIG_DIR", str(tmp_path))
    assert get_config_root() == tmp_path


def test_run_hydra_composes_the_selected_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rlp", "model=prejepa", "runtime.seed=7"])
    config = run_hydra(lambda cfg: cfg, config_name="train/lewm", selector=("model", "train"))
    assert config.core.world_model.name == "prejepa"
    assert config.runtime.seed == 7
    run_directory = Path(config.run.directory)
    assert run_directory.parent.parent == tmp_path / "logs"
    assert (run_directory / "config.yaml").is_file()
    assert (run_directory / "metadata.json").is_file()
    assert (run_directory / "run.log").is_file()


def test_run_hydra_composes_the_selected_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rlp-tool", "tool=cache_latents", "wm=model.pt", "dataset=data.lance"])
    config = run_hydra(
        lambda cfg: cfg,
        config_name="tools/collect_tworoom_mixed",
        selector=("tool", "tools"),
    )
    assert config.entrypoint._target_ == "rlp.tools.data.cache_latents._run"
    assert config.wm == "model.pt"
    assert config.dataset == "data.lance"


def test_run_hydra_records_validation_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rlp", "model=state"])
    with pytest.raises(SystemExit) as failure:
        run_hydra(dispatch, config_name="train/lewm", selector=("model", "train"))

    assert failure.value.code == 1
    run_directory = next((tmp_path / "logs").glob("*/*"))
    metadata = OmegaConf.load(run_directory / "metadata.json")
    assert metadata.status == "failed"
    assert "Missing required configuration" in metadata.error
    assert "Run failed" in (run_directory / "run.log").read_text()


def test_run_hydra_records_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rlp", "model=lewm"])

    def interrupt(config: object) -> None:
        del config
        raise KeyboardInterrupt

    with pytest.raises(SystemExit) as interrupted:
        run_hydra(interrupt, config_name="train/lewm", selector=("model", "train"))

    assert interrupted.value.code == 130
    run_directory = next((tmp_path / "logs").glob("*/*"))
    metadata = OmegaConf.load(run_directory / "metadata.json")
    assert metadata.status == "interrupted"
    assert metadata.error == "KeyboardInterrupt"
    assert "Run interrupted" in (run_directory / "run.log").read_text()


def test_pipeline_data_and_caches_default_outside_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RLP_DATA_HOME", str(tmp_path))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="train/pipeline")
    assert config.data_directory == str(tmp_path / "datasets")
    assert config.cache_directory == str(tmp_path / "caches")


def test_argparse_is_not_used_in_source() -> None:
    offenders: list[Path] = []
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text()
        if "import argparse" in source or "ArgumentParser" in source or "parse_args(" in source:
            offenders.append(path.relative_to(ROOT))
    assert offenders == []


def test_source_uses_the_shared_logging_surface() -> None:
    offenders: list[Path] = []
    for path in (ROOT / "src" / "rlp").rglob("*.py"):
        source = path.read_text()
        if "print(" in source or "from tqdm" in source or "import tqdm" in source:
            offenders.append(path.relative_to(ROOT))
        if path.name != "logging.py" and "from loguru" in source:
            offenders.append(path.relative_to(ROOT))
    assert offenders == []


def test_train_and_eval_configs_have_no_legacy_output_fields() -> None:
    forbidden = {"out", "out_value", "save_value", "save_prefix", "run_name", "directory"}

    def keys(value: object) -> Iterator[object]:
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    for group in ("train", "eval"):
        for path in (CONFIG_ROOT / group).glob("*.yaml"):
            config = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
            assert forbidden.isdisjoint(keys(config)), path.relative_to(ROOT)


def test_config_validation_rejects_invalid_cross_field_values() -> None:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name="eval/lewm", overrides=["evaluation.budget=4"])
    with pytest.raises(ValueError, match="receding_horizon"):
        validate_config(config)


def test_wandb_mode_is_validated() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        offline = compose(config_name="train/lewm", overrides=["logging.wandb.mode=offline"])
        invalid = compose(config_name="train/lewm", overrides=["logging.wandb.mode=invalid"])
    validate_config(offline)
    with pytest.raises(ValueError, match="logging.wandb.mode"):
        validate_config(invalid)


def test_tool_modules_are_import_safe() -> None:
    names = [module.name for module in pkgutil.walk_packages(rlp.tools.__path__, prefix="rlp.tools.")]
    for name in names:
        importlib.import_module(name)
