from pathlib import Path
import pytest
from src.utils.config import Config
from src.utils.initialization import load_config


def test_defaults_and_yaml_and_override_precedence(tmp_path):
    yaml = tmp_path / "c.yaml"
    yaml.write_text(
        "suite: {name: pythia-marks}\n"
        "data: {path: data/rc_train.json, num_examples: 100}\n"
        "attribution: {method: ig, ig_steps: 10}\n"
    )
    cfg = load_config(yaml, ["attribution.ig_steps=5", "selection.node_threshold=0.2"])

    assert isinstance(cfg, Config)
    assert cfg.suite.name == "pythia-marks"       # from yaml
    assert cfg.attribution.method == "ig"          # from yaml
    assert cfg.attribution.ig_steps == 5           # override beats yaml
    assert cfg.selection.node_threshold == 0.2     # override beats default
    assert cfg.seed == 42                          # schema default survives


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", [])


def test_unknown_key_rejected(tmp_path):
    yaml = tmp_path / "c.yaml"
    yaml.write_text("attribution: {no_such_key: 1}\n")
    with pytest.raises(Exception):
        load_config(yaml, [])
