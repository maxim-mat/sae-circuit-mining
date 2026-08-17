"""End-to-end smoke tests. Marked slow: these download Pythia-70M and 19 SAEs
and run real attribution, so they are excluded from the default run.

    C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_end_to_end.py -v -m slow
"""
import pytest
from pathlib import Path

from src.utils.initialization import load_config
from src.pipeline import run_pipeline

CONFIG = Path("configs/pythia_marks_rc.yaml")


def _overrides(tmp_path):
    # as_posix(): a Windows path in a dotlist override would otherwise carry
    # backslashes into the YAML parser.
    return ["data.num_examples=5",
            "attribution.ig_steps=2",
            "evaluation.num_examples=5",
            f"run.output_root={Path(tmp_path).as_posix()}",
            "logging.backend=[]"]


@pytest.mark.slow
def test_pipeline_runs_end_to_end_on_five_examples(tmp_path):
    cfg = load_config(CONFIG, _overrides(tmp_path))
    out = run_pipeline(cfg)
    assert (out["circuit"] / "manifest.json").exists()
    assert out["evaluation"] is not None


@pytest.mark.slow
def test_second_run_reuses_the_cache(tmp_path):
    args = _overrides(tmp_path)
    cfg = load_config(CONFIG, args)
    first = run_pipeline(cfg)
    second = run_pipeline(cfg)
    assert first["node_attribution"] == second["node_attribution"]
