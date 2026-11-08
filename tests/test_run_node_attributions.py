import json
from pathlib import Path

from scripts.run_node_attributions import BASE, plan
from scripts.sweep import collect
from src.utils.initialization import load_config
from src.utils.manifest import stage_dir, stage_params


def _dirs(tmp_path):
    out = {}
    for overrides in plan():
        cfg = load_config(BASE, overrides + [f"run.output_root={tmp_path.as_posix()}"])
        out[stage_dir(cfg.run.output_root, "node_attribution",
                      stage_params(cfg, "node_attribution")).relative_to(tmp_path).as_posix()] = cfg
    return out


def test_plan_covers_every_ioi_counterfactual_and_marks_structure(tmp_path):
    dirs = _dirs(tmp_path)
    assert len(dirs) == 12
    assert "ioi/attributions/nodes/abc_counterfactual" in dirs
    assert "ioi/attributions/nodes/random_names_s1_ioi_flip_s2_ioi_flip_counterfactual" in dirs
    for structure in ("simple", "nounpp", "rc", "within_rc"):
        assert f"{structure}/attributions/nodes/default" in dirs


def test_every_run_is_pythia_ig_with_the_metric_for_its_loader(tmp_path):
    for cfg in _dirs(tmp_path).values():
        assert cfg.suite.name == "pythia-marks"
        assert cfg.attribution.method == "ig"
        assert cfg.logging.backend == []
        expected = {"mib": ("mib_logit_diff", "template"), "sva": ("sva_logit_diff", "case")}
        assert (cfg.attribution.metric, cfg.data.template_field) == expected[cfg.data.loader]


def test_sweep_collect_reads_evaluations_under_per_n_roots(tmp_path):
    d = tmp_path / "n100" / "evaluation" / "abc"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "params": {"data": {"num_examples": 100},
                   "selection": {"aggregation": "none", "method": "threshold",
                                 "node_threshold": 0.1},
                   "evaluation": {"handle_errors": "default"}},
        "n_nodes": 5}))
    dest = tmp_path / "results.csv"

    collect(str(tmp_path), dest)

    assert len(Path(dest).read_text().splitlines()) == 2


def test_every_run_defaults_to_all_prompts_of_every_length(tmp_path):
    for cfg in _dirs(tmp_path).values():
        assert cfg.data.num_examples is None
        assert cfg.data.use_min_length_only is False
