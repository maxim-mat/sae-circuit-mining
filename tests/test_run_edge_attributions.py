import sys

import pytest

from scripts import run_edge_attributions as script


def _node_run(root, dataset, cf, finished=True):
    d = root / dataset / "attributions" / "nodes" / cf
    d.mkdir(parents=True)
    if finished:
        (d / "manifest.json").write_text("{}")
    return d


def test_finds_finished_node_runs_only(tmp_path):
    done = _node_run(tmp_path, "ioi", "abc_counterfactual")
    _node_run(tmp_path, "rc", "default", finished=False)
    simple = _node_run(tmp_path, "simple", "default")

    assert script.node_runs(tmp_path) == [done, simple]
    assert script.node_runs(tmp_path, only="ioi") == [done]


def test_methods_default_to_marks_case_and_their_intersection():
    assert script.DEFAULT_METHODS == ["marks", "case", "marks+case"]
    for name in script.DEFAULT_METHODS:
        assert script.METHODS[name]["threshold"] == 0.1


def test_one_failing_run_does_not_stop_the_rest(tmp_path, monkeypatch):
    a = _node_run(tmp_path, "ioi", "abc_counterfactual")
    b = _node_run(tmp_path, "simple", "default")
    seen = []

    def fake_run(node_dir, methods, force=False, device=None):
        seen.append((node_dir, sorted(methods), force))
        if node_dir == a:
            raise RuntimeError("boom")
        return {}

    monkeypatch.setattr(script, "run_positional_edges", fake_run)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-root", str(tmp_path),
                                      "--methods", "case", "mass", "--force"])
    assert script.main() == 1
    assert seen == [(a, ["case", "mass"], True), (b, ["case", "mass"], True)]


def test_unknown_method_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--output-root", str(tmp_path), "--methods", "nope"])
    with pytest.raises(SystemExit):
        script.main()
