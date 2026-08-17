import json
from src.utils.manifest import param_hash, stage_dir, find_cached, write_manifest


def test_hash_is_order_independent_and_stable():
    assert param_hash({"a": 1, "b": [2, 3]}) == param_hash({"b": [2, 3], "a": 1})
    assert param_hash({"a": 1}) != param_hash({"a": 2})
    assert len(param_hash({"a": 1})) == 8


def test_find_cached_misses_when_absent(tmp_path):
    assert find_cached(str(tmp_path), "node_attribution", {"a": 1}) is None


def test_find_cached_hits_after_write(tmp_path):
    params = {"suite": "pythia-marks", "steps": 10}
    d = stage_dir(str(tmp_path), "node_attribution", params)
    d.mkdir(parents=True)
    (d / "part-0.parquet").write_bytes(b"x")
    write_manifest(d, "node_attribution", params, {"row_count": 5})

    hit = find_cached(str(tmp_path), "node_attribution", params)
    assert hit == d
    m = json.loads((hit / "manifest.json").read_text())
    assert m["params"] == params
    assert m["stage"] == "node_attribution"
    assert m["row_count"] == 5
    assert m["git_commit"]


def test_different_params_do_not_collide(tmp_path):
    p1, p2 = {"steps": 10}, {"steps": 20}
    d1 = stage_dir(str(tmp_path), "s", p1); d1.mkdir(parents=True)
    write_manifest(d1, "s", p1, {})
    assert find_cached(str(tmp_path), "s", p2) is None


def test_incomplete_dir_is_not_a_hit(tmp_path):
    """A directory without a manifest is a crashed run, not a cache hit."""
    params = {"steps": 10}
    d = stage_dir(str(tmp_path), "s", params)
    d.mkdir(parents=True)
    assert find_cached(str(tmp_path), "s", params) is None


# -- the redundancy mechanism itself --------------------------------------

from src.utils.config import Config
from src.utils.manifest import resolve_stage_output, stage_params


def _cfg(tmp_path, **over):
    cfg = Config()
    cfg.suite.name = "gpt2-res"
    cfg.data.path = "x.json"
    cfg.run.output_root = str(tmp_path)
    for k, v in over.items():
        setattr(cfg.attribution, k, v)
    return cfg


def test_second_call_with_same_params_does_not_recompute(tmp_path):
    calls = []

    def compute(out_dir):
        calls.append(out_dir)
        (out_dir / "data.parquet").write_bytes(b"x")
        return {"row_count": 1}

    cfg = _cfg(tmp_path)
    d1, cached1 = resolve_stage_output(cfg, "node_attribution", compute)
    d2, cached2 = resolve_stage_output(cfg, "node_attribution", compute)

    assert len(calls) == 1, "compute_fn must not run on a cache hit"
    assert (d1, cached1, cached2) == (d2, False, True)


def test_changed_params_recompute_into_a_separate_dir(tmp_path):
    calls = []

    def compute(out_dir):
        calls.append(out_dir)
        return {}

    d1, _ = resolve_stage_output(_cfg(tmp_path, ig_steps=10), "node_attribution", compute)
    d2, _ = resolve_stage_output(_cfg(tmp_path, ig_steps=20), "node_attribution", compute)

    assert len(calls) == 2
    assert d1 != d2, "different params must not overwrite each other"
    assert d1.exists() and d2.exists(), "old hives are kept, not replaced"


def test_irrelevant_config_change_still_hits_the_cache(tmp_path):
    """logging is not in STAGE_DEPS['node_attribution'] -- changing it must not
    invalidate an expensive attribution run."""
    calls = []

    def compute(out_dir):
        calls.append(out_dir)
        return {}

    cfg = _cfg(tmp_path)
    resolve_stage_output(cfg, "node_attribution", compute)
    cfg.logging.run_name = "a-different-name"
    _, cached = resolve_stage_output(cfg, "node_attribution", compute)

    assert len(calls) == 1
    assert cached is True


def test_force_recomputes_and_overwrites(tmp_path):
    calls = []

    def compute(out_dir):
        calls.append(out_dir)
        return {}

    cfg = _cfg(tmp_path)
    resolve_stage_output(cfg, "node_attribution", compute)
    cfg.run.force = True
    _, cached = resolve_stage_output(cfg, "node_attribution", compute)

    assert len(calls) == 2
    assert cached is False


def test_manifest_records_what_produced_the_hive(tmp_path):
    import json
    d, _ = resolve_stage_output(_cfg(tmp_path, ig_steps=7), "node_attribution",
                                lambda out_dir: {"row_count": 42})
    m = json.loads((d / "manifest.json").read_text())
    assert m["params"]["attribution"]["ig_steps"] == 7
    assert m["params"]["data"]["counterfactual"] is None
    assert m["row_count"] == 42
    assert m["git_commit"] and m["created_utc"]


def test_force_replaces_outputs_rather_than_appending(tmp_path):
    """Regression: hive writers append, so recomputing into a populated
    directory duplicates every row -- and the manifest is then rewritten with
    the new run's counts, so the corruption is invisible. A real n=100 node hive
    reached 5,790,378 rows against a manifest claiming 1,345,653 this way."""
    def compute(out_dir):
        # Mimic pq.write_to_dataset: add a new file each call, never replace.
        existing = list(out_dir.glob("part-*.parquet"))
        (out_dir / f"part-{len(existing)}.parquet").write_bytes(b"x")
        return {"row_count": 10}

    cfg = _cfg(tmp_path)
    d1, _ = resolve_stage_output(cfg, "node_attribution", compute)
    assert len(list(d1.glob("part-*.parquet"))) == 1

    cfg.run.force = True
    d2, cached = resolve_stage_output(cfg, "node_attribution", compute)
    assert cached is False
    assert len(list(d2.glob("part-*.parquet"))) == 1, \
        "a forced recompute must replace the hive, not stack a second copy on it"


def test_a_crashed_half_written_hive_is_cleared_not_appended_to(tmp_path):
    """The same trap without force: an interrupted run leaves an unmanifested
    directory, and the retry must not add its rows to the wreckage."""
    params = stage_params(_cfg(tmp_path), "node_attribution")
    d = stage_dir(str(tmp_path), "node_attribution", params)
    d.mkdir(parents=True)
    (d / "part-0.parquet").write_bytes(b"partial")   # crashed mid-write

    def compute(out_dir):
        (out_dir / "part-0.parquet").write_bytes(b"clean")
        return {}

    out, cached = resolve_stage_output(_cfg(tmp_path), "node_attribution", compute)
    assert cached is False
    assert (out / "part-0.parquet").read_bytes() == b"clean"
    assert len(list(out.glob("part-*.parquet"))) == 1
