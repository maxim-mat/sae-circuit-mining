import json
import pytest
from src.data.loaders import load_sva_examples, Example


class FakeTokenizer:
    """Whitespace tokenizer: one token per word, so lengths are predictable."""
    def __call__(self, text):
        class R:
            input_ids = list(range(len(text.split())))
        return R()


class FakeModel:
    tokenizer = FakeTokenizer()


def _write(tmp_path, rows):
    p = tmp_path / "d.json"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_reads_jsonl_despite_json_extension(tmp_path):
    rows = [{"clean_prefix": "the boy", "clean_answer": " is",
             "patch_prefix": "the boys", "patch_answer": " are"}]
    ex = load_sva_examples(_write(tmp_path, rows), 10, FakeModel(),
                           use_min_length_only=False)
    assert len(ex) == 1
    assert isinstance(ex[0], Example)
    assert ex[0].clean_prefix == "the boy"
    assert ex[0].patch_answer == " are"


def test_drops_mismatched_prefix_lengths(tmp_path):
    rows = [{"clean_prefix": "the boy", "clean_answer": " is",
             "patch_prefix": "the tall boys", "patch_answer": " are"}]
    assert load_sva_examples(_write(tmp_path, rows), 10, FakeModel(),
                             use_min_length_only=False) == []


def test_drops_multi_token_answers(tmp_path):
    rows = [{"clean_prefix": "the boy", "clean_answer": " is here",
             "patch_prefix": "the boys", "patch_answer": " are"}]
    assert load_sva_examples(_write(tmp_path, rows), 10, FakeModel(),
                             use_min_length_only=False) == []


def test_min_length_only_keeps_one_length_and_restarts(tmp_path):
    """Position alignment for aggregation='none' requires a single length.
    A shorter example appearing later must discard everything collected so far."""
    rows = [
        {"clean_prefix": "a b c", "clean_answer": " x",
         "patch_prefix": "d e f", "patch_answer": " y"},
        {"clean_prefix": "a b", "clean_answer": " x",
         "patch_prefix": "d e", "patch_answer": " y"},
        {"clean_prefix": "g h", "clean_answer": " x",
         "patch_prefix": "i j", "patch_answer": " y"},
        {"clean_prefix": "k l m", "clean_answer": " x",
         "patch_prefix": "n o p", "patch_answer": " y"},
    ]
    ex = load_sva_examples(_write(tmp_path, rows), 10, FakeModel(),
                           use_min_length_only=True, shuffle=False)
    assert len(ex) == 2
    assert {e.clean_prefix for e in ex} == {"a b", "g h"}


def test_respects_num_examples(tmp_path):
    rows = [{"clean_prefix": "a b", "clean_answer": " x",
             "patch_prefix": "c d", "patch_answer": " y"} for _ in range(10)]
    assert len(load_sva_examples(_write(tmp_path, rows), 3, FakeModel(),
                                 use_min_length_only=False)) == 3


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sva_examples(tmp_path / "nope.json", 1, FakeModel())


# -- one counterfactual per run -------------------------------------------

from src.data.loaders import select_counterfactual


def test_selects_only_the_designated_counterfactual():
    """A run is scoped to one counterfactual: causal questions do not mix."""
    item = {"prompt": "clean text",
            "abc": {"prompt": "abc text"},
            "name_swap": {"prompt": "swapped text"}}
    cf = select_counterfactual(item, "abc")
    assert cf["prompt"] == "abc text"


def test_missing_counterfactual_lists_what_is_available():
    item = {"prompt": "clean", "abc": {"prompt": "x"}, "name_swap": {"prompt": "y"}}
    with pytest.raises(ValueError, match="abc.*name_swap|name_swap.*abc"):
        select_counterfactual(item, "no_such_cf")


def test_unset_counterfactual_is_an_error_not_a_silent_pick():
    """Defaulting to 'the first one' would silently answer a different causal
    question than the config names."""
    item = {"prompt": "clean", "abc": {"prompt": "x"}}
    with pytest.raises(ValueError):
        select_counterfactual(item, None)


def test_num_examples_none_takes_everything(tmp_path):
    """Marks cap at 100; None means the whole surviving set. On rc_train that is
    ~77,764 examples, so this is a real mode for the small structures and a
    650-GPU-hour mistake for rc -- see DataConfig.num_examples."""
    rows = [{"clean_prefix": f"a{i} b", "clean_answer": " x",
             "patch_prefix": f"c{i} d", "patch_answer": " y"} for i in range(7)]
    ex = load_sva_examples(_write(tmp_path, rows), None, FakeModel(),
                           use_min_length_only=False)
    assert len(ex) == 7


def test_num_examples_none_still_honours_the_other_filters(tmp_path):
    rows = [{"clean_prefix": "a b", "clean_answer": " x",
             "patch_prefix": "c d", "patch_answer": " y"},
            {"clean_prefix": "a b", "clean_answer": " two words",
             "patch_prefix": "c d", "patch_answer": " y"}]
    ex = load_sva_examples(_write(tmp_path, rows), None, FakeModel(),
                           use_min_length_only=False)
    assert len(ex) == 1, "the multi-token answer must still be dropped"


def _mib_ids(monkeypatch, n, seed=42, counterfactual="cf"):
    import sys
    import types

    from src.data.loaders import load_mib_examples

    class FakeDataset(list):
        column_names = ["prompt", "cf", "other"]

    rows = FakeDataset({"prompt": f"p{i}", "cf": {"prompt": f"c{i}"},
                        "other": {"prompt": f"o{i}"}} for i in range(50))
    fake = types.SimpleNamespace(load_dataset=lambda **_: rows)
    monkeypatch.setitem(sys.modules, "datasets", fake)
    examples = load_mib_examples("x", n, None, counterfactual=counterfactual, seed=seed)
    return [e.meta["source_idx"] for e in examples], examples


def test_mib_uncapped_keeps_every_prompt_in_order(monkeypatch):
    assert _mib_ids(monkeypatch, None)[0] == list(range(50))


def test_mib_cap_is_a_seeded_sample_not_the_first_rows(monkeypatch):
    ids, _ = _mib_ids(monkeypatch, 10)
    assert ids != list(range(10))
    assert ids == _mib_ids(monkeypatch, 10)[0]
    assert ids != _mib_ids(monkeypatch, 10, seed=7)[0]


def test_mib_samples_are_comparable_across_sizes_and_counterfactuals(monkeypatch):
    small, _ = _mib_ids(monkeypatch, 10)
    large, _ = _mib_ids(monkeypatch, 20)
    other_cf, examples = _mib_ids(monkeypatch, 10, counterfactual="other")
    assert large[:10] == small
    assert other_cf == small
    assert examples[0].clean_prefix == f"p{small[0]}"
    assert examples[0].patch_prefix == f"o{small[0]}"
