import pytest
from src.models.suites import build_suite
from src.attribution.edge_template import marks_template, total_effect_template


def _as_kinds(specs):
    return [(s.src.kind, s.src.block, s.dst.kind, s.dst.block,
             tuple(sorted((x.kind, x.block) for x in s.stop))) for s in specs]


def test_marks_template_matches_circuit_py_for_one_pythia_block():
    """circuit.py, layer L (parallel_attn=True so AM is skipped):
        MR = N(mlp_L,     resid_L)
        AR = N(attn_L,    resid_L, stop=[mlp_L])
        RM = N(resid_L-1, mlp_L,   stop=[attn_L])
        RA = N(resid_L-1, attn_L)
        RR = N(resid_L-1, resid_L, stop=[mlp_L, attn_L])
    """
    specs = _as_kinds(marks_template(build_suite("pythia-marks")))
    assert ("mlp", 1, "resid", 1, ()) in specs
    assert ("attn", 1, "resid", 1, (("mlp", 1),)) in specs
    assert ("resid", 0, "mlp", 1, (("attn", 1),)) in specs
    assert ("resid", 0, "attn", 1, ()) in specs
    assert ("resid", 0, "resid", 1, (("attn", 1), ("mlp", 1))) in specs


def test_parallel_attn_suppresses_the_attn_to_mlp_edge():
    """circuit.py: `if not parallel_attn: AM_effect = N(attn, mlp)`."""
    specs = _as_kinds(marks_template(build_suite("pythia-marks")))
    assert not any(s[0] == "attn" and s[2] == "mlp" for s in specs)


def test_embed_feeds_block_zero_in_place_of_resid_minus_one():
    """circuit.py: `prev_resid = resids[layer-1] if layer > 0 else embed`."""
    specs = _as_kinds(marks_template(build_suite("pythia-marks")))
    assert ("embed", -1, "attn", 0, ()) in specs
    assert ("embed", -1, "mlp", 0, (("attn", 0),)) in specs
    assert ("embed", -1, "resid", 0, (("attn", 0), ("mlp", 0))) in specs


def test_template_has_no_long_range_edges():
    """Long-range influence is composition along the resid chain, never an edge."""
    for spec in marks_template(build_suite("pythia-marks")):
        if spec.src.kind == "embed":
            assert spec.dst.block == 0
        else:
            assert spec.dst.block - spec.src.block <= 1


def test_final_resid_connects_to_the_y_node():
    specs = marks_template(build_suite("pythia-marks"))
    y_edges = [s for s in specs if s.dst.ordinal == -1]
    assert len(y_edges) == 1
    assert y_edges[0].src.kind == "resid" and y_edges[0].src.block == 5


def test_resid_only_suite_degenerates_to_the_resid_chain():
    """gpt2-res has no attn/mlp dictionaries, so the template is resid_i ->
    resid_i+1 with nothing to stop-gradient."""
    specs = marks_template(build_suite("gpt2-res"))
    chain = [s for s in specs if s.dst.ordinal != -1]
    assert all(s.src.kind == "resid" and s.dst.kind == "resid" for s in chain)
    assert all(s.dst.ordinal - s.src.ordinal == 1 for s in chain)
    assert all(s.stop == () for s in chain)


def test_total_effect_template_is_all_pairs_without_stopgrads():
    specs = total_effect_template(build_suite("gpt2-res"), pairs="all")
    assert all(s.stop == () for s in specs)
    assert any(s.dst.ordinal - s.src.ordinal > 1 for s in specs), \
        "total-effect mode exists precisely to score non-adjacent pairs"


def test_total_effect_consecutive_is_adjacent_only():
    specs = total_effect_template(build_suite("gpt2-res"), pairs="consecutive")
    assert all(s.dst.ordinal - s.src.ordinal == 1 for s in specs)
