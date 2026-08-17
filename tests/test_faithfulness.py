import pytest
from src.eval.faithfulness import faithfulness


def test_formula_matches_marks_ablation_py():
    """(F(C) - F(empty)) / (F(M) - F(empty))"""
    assert faithfulness(f_c=0.8, f_m=1.0, f_empty=0.0) == pytest.approx(0.8)
    assert faithfulness(f_c=0.5, f_m=1.0, f_empty=0.5) == pytest.approx(0.0)
    assert faithfulness(f_c=1.0, f_m=1.0, f_empty=0.5) == pytest.approx(1.0)


def test_circuit_beating_the_full_model_exceeds_one():
    """Not clamped -- values above 1 are a real, reportable outcome."""
    assert faithfulness(f_c=1.2, f_m=1.0, f_empty=0.0) == pytest.approx(1.2)


def test_degenerate_denominator_raises():
    with pytest.raises(ZeroDivisionError):
        faithfulness(f_c=0.5, f_m=1.0, f_empty=1.0)


from src.eval.faithfulness import resolve_start_layer
from src.models.suites import build_suite


def test_default_start_layer_skips_the_first_third():
    """Marks et al. 3.2 evaluate only the latter 2/3 of the circuit: early
    features key on specific tokens, which differ between train and test."""
    assert resolve_start_layer(build_suite("pythia-marks"), -1) == 2   # 6 blocks
    assert resolve_start_layer(build_suite("gpt2-res"), -1) == 4       # 12 blocks


def test_explicit_start_layer_is_used_verbatim():
    assert resolve_start_layer(build_suite("pythia-marks"), 0) == 0


def test_completeness_is_the_complement_s_faithfulness():
    """Marks section 3.2: completeness = faithfulness of M \ C, ideal 0. It uses
    the same formula, so a circuit whose complement recovers the full model
    scores 1 (nothing captured) and one whose complement is dead scores 0."""
    assert faithfulness(f_c=1.0, f_m=1.0, f_empty=0.0) == pytest.approx(1.0)
    assert faithfulness(f_c=0.0, f_m=1.0, f_empty=0.0) == pytest.approx(0.0)


def test_start_layer_zero_keeps_every_block_but_not_embed():
    """Their ablation.py filters submodules `>= start_layer` with default -1,
    which keeps every block; its faithfulness script also passes
    include_embed=False. start_layer=0 reproduces both at once, since embed
    carries block=-1."""
    suite = build_suite("pythia-marks")
    assert resolve_start_layer(suite, 0) == 0
    ablated = [s for s in suite.sites if s.block >= 0]
    assert len(ablated) == 18, "6 blocks x {attn, mlp, resid}"
    assert all(s.kind != "embed" for s in ablated)


def test_the_two_readings_of_the_one_third_rule_differ():
    """Paper §3.2 says skip the first 1/3; ablation.py's default skips nothing.
    Both are reported, so neither is chosen by an unremembered default."""
    suite = build_suite("pythia-marks")
    assert resolve_start_layer(suite, -1) == 2      # paper
    assert resolve_start_layer(suite, 0) == 0       # their code default
