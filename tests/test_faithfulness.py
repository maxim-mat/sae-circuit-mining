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
