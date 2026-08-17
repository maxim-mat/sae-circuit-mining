import pytest
from src.models.suites import build_suite, SUITES


def test_gpt2_ordinals_match_legacy_resid_ordering():
    """resid_post of block L is the same tensor as resid_pre of L+1, so only
    the final block's post-hook is a distinct site and gets ordinal 12."""
    s = build_suite("gpt2-res")
    by_id = {site.sae_id: site.ordinal for site in s.sites}
    for i in range(12):
        assert by_id[f"blocks.{i}.hook_resid_pre"] == i
    assert by_id["blocks.11.hook_resid_post"] == 12
    assert all(site.kind == "resid" for site in s.sites)


def test_pythia_has_all_19_marks_sites():
    s = build_suite("pythia-marks")
    assert len(s.sites) == 19                       # embed + 6 blocks x {attn, mlp, resid}
    kinds = [site.kind for site in s.sites]
    assert kinds.count("embed") == 1
    assert kinds.count("attn") == 6
    assert kinds.count("mlp") == 6
    assert kinds.count("resid") == 6


def test_pythia_ordinals_are_unique_strict_and_ascending():
    s = build_suite("pythia-marks")
    ordinals = [site.ordinal for site in s.sites]
    assert ordinals == sorted(ordinals)
    assert len(set(ordinals)) == len(ordinals)
    assert ordinals[0] == 0                          # embed first


def test_pythia_block_layout():
    s = build_suite("pythia-marks")
    by_ord = {site.ordinal: site for site in s.sites}
    assert by_ord[0].kind == "embed" and by_ord[0].block == -1
    for b in range(6):
        assert (by_ord[3 * b + 1].kind, by_ord[3 * b + 1].block) == ("attn", b)
        assert (by_ord[3 * b + 2].kind, by_ord[3 * b + 2].block) == ("mlp", b)
        assert (by_ord[3 * b + 3].kind, by_ord[3 * b + 3].block) == ("resid", b)


def test_parallel_attn_flag_records_concurrency():
    """Pythia runs attn and mlp in parallel; consecutive ordinals for those two
    sites are a linearisation, not a real dependency (Marks' circuit.py sets
    parallel_attn=True for pythia and skips the attn->mlp edge)."""
    assert build_suite("pythia-marks").parallel_attn is True
    assert build_suite("gpt2-res").parallel_attn is False


def test_hooks_are_transformerlens_names():
    s = build_suite("pythia-marks")
    hooks = {site.hook for site in s.sites}
    assert "blocks.0.hook_resid_pre" in hooks
    assert "blocks.3.hook_attn_out" in hooks
    assert "blocks.5.hook_mlp_out" in hooks
    assert "blocks.5.hook_resid_post" in hooks


def test_unknown_suite_raises():
    with pytest.raises(KeyError):
        build_suite("no-such-suite")


# -- how the model must be reconstituted for its SAEs ---------------------

from src.models.suites import model_load_kwargs, prepend_bos_for


class _FakeSAE:
    """Mimics the sae_lens 6.x shape: provenance lives on cfg.metadata."""
    def __init__(self, **metadata):
        class Meta:
            pass
        meta = Meta()
        for k, v in metadata.items():
            setattr(meta, k, v)

        class Cfg:
            pass
        self.cfg = Cfg()
        self.cfg.metadata = meta


def test_prepend_bos_is_read_from_metadata_not_cfg():
    """Regression: getattr(sae.cfg, "prepend_bos", True) always fell through to
    its default under sae_lens 6.x, so the pythia SAEs -- which declare False --
    were fed a BOS token they were never trained on."""
    assert prepend_bos_for({"a": _FakeSAE(prepend_bos=False)}) is False
    assert prepend_bos_for({"a": _FakeSAE(prepend_bos=True)}) is True


def test_prepend_bos_defaults_to_true_when_unstated():
    assert prepend_bos_for({"a": _FakeSAE()}) is True


def test_model_kwargs_carry_the_release_s_weight_processing():
    """gpt2-small-res-jb needs center_writing_weights; the pythia suite needs
    nothing. Loading either the other way is measurably wrong -- see
    docs/reference-numbers.md."""
    gpt2 = _FakeSAE(model_from_pretrained_kwargs={"center_writing_weights": True})
    assert model_load_kwargs({"a": gpt2}) == {"center_writing_weights": True}
    assert model_load_kwargs({"a": _FakeSAE(model_from_pretrained_kwargs=None)}) == {}
    assert model_load_kwargs({"a": _FakeSAE()}) == {}


def test_model_kwargs_are_a_copy_not_the_saes_dict():
    """Mutating the returned kwargs must not edit the SAE's own metadata."""
    sae = _FakeSAE(model_from_pretrained_kwargs={"center_writing_weights": True})
    model_load_kwargs({"a": sae})["center_writing_weights"] = False
    assert sae.cfg.metadata.model_from_pretrained_kwargs == {"center_writing_weights": True}
