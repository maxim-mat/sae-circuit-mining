import sys
import types

import pytest
import torch
from safetensors.torch import save_file

from src.attribution import nodes
from src.attribution.nodes import read_node_hive, run_node_attribution
from src.data.loaders import Example
from src.models.suites import Site, Suite, build_suite, load_model_and_saes
from src.models.transcoders import Transcoder, load_transcoder
from src.utils.config import Config

IN, OUT = "blocks.0.ln2.hook_normalized", "blocks.0.hook_mlp_out"


def _tensors(d_in=4, d_sae=16, d_out=4, **extra):
    torch.manual_seed(0)
    return {"W_enc": torch.randn(d_in, d_sae), "W_dec": torch.randn(d_sae, d_out),
            "b_enc": torch.zeros(d_sae), "b_dec": torch.zeros(d_out), **extra}


def test_encode_is_relu_of_the_affine_map_and_decode_is_affine():
    t = _tensors(b_enc=torch.full((16,), 0.5), b_dec=torch.ones(4))
    tc = Transcoder(t)
    x = torch.randn(1, 3, 4)
    f = torch.relu(x @ t["W_enc"] + 0.5)
    assert torch.equal(tc.encode(x), f)
    assert torch.equal(tc.decode(f), f @ t["W_dec"] + 1)
    assert tc.cfg.d_sae == 16


def test_a_threshold_makes_encode_jumprelu():
    t = _tensors(threshold=torch.full((16,), 0.5))
    x = torch.randn(1, 3, 4)
    pre = x @ t["W_enc"]
    assert torch.equal(Transcoder(t).encode(x), torch.relu(pre) * (pre > 0.5))


def test_transposed_weights_are_normalised_by_the_feature_count():
    t = _tensors(d_in=4, d_sae=16, d_out=6)
    flipped = Transcoder({**t, "W_enc": t["W_enc"].T, "W_dec": t["W_dec"].T})
    assert flipped.W_enc.shape == (4, 16) and flipped.W_dec.shape == (16, 6)


def test_missing_tensors_are_named():
    t = _tensors()
    del t["b_enc"]
    with pytest.raises(ValueError, match="b_enc"):
        Transcoder(t)


def test_loads_a_local_checkpoint_and_converts_dtype(tmp_path):
    save_file({k: v.to(torch.bfloat16) for k, v in _tensors().items()},
              tmp_path / "tc.safetensors")
    tc = load_transcoder(str(tmp_path), "tc.safetensors", "cpu", torch.float32)
    assert tc.W_enc.dtype == torch.float32


def _site(read=IN, loader="safetensors", release="r", sae_id="tc.safetensors"):
    return Site(ordinal=0, block=0, kind="mlp", release=release, sae_id=sae_id,
                hook=OUT, input_hook=read, loader=loader)


def test_site_reads_at_its_own_hook_unless_given_an_input_hook():
    assert _site().read_hook == IN
    assert _site(read=None).read_hook == OUT


def _fake_libs(monkeypatch):
    calls = {}
    model = types.SimpleNamespace(set_use_hook_mlp_in=lambda v: calls.update(mlp_in=v))

    def from_pretrained(name, device, dtype, **kwargs):
        calls["kwargs"] = kwargs
        return model

    monkeypatch.setitem(sys.modules, "transformer_lens", types.SimpleNamespace(
        HookedTransformer=types.SimpleNamespace(from_pretrained_no_processing=from_pretrained)))
    monkeypatch.setitem(sys.modules, "sae_lens", types.SimpleNamespace(SAE=None))
    return calls


def test_loading_a_transcoder_suite_must_be_opted_into(monkeypatch):
    _fake_libs(monkeypatch)
    suite = Suite("s", "m", False, (_site(),))
    with pytest.raises(ValueError, match="transcoder"):
        load_model_and_saes(suite, "cpu")


def test_loads_from_any_safetensors_source_and_applies_the_suite_kwargs(monkeypatch, tmp_path):
    calls = _fake_libs(monkeypatch)
    save_file(_tensors(), tmp_path / "tc.safetensors")
    site = _site(read="blocks.0.hook_mlp_in", release=str(tmp_path))
    suite = Suite("s", "m", False, (site,), model_kwargs={"center_writing_weights": True})

    _, saes = load_model_and_saes(suite, "cpu", allow_transcoders=True)

    assert isinstance(saes["tc.safetensors"], Transcoder)
    assert calls == {"kwargs": {"center_writing_weights": True}, "mlp_in": True}


def test_unknown_loader_is_rejected(monkeypatch):
    _fake_libs(monkeypatch)
    with pytest.raises(ValueError, match="nope"):
        load_model_and_saes(Suite("s", "m", False, (_site(loader="nope"),)), "cpu",
                            allow_transcoders=True)


def test_gpt2_transcoder_suite_reads_mlp_input_and_writes_mlp_output():
    sites = build_suite("gpt2-transcoders").sites
    assert len(sites) == 12
    assert [s.ordinal for s in sites] == list(range(12))
    assert sites[3].read_hook == "blocks.3.ln2.hook_normalized"
    assert sites[3].hook == "blocks.3.hook_mlp_out"
    assert sites[3].sae_id == "gpt2-small-dun-chl-mlp-tc3/sae.safetensors"


class _FakeModel:
    """Runs IN -> tanh -> OUT, so a dictionary reading the wrong hook is visible."""
    def __init__(self):
        self.embed = torch.randn(8, 4, requires_grad=True)
        self.hooks = {}
        self.out = {}

    def to_tokens(self, text, prepend_bos):
        return torch.tensor([[1, 2, 3]] if text == "clean" else [[4, 5, 6]])

    def add_hook(self, name, fn, dir):
        self.hooks.setdefault(name, []).append(fn)

    def reset_hooks(self):
        self.hooks = {}

    def zero_grad(self):
        pass

    def _fire(self, name, x):
        for fn in self.hooks.get(name, []):
            x = fn(x, None)
        return x

    def __call__(self, tokens):
        x = self._fire(IN, self.embed[tokens])
        y = self._fire(OUT, torch.tanh(x) * 3)
        self.out[int(tokens[0, 0])] = (x.detach()[0], y.detach()[0])
        return y


def test_node_attribution_encodes_input_and_splits_the_output_ie(monkeypatch, tmp_path):
    torch.manual_seed(0)
    tc, model, seen = Transcoder(_tensors()), _FakeModel(), {}
    site = _site()

    def load(*args, **kwargs):
        seen.update(kwargs)
        return model, {site.sae_id: tc}

    example = Example(clean_prefix="clean", clean_answer="", patch_prefix="patch")
    monkeypatch.setattr(nodes, "build_suite", lambda name: Suite("s", "m", False, (site,)))
    monkeypatch.setattr(nodes, "load_model_and_saes", load)
    monkeypatch.setattr(nodes, "prepend_bos_for", lambda saes: True)
    monkeypatch.setattr(nodes, "load_examples", lambda *a: [example])
    monkeypatch.setattr(nodes, "get_metric", lambda name: lambda logits, **_: logits.sum())
    monkeypatch.setattr(nodes, "integrated_gradient_grad_x",
                        lambda m, t, hook, acts, *_: torch.ones_like(acts))
    cfg = Config()
    cfg.suite.name, cfg.suite.device, cfg.data.path = "s", "cpu", "x"

    run_node_attribution(cfg, tmp_path)

    df = read_node_hive(tmp_path)
    (x_c, y_c), (x_p, y_p) = model.out[1], model.out[4]
    delta_f = tc.encode(x_p.unsqueeze(0))[0] - tc.encode(x_c.unsqueeze(0))[0]
    expected_features = (delta_f @ tc.W_dec).sum(dim=-1)
    by_token = df.groupby("token_idx")["attribution"].sum().sort_index().to_numpy()
    features = (df[df.feature_idx >= 0].groupby("token_idx")["attribution"]
                .sum().sort_index().to_numpy())

    assert seen == {"allow_transcoders": True}
    assert features == pytest.approx(expected_features.tolist(), abs=1e-4)
    assert by_token == pytest.approx((y_p - y_c).sum(dim=-1).tolist(), abs=1e-4)
