"""Position-resolved edges (src/attribution/positional_edges.py) on a fake model.

The fake is linear between sites, with a causal position mix at one site, so the
edge identities hold exactly:

  * edges into a downstream instance (q, d) sum, over every upstream (p, u) and
    upstream error, to left[q, d] * (change in d when only the upstream site is
    patched), where left is the integrated-gradients gradient at the downstream
    site (Marks et al., arXiv:2403.19647, `jvp` with `_pe_ig` gradients);
  * edges into the metric are the upstream site's own node IEs.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.attribution import positional_edges as pe
from src.attribution.edge_template import Y_SITE, EdgeSpec
from src.attribution.ig import integrated_gradient_grad_x
from src.data.loaders import Example
from src.models.suites import Site, Suite
from src.utils.config import Config
from src.utils.manifest import stage_params

SEQ, D, D_SAE, VOCAB, STEPS = 4, 6, 5, 8, 4
CLEAN, PATCH = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 5, 3, 6]])


class FakeSAE:
    def __init__(self):
        self.W_enc = torch.randn(D, D_SAE)
        self.W_dec = torch.randn(D_SAE, D)
        self.cfg = SimpleNamespace(d_sae=D_SAE, metadata=None)

    def encode(self, x):
        return x @ self.W_enc

    def decode(self, f):
        return f @ self.W_dec


class FakeModel:
    """TransformerLens hook protocol; `causal` sites mix positions p <= q first."""
    def __init__(self, hooks, causal=()):
        self.order, self.causal = list(hooks), set(causal)
        self.E = torch.randn(VOCAB, D)
        self.M = {h: torch.randn(D, D) / D ** 0.5 for h in self.order}
        self.U = torch.randn(D, VOCAB)
        self.mix = torch.tril(torch.full((SEQ, SEQ), 0.3)) + 0.7 * torch.eye(SEQ)
        self.hooks = []

    def add_hook(self, name, fn, dir="fwd"):
        self.hooks.append((name, fn))

    def reset_hooks(self):
        self.hooks = []

    def zero_grad(self, set_to_none=True):
        pass

    def run_with_hooks(self, tokens, fwd_hooks):
        saved, self.hooks = self.hooks, self.hooks + list(fwd_hooks)
        try:
            return self(tokens)
        finally:
            self.hooks = saved

    def to_tokens(self, text, prepend_bos=False):
        return {"clean": CLEAN, "patch": PATCH, "short": PATCH[:, :3]}[text]

    def __call__(self, tokens):
        x = self.E[tokens]
        for h in self.order:
            if h in self.causal:
                x = self.mix @ x
            x = x @ self.M[h]
            for name, fn in self.hooks:
                if name == h:
                    x = fn(x, hook=SimpleNamespace(name=h))
        return x @ self.U


def metric(logits, **_):
    return logits[0, -1, 0] ** 2 + logits[0, -1, 1]      # nonlinear: IG != clean gradient


def _setup(causal=("h1",)):
    torch.manual_seed(0)
    sites = [Site(ordinal=i, block=i, kind="resid", release="r", sae_id=f"s{i}", hook=f"h{i}")
             for i in range(3)]
    model = FakeModel([s.hook for s in sites], causal)
    saes = {s.sae_id: FakeSAE() for s in sites}
    return sites, model, saes


def _acts(model, hook, tokens, replace=None):
    out = {}

    def cap(t, hook):
        out["x"] = t[0].detach()
        return t

    hooks = [(hook, cap)]
    if replace is not None:
        hooks.insert(0, replace)
    model.run_with_hooks(tokens, hooks)
    return out["x"]


def _frame(cols):
    return pd.DataFrame(cols)


def _all_destinations(sae_ordinal):
    return {sae_ordinal: [(q, d) for q in range(SEQ) for d in [*range(D_SAE), -1]]}


def test_edges_into_an_instance_sum_to_its_ig_weighted_change():
    sites, model, saes = _setup()
    src, dst = sites[0], sites[2]
    edges = _frame(pe.example_edges(model, saes, [EdgeSpec(src, dst)], CLEAN, PATCH, metric,
                                    None, _all_destinations(2), STEPS))

    x_patch_src = _acts(model, src.hook, PATCH)
    y_clean = _acts(model, dst.hook, CLEAN)
    y_up = _acts(model, dst.hook, CLEAN, (src.hook, lambda t, hook: x_patch_src.unsqueeze(0)))
    sae = saes[dst.sae_id]
    g_change = sae.encode(y_up) - sae.encode(y_clean)
    res = lambda y: y - sae.decode(sae.encode(y))
    grad_x = integrated_gradient_grad_x(model, CLEAN, dst.hook, y_clean, _acts(model, dst.hook, PATCH),
                                        metric, None, STEPS)
    left = grad_x @ sae.W_dec.T

    got = edges.groupby(["dst_token", "dst_feature"])["edge_ie"].sum()
    assert len(got) > 0
    for q in range(SEQ):
        for d in range(D_SAE):           # absent = no non-zero edge into it
            assert got.get((q, d), 0.0) == pytest.approx(float(left[q, d] * g_change[q, d]),
                                                         rel=1e-4, abs=1e-6)
        expected_err = float((grad_x[q] * (res(y_up)[q] - res(y_clean)[q])).sum())
        assert got.get((q, -1), 0.0) == pytest.approx(expected_err, rel=1e-4, abs=1e-6)


def test_the_left_vector_is_the_ig_gradient_not_the_clean_one():
    sites, model, saes = _setup()
    edges = _frame(pe.example_edges(model, saes, [EdgeSpec(sites[1], sites[2])], CLEAN, PATCH, metric,
                                    None, _all_destinations(2), 1))
    edges_ig = _frame(pe.example_edges(model, saes, [EdgeSpec(sites[1], sites[2])], CLEAN, PATCH, metric,
                                       None, _all_destinations(2), STEPS))
    assert not np.allclose(edges["edge_ie"].sum(), edges_ig["edge_ie"].sum())


def test_positions_are_resolved_and_causal():
    sites, model, saes = _setup()
    across = _frame(pe.example_edges(model, saes, [EdgeSpec(sites[0], sites[2])], CLEAN, PATCH, metric,
                                     None, _all_destinations(2), STEPS))
    same = _frame(pe.example_edges(model, saes, [EdgeSpec(sites[1], sites[2])], CLEAN, PATCH, metric,
                                   None, _all_destinations(2), STEPS))
    assert (across["src_token"] <= across["dst_token"]).all()
    assert (across["src_token"] < across["dst_token"]).any()       # crossed the causal mix
    assert (same["src_token"] == same["dst_token"]).all()
    # Tokens 0 and 2 are identical in clean and patch and nothing mixes before h0,
    # so nothing at those upstream positions moved.
    assert not across["src_token"].isin([0, 2]).any()


def test_only_requested_destinations_get_edges():
    sites, model, saes = _setup()
    edges = _frame(pe.example_edges(model, saes, [EdgeSpec(sites[1], sites[2])], CLEAN, PATCH, metric,
                                    None, {2: [(3, 1), (3, -1)]}, STEPS))
    # The metric reads the last position only, so that is where gradient reaches.
    assert set(zip(edges["dst_token"], edges["dst_feature"])) == {(3, 1), (3, -1)}


def test_a_stop_gradient_on_the_only_path_removes_the_edges():
    sites, model, saes = _setup()
    edges = pe.example_edges(model, saes, [EdgeSpec(sites[0], sites[2], stop=(sites[1],))],
                             CLEAN, PATCH, metric, None, _all_destinations(2), STEPS)
    assert len(edges["edge_ie"]) == 0


def test_edges_into_the_metric_are_the_node_ies():
    sites, model, saes = _setup()
    last = sites[2]
    edges = _frame(pe.example_edges(model, saes, [EdgeSpec(last, Y_SITE)], CLEAN, PATCH, metric,
                                    None, {}, STEPS))
    x_c, x_p = _acts(model, last.hook, CLEAN), _acts(model, last.hook, PATCH)
    grad_x = integrated_gradient_grad_x(model, CLEAN, last.hook, x_c, x_p, metric, None, STEPS)
    sae = saes[last.sae_id]
    assert (edges["dst_ordinal"] == Y_SITE.ordinal).all()
    assert edges["edge_ie"].sum() == pytest.approx(float((grad_x * (x_p - x_c)).sum()), rel=1e-4)
    f_ie = (grad_x @ sae.W_dec.T) * (sae.encode(x_p) - sae.encode(x_c))
    feat = edges[edges.src_feature >= 0].set_index(["src_token", "src_feature"])["edge_ie"]
    for (p, u), v in feat.items():
        assert v == pytest.approx(float(f_ie[p, u]), rel=1e-4)


# --- one store per method -------------------------------------------------------

NODE_SEED = 7


def _node_run(tmp_path, n_examples=3, token_ids=CLEAN[0].tolist(), skipped=()):
    """A finished node run whose hive the destination rules can read. Its token_id
    column records the clean prompt, as the node stage writes it; `skipped`
    examples have no rows, as when the node stage skipped them."""
    cfg = Config()
    cfg.suite.name, cfg.suite.device, cfg.data.path = "s", "cpu", "x"
    cfg.attribution.ig_steps = STEPS
    cfg.seed = NODE_SEED
    node_dir = tmp_path / "ds" / "attributions" / "nodes" / "cf"
    rows = []
    for ex in (e for e in range(n_examples) if e not in skipped):
        for site in range(3):
            rows.append((ex, site, site, "resid", 3, token_ids[3], 1, 0.5 if site == 1 else 0.05))
            rows.append((ex, site, site, "resid", 1, token_ids[1], 2, 0.05))
            rows.append((ex, site, site, "resid", 3, token_ids[3], -1, 0.2 if ex == 0 else 0.0))
    df = pd.DataFrame(rows, columns=["example_idx", "site_ordinal", "block", "kind", "token_idx",
                                     "token_id", "feature_idx", "attribution"])
    from src.attribution.nodes import NODE_SCHEMA
    pq.write_to_dataset(pa.Table.from_pandas(df, schema=NODE_SCHEMA, preserve_index=False),
                        root_path=str(node_dir / "nodes"), partition_cols=["kind", "site_ordinal"])
    (node_dir / "manifest.json").write_text(json.dumps({
        "stage": "node_attribution", "params": stage_params(cfg, "node_attribution"),
        "git_commit": "abc", "n_examples": n_examples - len(skipped), "n_skipped": len(skipped)}))
    return node_dir


@pytest.fixture
def fake_libs(monkeypatch):
    sites, model, saes = _setup()
    suite = Suite("s", "m", False, tuple(sites))
    calls = {"loads": 0}

    def load(*a, **k):
        calls["loads"] += 1
        return model, saes

    monkeypatch.setattr(pe, "build_suite", lambda name: suite)
    monkeypatch.setattr(pe, "load_model_and_saes", load)
    monkeypatch.setattr(pe, "prepend_bos_for", lambda saes: False)
    monkeypatch.setattr(pe, "get_metric", lambda name: metric)
    def examples(data_cfg, model, seed):
        calls["seed"] = seed
        return [Example(clean_prefix="clean", clean_answer="", patch_prefix="patch")
                for _ in range(3)]

    monkeypatch.setattr(pe, "load_examples", examples)
    return calls


METHODS = {"case": {"rule": "case", "threshold": 0.1},
           "marks": {"rule": "marks", "threshold": 0.1, "aggregation": "sum"}}


def test_each_method_gets_its_own_folder_and_manifest(tmp_path, fake_libs):
    node_dir = _node_run(tmp_path)
    pe.run_positional_edges(node_dir, METHODS)

    case, marks = (pe.edge_dir(node_dir, m) for m in ("case", "marks"))
    assert case == tmp_path / "ds" / "attributions" / "edges" / "cf" / "case"
    for out, name in ((case, "case"), (marks, "marks")):
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["params"]["method"] == METHODS[name]
        assert manifest["params"]["node_run"]["git_commit"] == "abc"
        assert manifest["params"]["node_run"]["path"] == node_dir.as_posix()
    assert fake_libs["loads"] == 1                     # one pass serves every method

    c, m = pe.read_edges(case), pe.read_edges(marks)
    assert c.dtypes["dst_ordinal"] == "int16"           # not the hive's Categorical
    inner = lambda df: set(zip(df.example_idx, df.dst_ordinal, df.dst_token, df.dst_feature)) - {
        (e, -1, -1, -1) for e in range(3)}
    # case: |IE| > 0.1 -> feature 1 at site 1 (all examples), errors in example 0. Site 0
    # is a destination too but has no incoming edge in the template.
    assert inner(c) == ({(e, 1, 3, 1) for e in range(3)} | {(0, 1, 3, -1), (0, 2, 3, -1)})
    # marks: mean over 3 examples -> feature 1 at site 1 only; errors average 0.2 / 3.
    assert inner(m) == {(e, 1, 3, 1) for e in range(3)}
    # Shared destinations carry identical edges; metric edges are in every store.
    key = ["example_idx", "src_ordinal", "src_token", "src_feature", "dst_ordinal", "dst_token",
           "dst_feature"]
    shared = c.merge(m, on=key, suffixes=("_c", "_m"))
    assert len(shared) == len(m) and np.allclose(shared.edge_ie_c, shared.edge_ie_m)
    assert (c.dst_ordinal == -1).sum() == (m.dst_ordinal == -1).sum() > 0


def test_finished_methods_are_skipped_and_changed_ones_refused(tmp_path, fake_libs):
    node_dir = _node_run(tmp_path)
    pe.run_positional_edges(node_dir, METHODS)
    pe.run_positional_edges(node_dir, METHODS)
    assert fake_libs["loads"] == 1

    changed = {"case": {"rule": "case", "threshold": 0.2}}
    with pytest.raises(RuntimeError, match="threshold"):
        pe.run_positional_edges(node_dir, changed)
    pe.run_positional_edges(node_dir, changed, force=True)
    assert json.loads((pe.edge_dir(node_dir, "case") / "manifest.json").read_text()
                      )["params"]["method"]["threshold"] == 0.2


def test_a_node_run_with_different_examples_is_refused(tmp_path, fake_libs, monkeypatch):
    node_dir = _node_run(tmp_path)
    monkeypatch.setattr(pe, "load_examples", lambda *a: [])
    with pytest.raises(RuntimeError, match="examples"):
        pe.run_positional_edges(node_dir, METHODS)


def test_skipped_examples_are_logged_and_counted(tmp_path, fake_libs, monkeypatch, caplog):
    node_dir = _node_run(tmp_path, skipped=(1,))
    monkeypatch.setattr(pe, "load_examples", lambda *a: [
        Example(clean_prefix="clean", clean_answer="", patch_prefix=p)
        for p in ("patch", "short", "patch")])

    with caplog.at_level("WARNING", logger=pe.__name__):
        pe.run_positional_edges(node_dir, METHODS)

    assert any("Example 1" in r.message and "skipping" in r.message for r in caplog.records)
    for name in METHODS:
        manifest = json.loads((pe.edge_dir(node_dir, name) / "manifest.json").read_text())
        assert manifest["n_skipped"] == 1
        assert manifest["n_examples"] == 2
        assert set(pe.read_edges(pe.edge_dir(node_dir, name)).example_idx) == {0, 2}


def test_examples_are_loaded_with_the_node_runs_seed(tmp_path, fake_libs):
    pe.run_positional_edges(_node_run(tmp_path), METHODS)
    assert fake_libs["seed"] == NODE_SEED


def test_a_prompt_that_differs_from_the_node_run_is_refused(tmp_path, fake_libs):
    # Same count and length, different prompt: the node hive saw token 9 at position 3.
    node_dir = _node_run(tmp_path, token_ids=[1, 2, 3, 9])
    with pytest.raises(RuntimeError, match="prompt"):
        pe.run_positional_edges(node_dir, METHODS)
    assert not (pe.edge_dir(node_dir, "case") / "manifest.json").exists()
