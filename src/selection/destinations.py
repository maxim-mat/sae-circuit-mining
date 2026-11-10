"""Which node instances an edge run computes edges INTO, per filtering method.

An instance is one row of the node hive: (example, position, site, feature),
feature_idx = -1 being the SAE error node, which is a node like any other here
(Marks et al., arXiv:2403.19647). Edge values do not depend on the method -- only
which destinations get a backward pass -- so a method is just a rule over rows:

  case        |IE| of the instance itself > threshold. Marks' own jvp gates its
              downstream side this way (`effects.abs() > node_threshold` on
              per-example, per-position effects).
  marks       every moved (IE != 0) instance of an activity whose |mean IE| >
              threshold. The mean is over ALL examples (absent = 0), of the
              per-example IE, which `aggregation` defines: "sum" sums positions
              first so an activity is (site, feature); "none" keeps positions so
              it is (site, position, feature). Marks et al. §3.1, App. A.2.
  marks+case  marks AND case.
  recurrence  moved instances of activities whose per-example |IE| > threshold
              in at least `min_frac` of all examples.
  mass        moved instances of the smallest activity set holding `mass` of
              the total |mean IE| (node pruning by share of effect, as in
              Ameisen et al. 2025, "Circuit Tracing").
  ttest_bh    moved instances of activities whose mean IE differs from 0 at
              BH-FDR `fdr` (one-sample t-test over all examples).

Error nodes go through the same rules as features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ALPHABET_RULES = ("marks", "recurrence", "mass", "ttest_bh")


def _keys(raw: pd.DataFrame, by_position: bool) -> np.ndarray:
    """One int64 per activity: site | position+1 (0 when summed) | feature+1."""
    token = raw["token_idx"].to_numpy(np.int64) + 1 if by_position else 0
    return ((raw["site_ordinal"].to_numpy(np.int64) << 40) | (token << 24)
            | (raw["feature_idx"].to_numpy(np.int64) + 1))


def _alphabet(raw: pd.DataFrame, spec: dict, n: int) -> np.ndarray:
    """Bool per raw row: does the row's activity pass the method's activity rule?"""
    aggregation = spec.get("aggregation", "sum")
    if aggregation not in ("sum", "none"):
        raise ValueError(f"Unknown aggregation {aggregation!r}; expected 'sum' or 'none'")
    keys = _keys(raw, aggregation == "none")
    per_example = (pd.DataFrame({"key": keys, "ex": raw["example_idx"].to_numpy(),
                                 "ie": raw["attribution"].to_numpy(np.float64)})
                   .groupby(["key", "ex"], sort=False)["ie"].sum())
    by_key = per_example.groupby(level="key", sort=False)
    mean = by_key.sum() / n

    rule = spec["rule"]
    if rule == "marks":
        keep = mean.abs() > spec["threshold"]
    elif rule == "recurrence":
        hits = (per_example.abs() > spec["threshold"]).groupby(level="key", sort=False).sum()
        keep = hits / n >= spec["min_frac"]
    elif rule == "mass":
        ranked = mean.abs().sort_values(ascending=False)
        n_keep = np.searchsorted((ranked.cumsum() / ranked.sum()).to_numpy(), spec["mass"]) + 1
        keep = mean.index.isin(ranked.index[:n_keep])
        keep = pd.Series(keep, index=mean.index)
    else:                                                     # ttest_bh
        if n < 2:
            raise ValueError("ttest_bh needs at least 2 examples")
        sum_sq = (per_example ** 2).groupby(level="key", sort=False).sum()
        var = ((sum_sq - n * mean ** 2) / (n - 1)).clip(lower=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = mean.to_numpy() / np.sqrt(var.to_numpy() / n)
        p = np.nan_to_num(2 * stats.t.sf(np.abs(t), n - 1), nan=1.0)
        keep = pd.Series(stats.false_discovery_control(p, method="bh") <= spec["fdr"],
                         index=mean.index)
    return np.isin(keys, keep.index[keep.to_numpy()].to_numpy())


def destination_masks(raw: pd.DataFrame, methods: dict[str, dict], n_examples: int) -> pd.DataFrame:
    """Bool column per method over the rows of `raw` (example_idx, site_ordinal,
    token_idx, feature_idx, attribution): True where the row is a destination.

    `n_examples` is the node run's example count, the denominator of every mean.
    """
    ie = raw["attribution"].to_numpy(np.float64)
    out = {}
    for name, spec in methods.items():
        rule = spec.get("rule")
        if rule == "case":
            out[name] = np.abs(ie) > spec["threshold"]
        elif rule == "marks+case":
            out[name] = (np.abs(ie) > spec["threshold"]) & _alphabet(raw, {**spec, "rule": "marks"},
                                                                     n_examples)
        elif rule in ALPHABET_RULES:
            out[name] = (ie != 0) & _alphabet(raw, spec, n_examples)
        else:
            raise ValueError(f"Unknown destination rule {rule!r} for method {name!r}; expected "
                             f"one of {('case', 'marks+case', *ALPHABET_RULES)}")
    return pd.DataFrame(out, index=raw.index)
