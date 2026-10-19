import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pm4py")
from scripts.simple_discovery import filter_rows, read_nodes  # noqa: E402


def test_hive_sites_sort_numerically_not_as_strings(tmp_path):
    df = pd.DataFrame({"example_idx": [0, 0], "site_ordinal": [10, 3],
                       "feature_idx": [1, 2], "attribution": [0.5, 0.5]})
    pq.write_to_dataset(pa.Table.from_pandas(df), str(tmp_path / "nodes"),
                        partition_cols=["site_ordinal"])
    out = read_nodes(tmp_path).sort_values("site_ordinal")
    assert list(out["site_ordinal"]) == [3, 10]


def test_script_does_not_import_torch():
    # torch's OpenMP runtime aborts pm4py's ILP solver (OMP Error #15) on the
    # Windows anaconda env, so the script must stay torch-free.
    probe = "import sys, scripts.simple_discovery; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "False"


def _raw():
    return pd.DataFrame({
        "example_idx": [0, 0, 0, 1, 1],
        "token_idx":   [0, 1, 2, 0, 1],
        "feature_idx": [7, 7, -1, 7, 9],
        "attribution": [0.5, 0.1, 0.9, -0.3, 0.05],
    })


def test_filters_each_row_on_its_own_abs_ie_strictly():
    # Feature 7's example mean is (0.5 + 0.1 - 0.3) / 2 = 0.15, but the
    # filter is per row: 0.1 fails (strict >), -0.3 passes on |IE|.
    out = filter_rows(_raw(), 0.1)
    assert list(out["attribution"]) == [0.5, -0.3]


def test_drops_sae_error_nodes_whatever_their_effect():
    assert -1 not in set(filter_rows(_raw(), 0.0)["feature_idx"])
