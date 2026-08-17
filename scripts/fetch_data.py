"""Download Marks et al.'s subject-verb agreement stimuli into data/marks/.

The datasets are not committed (see .gitignore: raw data never goes in the
repo), so this is the one setup step the SVA configs need on a fresh machine.

    python -m scripts.fetch_data

Source: saprmarks/feature-circuits, arXiv:2403.19647. Files are jsonl despite
the .json extension. Structures: simple, nounpp, rc, within_rc; *_train for
discovery, *_test for the faithfulness evaluation.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/saprmarks/feature-circuits/main/data"
STRUCTURES = ["simple", "nounpp", "rc", "within_rc"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/marks"))
    p.add_argument("--structures", nargs="*", default=STRUCTURES,
                   help=f"Subset to fetch (default: all of {STRUCTURES}).")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for structure in args.structures:
        for split in ("train", "test"):
            name = f"{structure}_{split}.json"
            dest = args.out / name
            if dest.exists():
                print(f"  have {dest}")
                continue
            url = f"{BASE}/{name}"
            print(f"  fetching {url}")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                raise SystemExit(f"Failed to fetch {url}: {e}") from e
            rows = sum(1 for _ in dest.open(encoding="utf-8"))
            print(f"    -> {dest} ({rows:,} rows)")


if __name__ == "__main__":
    main()
