"""Sanity-check a suite's SAEs: variance explained and L0 per site.

Marks report per-layer variance explained in their Appendix B.1 tables
(arXiv:2403.19647); anything far below those is a loading bug, not a research
finding. The pythia-marks suite serves *processed* (folded/normalised)
conversions of their dictionary_learning dictionaries, so this is the check
that the conversion survived -- nothing downstream is trustworthy if it did not.

Usage:
    python -m scripts.check_sae_reconstruction --suite pythia-marks
    python -m scripts.check_sae_reconstruction --suite gpt2-res --n-seqs 50

Acceptance: variance explained >= 0.80 at every site, L0 in the tens to low
hundreds. Writes a markdown table to docs/reference-numbers.md.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.models.suites import build_suite, load_model_and_saes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VE_FLOOR = 0.80


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="pythia-marks")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset", default="NeelNanda/pile-10k")
    p.add_argument("--n-seqs", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    # Its own file, overwritten per run: prepending into a curated document
    # duplicates that document's headings and grows it without bound.
    p.add_argument("--out", type=Path, default=Path("docs/sae-reconstruction.md"))
    # GPU courtesy: this is a diagnostic, not a training job, and it should not
    # evict whatever else is resident. One sequence at a time keeps the working
    # set at seq_len x d_model per site; the allocator cap is a hard ceiling in
    # case a future edit forgets that.
    p.add_argument("--gpu-mem-fraction", type=float, default=0.25,
                   help="Cap this process's share of GPU memory (0 = uncapped).")
    # The BOS position carries an enormous, atypical activation that SAEs
    # reconstruct badly; including it depresses variance explained at every site
    # and is not what published SAE evaluations report.
    p.add_argument("--keep-bos", action="store_true",
                   help="Score the BOS position too (default: skip it).")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and args.gpu_mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
        logger.info("GPU memory capped at %.0f%% of device",
                    args.gpu_mem_fraction * 100)

    from datasets import load_dataset

    suite = build_suite(args.suite)
    model, saes = load_model_and_saes(suite, args.device)
    dataset = load_dataset(args.dataset, split="train", streaming=True)

    texts = []
    for row in dataset:
        texts.append(row["text"])
        if len(texts) >= args.n_seqs:
            break

    # Accumulate over sequences rather than concatenating activations: 50 x 128
    # tokens x d_model x 19 sites does not need to be resident at once.
    stats = {site.sae_id: {"sq_err": 0.0, "sq_dev": 0.0, "l0": 0.0, "n": 0}
             for site in suite.sites}

    for text in texts:
        tokens = model.to_tokens(text)[:, :args.seq_len].to(args.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=[site.hook for site in suite.sites])
            for site in suite.sites:
                sae = saes[site.sae_id]
                x = cache[site.hook]
                if not args.keep_bos:
                    x = x[:, 1:, :]
                f = sae.encode(x)
                x_hat = sae.decode(f)
                s = stats[site.sae_id]
                s["sq_err"] += float(((x - x_hat) ** 2).sum())
                s["sq_dev"] += float(((x - x.mean(dim=(0, 1), keepdim=True)) ** 2).sum())
                s["l0"] += float((f > 0).float().sum(-1).sum())
                s["n"] += x.shape[0] * x.shape[1]

    rows, failures = [], []
    for site in suite.sites:
        s = stats[site.sae_id]
        ve = 1.0 - s["sq_err"] / s["sq_dev"] if s["sq_dev"] else float("nan")
        l0 = s["l0"] / s["n"] if s["n"] else float("nan")
        rows.append((site, ve, l0))
        if not (ve >= VE_FLOOR):
            failures.append((site.sae_id, ve))

    lines = [
        "# Reference numbers",
        "",
        f"## Q1 — SAE reconstruction, suite `{args.suite}`",
        "",
        f"{args.n_seqs} sequences x {args.seq_len} tokens from `{args.dataset}`, "
        f"seed {args.seed}, BOS position {'included' if args.keep_bos else 'skipped'}. "
        f"Acceptance: variance explained >= {VE_FLOOR:.2f}.",
        "",
        "| ordinal | kind | block | sae_id | variance explained | mean L0 |",
        "|---|---|---|---|---|---|",
    ]
    for site, ve, l0 in rows:
        lines.append(f"| {site.ordinal} | {site.kind} | {site.block} | "
                     f"`{site.sae_id}` | {ve:.4f} | {l0:.1f} |")
    lines += ["", ("**FAIL** — sites below the floor: "
                   + ", ".join(f"{sid} ({ve:.3f})" for sid, ve in failures))
              if failures else "**PASS** — every site is above the floor.", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if failures:
        raise SystemExit(
            "SAE reconstruction below the acceptance floor; investigate the "
            "release before running the reproduction (plan Task 12 Step 1)."
        )


if __name__ == "__main__":
    main()
