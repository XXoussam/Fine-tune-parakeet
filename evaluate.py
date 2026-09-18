#!/usr/bin/env python3
"""Evaluate + visualize a fine-tuned Parakeet run - runs entirely locally.

No NeMo, no PyTorch, no GPU: this only reads the validation_predictions.jsonl
that finetune_parakeet.py writes on the VM after training (raw reference and
hypothesis text pairs), computes WER itself, and plots the results. Copy that
one small file back from the VM (see the comment at the top of
finetune_parakeet.py for the scp command) and point this script at it.

Usage:
    python evaluate.py --predictions validation_predictions.jsonl --out-dir eval_report
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Word-level edit distance (substitutions+insertions+deletions), reference word count."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    r, h = len(ref_words), len(hyp_words)

    prev_row = list(range(h + 1))
    for i in range(1, r + 1):
        curr_row = [i] + [0] * h
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(prev_row[j - 1], prev_row[j], curr_row[j - 1])
        prev_row = curr_row

    return prev_row[h], r


def load_predictions(path: Path) -> list[dict]:
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def evaluate(entries: list[dict]) -> list[dict]:
    rows = []
    for entry in entries:
        edits, ref_word_count = word_errors(entry.get("reference", ""), entry.get("hypothesis", ""))
        wer = edits / ref_word_count if ref_word_count else (0.0 if not entry.get("hypothesis") else 1.0)
        rows.append(
            {
                "audio_filepath": entry.get("audio_filepath", ""),
                "duration": entry.get("duration"),
                "reference": entry.get("reference", ""),
                "hypothesis": entry.get("hypothesis", ""),
                "edits": edits,
                "ref_words": ref_word_count,
                "wer": wer,
            }
        )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audio_filepath", "duration", "wer", "ref_words", "reference", "hypothesis"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def plot_reports(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    wers = [r["wer"] for r in rows]
    durations = [r["duration"] for r in rows if r["duration"] is not None]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(wers, bins=20)
    ax.set_xlabel("Per-sample WER")
    ax.set_ylabel("Count")
    ax.set_title("WER distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "wer_distribution.png", dpi=150)
    plt.close(fig)

    if durations:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([r["duration"] for r in rows], wers, s=10, alpha=0.6)
        ax.set_xlabel("Clip duration (s)")
        ax.set_ylabel("Per-sample WER")
        ax.set_title("WER vs. clip duration")
        fig.tight_layout()
        fig.savefig(out_dir / "wer_vs_duration.png", dpi=150)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True, help="validation_predictions.jsonl from the VM")
    parser.add_argument("--out-dir", type=Path, default=Path("eval_report"), help="Where to write CSV/plots")
    parser.add_argument("--top-n-worst", type=int, default=10, help="How many worst samples to print")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots (CSV + summary only)")
    args = parser.parse_args()

    entries = load_predictions(args.predictions)
    if not entries:
        raise SystemExit(f"No entries found in {args.predictions}")

    rows = evaluate(entries)

    total_edits = sum(r["edits"] for r in rows)
    total_ref_words = sum(r["ref_words"] for r in rows)
    corpus_wer = total_edits / total_ref_words if total_ref_words else 0.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "per_sample_wer.csv")

    print(f"Samples evaluated : {len(rows)}")
    print(f"Corpus WER        : {corpus_wer:.4f} ({total_edits} edits / {total_ref_words} reference words)")
    print(f"CSV written to    : {args.out_dir / 'per_sample_wer.csv'}")

    worst = sorted(rows, key=lambda r: r["wer"], reverse=True)[: args.top_n_worst]
    print(f"\nWorst {len(worst)} samples by WER:")
    for row in worst:
        print(f"  wer={row['wer']:.2f}  {row['audio_filepath']}")
        print(f"    ref: {row['reference']}")
        print(f"    hyp: {row['hypothesis']}")

    if not args.no_plots:
        plot_reports(rows, args.out_dir)
        print(f"\nPlots written to  : {args.out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
