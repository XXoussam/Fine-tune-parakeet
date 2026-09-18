#!/usr/bin/env python3
"""Build NeMo ASR manifest(s) (jsonl) for fine-tuning finetune_parakeet.py.

Each output line is: {"audio_filepath": "...", "duration": 12.3, "text": "..."}

Three input layouts are supported:

1. --csv path/to/data.csv with two columns "audio_filepath,text" (a header
   row is optional; audio paths may be relative to the CSV's directory).

2. --audio-dir path/to/wavs, where each "name.wav" has a matching "name.txt"
   sibling file containing its transcript.

3. --dataset-json path/to/dataset.json - a JSON array of objects with
   "clip_path" and "human_labeled" keys (this project's labeled dataset
   format; paths may be relative to the JSON file's directory).

By default all entries go to one manifest (--out). Pass --val-split to
instead hold out a fraction for validation and write two manifests
(--train-out / --val-out) - finetune_parakeet.py needs both a train and a
validation manifest.

Usage:
    python prepare_manifest.py --csv data/train.csv --out data/train_manifest.json
    python prepare_manifest.py --audio-dir data/train_wavs --out data/train_manifest.json
    python prepare_manifest.py --dataset-json data/dataset.json \
        --val-split 0.1 --train-out data/train_manifest.json --val-out data/val_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import wave
from pathlib import Path


def audio_duration_seconds(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.getnframes() / float(wav_file.getframerate())
    import soundfile as sf  # only needed for non-wav formats

    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def entries_from_csv(csv_path: Path) -> list[dict]:
    entries = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if rows and rows[0][:2] == ["audio_filepath", "text"]:
        rows = rows[1:]
    for row in rows:
        if not row:
            continue
        audio_filepath, text = row[0].strip(), row[1].strip()
        audio_path = Path(audio_filepath)
        if not audio_path.is_absolute():
            audio_path = (csv_path.parent / audio_path).resolve()
        entries.append({"audio_filepath": audio_path, "text": text})
    return entries


def entries_from_audio_dir(audio_dir: Path) -> list[dict]:
    entries = []
    for audio_path in sorted(audio_dir.glob("*.wav")):
        transcript_path = audio_path.with_suffix(".txt")
        if not transcript_path.is_file():
            print(f"Skipping {audio_path.name}: no matching .txt transcript")
            continue
        text = transcript_path.read_text(encoding="utf-8").strip()
        entries.append({"audio_filepath": audio_path, "text": text})
    return entries


def entries_from_dataset_json(dataset_json_path: Path) -> list[dict]:
    with dataset_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for item in data:
        audio_filepath = item.get("clip_path")
        text = (item.get("human_labeled") or "").strip()
        if not audio_filepath or not text:
            continue
        audio_path = Path(audio_filepath)
        if not audio_path.is_absolute():
            audio_path = (dataset_json_path.parent / audio_path).resolve()
        entries.append({"audio_filepath": audio_path, "text": text})
    return entries


def write_manifest(entries: list[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as out_file:
        for entry in entries:
            audio_path: Path = entry["audio_filepath"]
            if not audio_path.is_file():
                print(f"Skipping missing audio file: {audio_path}")
                continue
            duration = audio_duration_seconds(audio_path)
            out_file.write(
                json.dumps({"audio_filepath": str(audio_path), "duration": duration, "text": entry["text"]})
                + "\n"
            )
            written += 1
    print(f"Wrote {written} entries to {out_path}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV file with audio_filepath,text columns")
    source.add_argument("--audio-dir", type=Path, help="Directory of name.wav + name.txt pairs")
    source.add_argument("--dataset-json", type=Path, help="dataset.json with clip_path/human_labeled entries")

    parser.add_argument("--out", type=Path, help="Output manifest path (jsonl) - single-manifest mode")
    parser.add_argument(
        "--val-split", type=float, default=None, help="Fraction held out for validation, e.g. 0.1 - splits into two manifests instead of one"
    )
    parser.add_argument("--train-out", type=Path, help="Train manifest output path (used with --val-split)")
    parser.add_argument("--val-out", type=Path, help="Validation manifest output path (used with --val-split)")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed before splitting (with --val-split)")
    args = parser.parse_args()

    if args.val_split is not None:
        if not args.train_out or not args.val_out:
            parser.error("--val-split requires --train-out and --val-out")
    elif not args.out:
        parser.error("--out is required unless --val-split is set")

    if args.csv:
        entries = entries_from_csv(args.csv)
    elif args.audio_dir:
        entries = entries_from_audio_dir(args.audio_dir)
    else:
        entries = entries_from_dataset_json(args.dataset_json)

    if not entries:
        raise SystemExit("No entries found - check the input path and layout.")

    if args.val_split is None:
        written = write_manifest(entries, args.out)
        return 0 if written else 1

    random.Random(args.seed).shuffle(entries)
    split_at = round(len(entries) * (1 - args.val_split))
    train_written = write_manifest(entries[:split_at], args.train_out)
    val_written = write_manifest(entries[split_at:], args.val_out)
    return 0 if (train_written and val_written) else 1


if __name__ == "__main__":
    raise SystemExit(main())
