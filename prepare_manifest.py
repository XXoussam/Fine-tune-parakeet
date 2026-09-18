#!/usr/bin/env python3
"""Build a NeMo ASR manifest (jsonl) for fine-tuning finetune_parakeet.py.

Each output line is: {"audio_filepath": "...", "duration": 12.3, "text": "..."}

Two input layouts are supported:

1. --csv path/to/data.csv with two columns "audio_filepath,text" (a header
   row is optional; audio paths may be relative to the CSV's directory).

2. --audio-dir path/to/wavs, where each "name.wav" has a matching "name.txt"
   sibling file containing its transcript.

Usage:
    python prepare_manifest.py --csv data/train.csv --out data/train_manifest.json
    python prepare_manifest.py --audio-dir data/train_wavs --out data/train_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV file with audio_filepath,text columns")
    source.add_argument("--audio-dir", type=Path, help="Directory of name.wav + name.txt pairs")
    parser.add_argument("--out", type=Path, required=True, help="Output manifest path (jsonl)")
    args = parser.parse_args()

    entries = entries_from_csv(args.csv) if args.csv else entries_from_audio_dir(args.audio_dir)
    if not entries:
        raise SystemExit("No entries found - check the input path and layout.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as out_file:
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

    print(f"Wrote {written} entries to {args.out}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
