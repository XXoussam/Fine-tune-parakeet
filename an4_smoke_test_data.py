#!/usr/bin/env python3
"""Download NVIDIA's AN4 tutorial dataset and build NeMo manifests.

Same data nemo.ipynb uses on Colab, refactored into a standalone script so
a headless GPU VM (no Jupyter) can build it too. Used for both a quick
pipeline smoke test and a full AN4 training dry run, before pointing
finetune_parakeet.py at the real dataset.

Needs `ffmpeg` on PATH (sph -> wav conversion): `apt-get install -y ffmpeg`

Usage:
    python an4_smoke_test_data.py --out-dir an4_data
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import urllib.request
import wave
from pathlib import Path

AN4_URL = "https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz"


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


def download_and_extract(out_dir: Path) -> Path:
    an4_dir = out_dir / "an4"
    if an4_dir.is_dir():
        print(f"Found existing extracted dataset: {an4_dir}")
        return an4_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "an4_sphere.tar.gz"
    print(f"Downloading {AN4_URL}")
    urllib.request.urlretrieve(AN4_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=out_dir)
    return an4_dir


def convert_and_build_manifest(an4_dir: Path, target_dir: Path, split: str) -> Path:
    wavs_dir = target_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    sph_list = sorted(an4_dir.glob("**/*.sph"))
    print(f"Found {len(sph_list)} SPH files")
    for sph_path in sph_list:
        wav_path = wavs_dir / (sph_path.stem + ".wav")
        if wav_path.is_file():
            continue
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(sph_path), str(wav_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    transcripts_path = an4_dir / "etc" / f"an4_{split}.transcription"
    manifest_path = target_dir / f"{split}_manifest.json"
    written = 0
    with transcripts_path.open() as fin, manifest_path.open("w") as fout:
        for line in fin:
            transcript = line[: line.find("(") - 1].lower().replace("<s>", "").replace("</s>", "").strip()
            file_id = line[line.find("(") + 1 : -2]
            audio_path = wavs_dir / f"{file_id}.wav"
            if not audio_path.is_file():
                print(f"Skipping missing audio file: {audio_path}")
                continue
            fout.write(
                json.dumps(
                    {
                        "audio_filepath": str(audio_path),
                        "duration": wav_duration_seconds(audio_path),
                        "text": transcript,
                    }
                )
                + "\n"
            )
            written += 1

    print(f"Wrote {written} entries to {manifest_path}")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("an4_data"), help="Where to download/convert data")
    args = parser.parse_args()

    an4_dir = download_and_extract(args.out_dir)
    target_dir = args.out_dir / "an4_converted"

    train_manifest = convert_and_build_manifest(an4_dir, target_dir, "train")
    test_manifest = convert_and_build_manifest(an4_dir, target_dir, "test")

    print(f"\nTrain manifest: {train_manifest}")
    print(f"Test manifest : {test_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
