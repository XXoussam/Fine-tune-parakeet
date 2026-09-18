import shutil
import os
from pathlib import Path

import torch
from transformers import pipeline
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
AUDIO_FILE = r"C:\Users\osaoudi\Desktop\EASPORTS\VoiceComAnalysis\FineTune\audio\2086-149220-0033.wav"


def ensure_ffmpeg_on_path():
    if shutil.which("ffmpeg"):
        return

    candidates = [
        r"C:\Program Files\Shotcut\ffmpeg.exe",
        r"C:\Program Files\OpenModelica1.22.2-64bit\tools\msys\mingw64\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\Cadence\Fidelity 2025.1\bin\ffmpeg.exe",
    ]

    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            os.environ["PATH"] = (
                str(candidate_path.parent)
                + os.pathsep
                + os.environ.get("PATH", "")
            )
            return

    raise FileNotFoundError("ffmpeg.exe not found")


def main():
    ensure_ffmpeg_on_path()

    device = 0 if torch.cuda.is_available() else -1

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using {'GPU' if device == 0 else 'CPU'}")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
        chunk_length_s=30,
        return_timestamps=False,
    )

    print(f"\nTranscribing:\n{AUDIO_FILE}\n")

    result = pipe(AUDIO_FILE)

    if isinstance(result, dict):
        transcript = result.get("text", "")
    else:
        transcript = str(result)

    print("=== TRANSCRIPT ===")
    print(transcript.strip())
    print("==================")


if __name__ == "__main__":
    main()