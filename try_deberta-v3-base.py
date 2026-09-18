import os
import shutil
from pathlib import Path

import certifi
import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "microsoft/deberta-v3-base"
EXAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."


os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


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

    print("Warning: ffmpeg.exe not found")


def main():
    ensure_ffmpeg_on_path()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModel.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    print(f"Using {device.upper()}")

    print("\nInput text:")
    print(EXAMPLE_TEXT)

    inputs = tokenizer(
        EXAMPLE_TEXT,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    print("\nModel loaded successfully.")
    print(f"Last hidden state shape: {last_hidden_state.shape}")

    cls_embedding = last_hidden_state[:, 0, :]

    print(f"CLS embedding shape: {cls_embedding.shape}")

    print("\nFirst 10 embedding values:")
    print(cls_embedding[0][:10].cpu().numpy())


if __name__ == "__main__":
    main()