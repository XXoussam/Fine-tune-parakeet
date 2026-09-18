import json
import soundfile as sf
from pathlib import Path

SOURCE_JSON = r"C:\Users\osaoudi\Desktop\EASPORTS\VoiceComAnalysis\data-preparation(STT)\dataset.json"

OUTPUT_MANIFEST = "train_manifest.json"

with open(SOURCE_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

count = 0

with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as out:
    for item in data:

        audio_path = item["clip_path"]
        text = item["human_labeled"].strip()

        try:
            info = sf.info(audio_path)

            entry = {
                "audio_filepath": audio_path,
                "duration": float(info.duration),
                "text": text,
            }

            out.write(
                json.dumps(entry, ensure_ascii=False)
                + "\n"
            )

            count += 1

        except Exception as e:
            print(f"Skipped: {audio_path}")
            print(e)

print(f"Created manifest with {count} samples")