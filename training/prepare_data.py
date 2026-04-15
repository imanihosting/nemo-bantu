from pathlib import Path
import json


def create_manifest(input_dir: Path, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as out:
        for wav_file in sorted(input_dir.glob("*.wav")):
            txt_file = wav_file.with_suffix(".txt")
            if not txt_file.exists():
                continue
            text = txt_file.read_text(encoding="utf-8").strip()
            item = {
                "audio_filepath": str(wav_file.resolve()),
                "text": text,
                "duration": 0.0,
            }
            out.write(json.dumps(item, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    manifest = Path("data/manifests/train_manifest.jsonl")
    create_manifest(input_dir=raw_dir, manifest_path=manifest)
    print(f"Manifest generated: {manifest}")
