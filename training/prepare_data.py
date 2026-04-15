from pathlib import Path
import argparse
import json

import soundfile as sf


def _audio_duration_seconds(audio_path: Path) -> float:
    info = sf.info(str(audio_path))
    if info.samplerate <= 0:
        raise ValueError(f"Invalid sample rate in {audio_path}")
    return round(info.frames / info.samplerate, 4)


def create_manifest(input_dir: Path, manifest_path: Path, min_duration: float = 0.2) -> int:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with manifest_path.open("w", encoding="utf-8") as out:
        for wav_file in sorted(input_dir.glob("*.wav")):
            txt_file = wav_file.with_suffix(".txt")
            if not txt_file.exists():
                continue

            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                continue

            duration = _audio_duration_seconds(wav_file)
            if duration < min_duration:
                continue

            item = {
                "audio_filepath": str(wav_file.resolve()),
                "text": text,
                "duration": duration,
            }
            out.write(json.dumps(item, ensure_ascii=True) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create NeMo manifest from wav/txt pairs.")
    parser.add_argument("--input-dir", default="data/raw", help="Directory with .wav and .txt pairs")
    parser.add_argument("--manifest", default="data/manifests/train_manifest.jsonl", help="Output manifest path")
    parser.add_argument("--min-duration", type=float, default=0.2, help="Drop clips shorter than threshold")
    args = parser.parse_args()

    created = create_manifest(
        input_dir=Path(args.input_dir),
        manifest_path=Path(args.manifest),
        min_duration=args.min_duration,
    )
    print(f"Manifest generated: {args.manifest} | entries={created}")
