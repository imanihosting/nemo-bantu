from pathlib import Path
import argparse
import re

from datasets import Audio, load_dataset
import soundfile as sf


def _slug(text: str, idx: int) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    if not cleaned:
        cleaned = f"sample_{idx}"
    return cleaned[:72]


def _extract_text(example: dict) -> str:
    for key in ("text", "sentence", "transcript", "transcription", "normalized_text"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def export_shona_dataset(dataset_name: str, split: str, output_dir: Path, sample_rate: int, limit: int | None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_name, split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))

    written = 0
    for i, row in enumerate(ds):
        if limit is not None and written >= limit:
            break

        text = _extract_text(row)
        if not text:
            continue

        audio = row.get("audio")
        if not isinstance(audio, dict):
            continue
        array = audio.get("array")
        sr = int(audio.get("sampling_rate", sample_rate))
        if array is None:
            continue

        stem = _slug(text=text, idx=i)
        wav_path = output_dir / f"{stem}.wav"
        txt_path = output_dir / f"{stem}.txt"

        sf.write(str(wav_path), array, sr)
        txt_path.write_text(text + "\n", encoding="utf-8")
        written += 1

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and export Hugging Face Shona dataset into wav/txt pairs.")
    parser.add_argument("--dataset", default="badrex/shona-speech")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/raw/shona")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples to export")
    args = parser.parse_args()

    total = export_shona_dataset(
        dataset_name=args.dataset,
        split=args.split,
        output_dir=Path(args.output_dir),
        sample_rate=args.sample_rate,
        limit=args.limit,
    )
    print(f"Export complete: {total} samples -> {args.output_dir}")
