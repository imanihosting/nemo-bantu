from pathlib import Path
import argparse
import json


def validate_manifest(manifest_path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    if not manifest_path.exists():
        return 0, [f"Manifest not found: {manifest_path}"]

    for idx, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        count += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"Line {idx}: invalid JSON")
            continue

        for key in ("audio_filepath", "text", "duration"):
            if key not in item:
                errors.append(f'Line {idx}: missing key "{key}"')

        audio_path = Path(item.get("audio_filepath", ""))
        if not audio_path.exists():
            errors.append(f"Line {idx}: audio file not found {audio_path}")

        if not str(item.get("text", "")).strip():
            errors.append(f"Line {idx}: empty text")

        try:
            duration = float(item.get("duration", 0.0))
            if duration <= 0:
                errors.append(f"Line {idx}: non-positive duration")
        except (TypeError, ValueError):
            errors.append(f"Line {idx}: invalid duration")

    return count, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate NeMo manifest JSONL file.")
    parser.add_argument("--manifest", default="data/manifests/train_manifest.jsonl")
    args = parser.parse_args()

    total, errs = validate_manifest(Path(args.manifest))
    if errs:
        print(f"Validation failed. entries={total}")
        for err in errs[:50]:
            print(f"- {err}")
        raise SystemExit(1)

    print(f"Manifest valid. entries={total}")
