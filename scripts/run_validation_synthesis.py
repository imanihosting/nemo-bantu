#!/usr/bin/env python3
"""Synthesise the validation prompt set for native-speaker grading.

For each prompt in ``data/validation/<language>.jsonl``, run the full TTS
pipeline (normalize -> G2P -> NeMo synthesis or fallback) and write:

    validation_runs/<run_id>/<language>/<prompt_id>.wav
    validation_runs/<run_id>/<language>/grading.csv

The CSV is the form a native-speaker grader fills in. Columns:
    prompt_id, focus, text, ipa, accuracy_1to5, naturalness_1to5, notes
"""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from frontend.registry import SUPPORTED_LANGUAGES, get_pack
from inference.pipeline import run_tts_pipeline


def _run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", default="data/validation")
    parser.add_argument("--output-root", default="validation_runs")
    parser.add_argument("--languages", nargs="+", default=sorted(SUPPORTED_LANGUAGES))
    parser.add_argument("--voice", default="female_1")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    run_id = _run_id()
    val_root = Path(args.validation_dir)
    out_root = Path(args.output_root) / run_id

    for language in args.languages:
        prompts_path = val_root / f"{language}.jsonl"
        if not prompts_path.exists():
            print(f"skip {language}: {prompts_path} not found")
            continue

        lang_dir = out_root / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, str]] = []

        with prompts_path.open(encoding="utf-8") as fh:
            for line in fh:
                prompt = json.loads(line)
                result, debug = run_tts_pipeline(
                    text=prompt["text"],
                    language=language,
                    voice=args.voice,
                    speed=args.speed,
                )
                wav_path = lang_dir / f"{prompt['id']}.wav"
                wav_path.write_bytes(result.audio_bytes)
                rows.append({
                    "prompt_id": prompt["id"],
                    "focus": prompt.get("focus", ""),
                    "text": prompt["text"],
                    "ipa": debug.get("phonemes", ""),
                    "accuracy_1to5": "",
                    "naturalness_1to5": "",
                    "notes": "",
                })
                # Confirm the language pack actually loaded — catches missing
                # registry registration during refactors.
                _ = get_pack(language)

        csv_path = lang_dir / "grading.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{language}: {len(rows)} prompts -> {lang_dir}")


if __name__ == "__main__":
    main()
