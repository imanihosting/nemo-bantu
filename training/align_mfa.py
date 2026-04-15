import argparse
import subprocess
from pathlib import Path


def run_mfa_alignment(corpus_dir: Path, dictionary_path: Path, acoustic_model: str, output_dir: Path) -> None:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Missing corpus dir: {corpus_dir}")
    if not dictionary_path.exists():
        raise FileNotFoundError(f"Missing dictionary file: {dictionary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "mfa",
        "align",
        "--clean",
        str(corpus_dir),
        str(dictionary_path),
        acoustic_model,
        str(output_dir),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MFA alignment.")
    parser.add_argument("--corpus-dir", default="data/raw/shona")
    parser.add_argument("--dictionary", default="frontend/lexicons/shona.txt")
    parser.add_argument("--acoustic-model", default="english_mfa")
    parser.add_argument("--output-dir", default="data/processed/aligned/shona")
    args = parser.parse_args()

    run_mfa_alignment(
        corpus_dir=Path(args.corpus_dir),
        dictionary_path=Path(args.dictionary),
        acoustic_model=args.acoustic_model,
        output_dir=Path(args.output_dir),
    )
