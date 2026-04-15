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
    run_mfa_alignment(
        corpus_dir=Path("data/raw"),
        dictionary_path=Path("frontend/lexicons/shona.txt"),
        acoustic_model="english_mfa",
        output_dir=Path("data/processed/aligned"),
    )
