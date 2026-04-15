from pathlib import Path
import subprocess


def train_hifigan(config_path: Path) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    command = [
        "python",
        "-m",
        "nemo.collections.tts.models.hifigan",
        f"--config-path={config_path.parent}",
        f"--config-name={config_path.stem}",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    train_hifigan(config_path=Path("configs/training/hifigan_train.yaml"))
