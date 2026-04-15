"""Train FastPitch model using NeMo + PyTorch Lightning.

Usage:
    python training/train_fastpitch.py                           # defaults
    python training/train_fastpitch.py trainer.max_epochs=500    # override via CLI
    python training/train_fastpitch.py --config-name=fastpitch_shona  # choose config
"""
from pathlib import Path
import sys

import lightning.pytorch as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(
    config_path=str(Path(__file__).resolve().parent.parent / "configs" / "training"),
    config_name="fastpitch_shona",
)
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = FastPitchModel(cfg=cfg.model, trainer=trainer)

    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])

    trainer.fit(model)


if __name__ == "__main__":
    main()
