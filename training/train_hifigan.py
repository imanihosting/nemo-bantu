"""Train HiFi-GAN vocoder using NeMo + PyTorch Lightning.

Usage:
    python training/train_hifigan.py                            # defaults
    python training/train_hifigan.py trainer.max_steps=100000   # override
"""
from pathlib import Path

import lightning.pytorch as pl

from nemo.collections.tts.models import HifiGanModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(
    config_path=str(Path(__file__).resolve().parent.parent / "configs" / "training"),
    config_name="hifigan_shona",
)
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(model)


if __name__ == "__main__":
    main()
