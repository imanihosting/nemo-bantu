"""Train FastPitch model using NeMo + PyTorch Lightning.

Usage:
    python training/train_fastpitch.py                           # defaults
    python training/train_fastpitch.py trainer.max_epochs=500    # override via CLI
    python training/train_fastpitch.py --config-name=fastpitch_shona  # choose config
"""
import os
import signal
import sys
import logging
from pathlib import Path

import torch
import lightning.pytorch as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

log = logging.getLogger(__name__)

# ── Performance: use tensor cores on GB10 properly ────────────────────────────
torch.set_float32_matmul_precision("high")

# ── Graceful shutdown on SIGTERM (e.g. from watchdog or system) ──────────────
def _handle_signal(signum, frame):
    log.warning(f"Received signal {signum}. Initiating graceful shutdown...")
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_signal)


@hydra_runner(
    config_path=str(Path(__file__).resolve().parent.parent / "configs" / "training"),
    config_name="fastpitch_shona",
)
def main(cfg):
    # ── Resource diagnostics at startup ──────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Python     : {sys.version.split()[0]}")
    log.info(f"PyTorch    : {torch.__version__}")
    log.info(f"CUDA avail : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU        : {torch.cuda.get_device_name(0)}")
    log.info(f"CPU cores  : {os.cpu_count()}")

    import psutil
    mem = psutil.virtual_memory()
    log.info(f"RAM total  : {mem.total // (1024**3)} GB")
    log.info(f"RAM avail  : {mem.available // (1024**3)} GB")
    log.info("=" * 60)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = FastPitchModel(cfg=cfg.model, trainer=trainer)

    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])

    trainer.fit(model)


if __name__ == "__main__":
    main()
