#!/usr/bin/env python3
"""Train (fine-tune) HiFi-GAN vocoder on Shona audio data.

Fine-tunes the pretrained English HiFi-GAN (tts_en_hifigan) on Shona audio
so the vocoder learns to reconstruct this specific speaker's voice.

Usage:
    # Fine-tune from pretrained English HiFi-GAN (default):
    python training/train_hifigan.py

    # Resume training from checkpoint:
    python training/train_hifigan.py  # (auto-resumes via exp_manager)

    # Train from scratch (not recommended):
    python training/train_hifigan.py --no-pretrained
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

# ── PyTorch 2.6 weights_only workaround ──────────────────────────────────────
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

torch.set_float32_matmul_precision("high")

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "configs" / "training" / "hifigan_shona.yaml"
MANIFEST_PATH = PROJECT_DIR / "data" / "manifests" / "shona_train_manifest.jsonl"
TRAIN_MANIFEST = PROJECT_DIR / "data" / "manifests" / "shona_hifigan_train.jsonl"
VAL_MANIFEST = PROJECT_DIR / "data" / "manifests" / "shona_hifigan_val.jsonl"


def create_train_val_split(val_ratio: float = 0.1, seed: int = 42):
    """Split the main manifest into train/val for HiFi-GAN training.

    Only creates the split if the files don't already exist.
    """
    if TRAIN_MANIFEST.exists() and VAL_MANIFEST.exists():
        train_count = sum(1 for _ in open(TRAIN_MANIFEST))
        val_count = sum(1 for _ in open(VAL_MANIFEST))
        print(f"  ✅ Using existing split: {train_count} train, {val_count} val")
        return

    print(f"  📂 Creating train/val split from {MANIFEST_PATH.name}...")

    with open(MANIFEST_PATH) as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    val_size = max(1, int(len(lines) * val_ratio))
    val_lines = lines[:val_size]
    train_lines = lines[val_size:]

    with open(TRAIN_MANIFEST, "w") as f:
        f.writelines(train_lines)

    with open(VAL_MANIFEST, "w") as f:
        f.writelines(val_lines)

    print(f"  ✅ Split created: {len(train_lines)} train, {len(val_lines)} val")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune HiFi-GAN on Shona audio")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from scratch instead of fine-tuning")
    parser.add_argument("--pretrained-name", type=str, default="tts_en_hifigan",
                        help="Pretrained model name to fine-tune from (default: tts_en_hifigan)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  🎵 HiFi-GAN Shona Vocoder Training")
    print(f"  Config  : {CONFIG_PATH.name}")
    print(f"  Pretrain: {'None (from scratch)' if args.no_pretrained else args.pretrained_name}")
    print(f"{'='*60}\n")

    # ── Step 1: Create train/val split ────────────────────────────────────────
    create_train_val_split()

    # ── Step 2: Load config ──────────────────────────────────────────────────
    print("\n⏳ Loading config and model...")
    from omegaconf import OmegaConf
    import lightning.pytorch as pl
    from nemo.collections.tts.models import HifiGanModel
    from nemo.utils.exp_manager import exp_manager

    cfg = OmegaConf.load(str(CONFIG_PATH))

    # ── Step 3: Setup trainer ────────────────────────────────────────────────
    trainer = pl.Trainer(**cfg.trainer)

    # ── Step 4: Setup experiment manager ─────────────────────────────────────
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    print(f"  📁 Experiment dir: {log_dir}")

    # ── Step 5: Load model ───────────────────────────────────────────────────
    if args.no_pretrained:
        # Train from scratch
        print("  🔨 Initializing HiFi-GAN from scratch...")
        model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    else:
        # Fine-tune from pretrained
        print(f"  ⏳ Loading pretrained {args.pretrained_name}...")
        model = HifiGanModel.from_pretrained(model_name=args.pretrained_name)
        print(f"  ✅ Pretrained model loaded")

        # Override config with our Shona config for dataset setup
        # Keep generator weights but update dataset and training params
        model._cfg = cfg.model
        model.setup_training_data(cfg.model.train_ds)
        model.setup_validation_data(cfg.model.validation_ds)
        print(f"  ✅ Dataset config updated for Shona")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  🖥️  Device: {device}")

    # ── Step 6: Train ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🚀 Starting training...")
    print(f"  Max epochs: {cfg.trainer.max_epochs}")
    print(f"  Batch size: {cfg.model.train_ds.dataloader_params.batch_size}")
    print(f"{'='*60}\n")

    trainer.fit(model)

    print(f"\n{'='*60}")
    print(f"  ✅ Training complete!")
    print(f"  📁 Checkpoints: {log_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
