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

# ── Monkey-patch HiFi-GAN to handle off-by-one mel frames ─────────────────────
# The STFT roundtrip (audio→mel→generator→audio→mel) can produce spectrograms
# that differ by 1 frame, causing F.l1_loss to fail on dimension mismatch.
import torch.nn.functional as F

def _patched_training_step(self, batch, batch_idx):
    audio, audio_len, audio_mel, _ = self._process_batch(batch)

    audio_trg_mel, _ = self.trg_melspec_fn(audio, audio_len)
    audio = audio.unsqueeze(1)

    audio_pred = self.generator(x=audio_mel)

    # Truncate audio_pred to match audio length (generator may produce
    # slightly different length due to STFT framing)
    min_audio_len = min(audio.shape[2], audio_pred.shape[2])
    audio = audio[:, :, :min_audio_len]
    audio_pred = audio_pred[:, :, :min_audio_len]

    audio_pred_mel, _ = self.trg_melspec_fn(audio_pred.squeeze(1), audio_len)

    optim_g, optim_d = self.optimizers()

    # Train discriminator
    optim_d.zero_grad()
    mpd_score_real, mpd_score_gen, _, _ = self.mpd(y=audio, y_hat=audio_pred.detach())
    loss_disc_mpd, _, _ = self.discriminator_loss(
        disc_real_outputs=mpd_score_real, disc_generated_outputs=mpd_score_gen
    )
    msd_score_real, msd_score_gen, _, _ = self.msd(y=audio, y_hat=audio_pred.detach())
    loss_disc_msd, _, _ = self.discriminator_loss(
        disc_real_outputs=msd_score_real, disc_generated_outputs=msd_score_gen
    )
    loss_d = loss_disc_msd + loss_disc_mpd
    self.manual_backward(loss_d)
    optim_d.step()

    # Train generator — truncate mels to min length for L1 loss
    optim_g.zero_grad()
    min_mel_len = min(audio_pred_mel.shape[2], audio_trg_mel.shape[2])
    loss_mel = F.l1_loss(audio_pred_mel[:, :, :min_mel_len], audio_trg_mel[:, :, :min_mel_len])
    _, mpd_score_gen, fmap_mpd_real, fmap_mpd_gen = self.mpd(y=audio, y_hat=audio_pred)
    _, msd_score_gen, fmap_msd_real, fmap_msd_gen = self.msd(y=audio, y_hat=audio_pred)
    loss_fm_mpd = self.feature_loss(fmap_r=fmap_mpd_real, fmap_g=fmap_mpd_gen)
    loss_fm_msd = self.feature_loss(fmap_r=fmap_msd_real, fmap_g=fmap_msd_gen)
    loss_gen_mpd, _ = self.generator_loss(disc_outputs=mpd_score_gen)
    loss_gen_msd, _ = self.generator_loss(disc_outputs=msd_score_gen)
    loss_g = loss_gen_msd + loss_gen_mpd + loss_fm_msd + loss_fm_mpd + loss_mel * self.l1_factor
    self.manual_backward(loss_g)
    optim_g.step()

    self.update_lr()

    metrics = {
        "g_loss_fm_mpd": loss_fm_mpd,
        "g_loss_fm_msd": loss_fm_msd,
        "g_loss_gen_mpd": loss_gen_mpd,
        "g_loss_gen_msd": loss_gen_msd,
        "g_loss": loss_g,
        "d_loss_mpd": loss_disc_mpd,
        "d_loss_msd": loss_disc_msd,
        "d_loss": loss_d,
        "global_step": self.global_step,
        "lr": optim_g.param_groups[0]['lr'],
    }
    self.log_dict(metrics, on_step=True, sync_dist=True)
    self.log("g_l1_loss", loss_mel, prog_bar=True, logger=False, sync_dist=True)

def _patched_validation_step(self, batch, batch_idx):
    audio, audio_len, audio_mel, audio_mel_len = self._process_batch(batch)
    audio_pred = self(spec=audio_mel)
    # Truncate to matching length
    min_audio_len = min(audio.shape[1], audio_pred.shape[2])
    audio_trimmed = audio[:, :min_audio_len]
    audio_pred_trimmed = audio_pred[:, :, :min_audio_len]
    audio_pred_mel, _ = self.audio_to_melspec_precessor(audio_pred_trimmed.squeeze(1), audio_len)
    # Truncate mels to handle off-by-one frame counts
    min_len = min(audio_mel.shape[2], audio_pred_mel.shape[2])
    loss_mel = F.l1_loss(audio_mel[:, :, :min_len], audio_pred_mel[:, :, :min_len])
    self.log_dict({"val_loss": loss_mel}, on_epoch=True, sync_dist=True)

# Apply patches before model instantiation
def _apply_validation_patch():
    from nemo.collections.tts.models import HifiGanModel
    HifiGanModel.training_step = _patched_training_step
    HifiGanModel.validation_step = _patched_validation_step

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
    from nemo.utils.exp_manager import exp_manager

    # Apply validation_step fix before importing HifiGanModel
    _apply_validation_patch()
    from nemo.collections.tts.models import HifiGanModel

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

        # Critical: update ds_class so _process_batch handles VocoderDataset
        # dict batches correctly (pretrained model defaults to MelAudioDataset)
        model.ds_class = cfg.model.train_ds.dataset._target_

        # Assign trainer before setup_training_data (needs trainer.world_size)
        model.set_trainer(trainer)

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
