#!/usr/bin/env python3
"""HiFi-GAN Phase 2 training — fine-tune on FastPitch-generated mels.

Phase 1 trained HiFi-GAN on ground-truth mels (audio → mel → reconstruct).
Phase 2 trains on FastPitch-generated mels paired with ground-truth audio,
so the vocoder learns to handle the synthetic mel characteristics.

This produces much more natural-sounding audio at inference time.

Usage:
    # Generate FastPitch mels first:
    python scripts/generate_fp_mels.py

    # Then run Phase 2 training:
    python training/train_hifigan_phase2.py
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ── PyTorch 2.6 weights_only workaround ──────────────────────────────────────
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

torch.set_float32_matmul_precision("high")


# ── Custom Dataset: pairs FastPitch mels with ground-truth audio ─────────────
import torchaudio
from torch.utils.data import Dataset, DataLoader


class FastPitchMelDataset(Dataset):
    """Dataset that loads (FastPitch mel, ground-truth audio) pairs.

    Each item returns:
        audio: [n_samples] — ground-truth audio segment
        audio_len: int
        mel: [n_mel, mel_frames] — FastPitch-generated mel spectrogram
        mel_len: int
    """

    def __init__(self, manifest_path: str, sample_rate: int = 22050,
                 n_samples: int = 8192, min_duration: float = 0.5,
                 hop_length: int = 256):
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.hop_length = hop_length
        self.n_mel_segments = n_samples // hop_length  # 32 frames for 8192 samples
        self.entries = []

        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("duration", 999) >= min_duration:
                    mel_path = entry["mel_filepath"]
                    audio_path = entry["audio_filepath"]
                    if Path(mel_path).exists() and Path(audio_path).exists():
                        self.entries.append(entry)

        print(f"  📂 FastPitchMelDataset: {len(self.entries)} entries loaded")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        audio, sr = torchaudio.load(entry["audio_filepath"])
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        audio = audio[0]  # mono, [T]

        mel = torch.load(entry["mel_filepath"], map_location="cpu")  # [n_mel, M]

        # Reconcile audio length to mel.shape[1] * hop_length so paired
        # crop windows always line up (FastPitch mels may not match the
        # recorded audio's exact frame count).
        expected_audio_len = mel.shape[1] * self.hop_length
        if audio.shape[0] > expected_audio_len:
            audio = audio[:expected_audio_len]
        elif audio.shape[0] < expected_audio_len:
            audio = F.pad(audio, (0, expected_audio_len - audio.shape[0]))

        # Paired random crop: same start in mel and audio domains.
        if mel.shape[1] >= self.n_mel_segments:
            mel_start = random.randint(0, mel.shape[1] - self.n_mel_segments)
            mel = mel[:, mel_start:mel_start + self.n_mel_segments]
            audio_start = mel_start * self.hop_length
            audio = audio[audio_start:audio_start + self.n_samples]
        else:
            mel = F.pad(mel, (0, self.n_mel_segments - mel.shape[1]))
            audio = F.pad(audio, (0, self.n_samples - audio.shape[0]))

        return audio, audio.shape[0], mel, mel.shape[1]


def collate_fn(batch):
    audios, audio_lens, mels, mel_lens = zip(*batch)
    return (
        torch.stack(audios),
        torch.tensor(audio_lens),
        torch.stack(mels),
        torch.tensor(mel_lens),
    )


# ── Monkey-patch HiFi-GAN for Phase 2 ────────────────────────────────────────
# Override training_step to:
# 1. Use FastPitch mels as input to generator (instead of audio → mel)
# 2. Compare generated audio against ground-truth audio
# 3. Handle length mismatches

def _phase2_training_step(self, batch, batch_idx):
    audio, audio_len, fp_mel, mel_len = batch

    # Compute target mel from ground-truth audio
    audio_trg_mel, _ = self.trg_melspec_fn(audio, audio_len)
    audio = audio.unsqueeze(1)

    # Generate audio from FastPitch mels
    audio_pred = self.generator(x=fp_mel)

    # Truncate to matching length
    min_audio_len = min(audio.shape[2], audio_pred.shape[2])
    audio = audio[:, :, :min_audio_len]
    audio_pred = audio_pred[:, :, :min_audio_len]

    # Compute mel of generated audio
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

    # Train generator
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
        "d_loss": loss_d,
        "global_step": self.global_step,
        "lr": optim_g.param_groups[0]['lr'],
    }
    self.log_dict(metrics, on_step=True, sync_dist=True)
    self.log("g_l1_loss", loss_mel, prog_bar=True, logger=False, sync_dist=True)


def _phase2_validation_step(self, batch, batch_idx):
    audio, audio_len, fp_mel, mel_len = batch
    audio_pred = self(spec=fp_mel)
    # Compute mels for both
    audio_mel, _ = self.audio_to_melspec_precessor(audio, audio_len)
    min_audio_len = min(audio.shape[1], audio_pred.shape[2])
    audio_pred_trimmed = audio_pred[:, :, :min_audio_len]
    audio_pred_mel, _ = self.audio_to_melspec_precessor(audio_pred_trimmed.squeeze(1), audio_len)
    min_mel = min(audio_mel.shape[2], audio_pred_mel.shape[2])
    loss = F.l1_loss(audio_mel[:, :, :min_mel], audio_pred_mel[:, :, :min_mel])
    self.log_dict({"val_loss": loss}, on_epoch=True, sync_dist=True)


def _apply_phase2_patches():
    from nemo.collections.tts.models import HifiGanModel
    HifiGanModel.training_step = _phase2_training_step
    HifiGanModel.validation_step = _phase2_validation_step


# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
HIFIGAN_CONFIG = PROJECT_DIR / "configs" / "training" / "hifigan_shona.yaml"
MEL_MANIFEST = PROJECT_DIR / "data" / "manifests" / "shona_fp_mels_manifest.jsonl"
PHASE1_CKPT_DIR = PROJECT_DIR / "outputs" / "hifigan_shona" / "HiFiGAN_Shona" / "checkpoints"

import re

def find_best_phase1_ckpt():
    pattern = re.compile(r"HiFiGAN_Shona--val_loss=([\d.]+)-epoch=(\d+)\.ckpt$")
    best_path, best_loss = None, float("inf")
    for f in PHASE1_CKPT_DIR.glob("HiFiGAN_Shona--*.ckpt"):
        if "-last" in f.name:
            continue
        m = pattern.search(f.name)
        if m:
            loss = float(m.group(1))
            if loss < best_loss:
                best_loss = loss
                best_path = f
    return best_path


def create_train_val_split(val_ratio=0.1, seed=42):
    """Split the mel manifest into train/val."""
    train_path = PROJECT_DIR / "data" / "manifests" / "shona_fp_mels_train.jsonl"
    val_path = PROJECT_DIR / "data" / "manifests" / "shona_fp_mels_val.jsonl"

    if train_path.exists() and val_path.exists():
        train_count = sum(1 for _ in open(train_path))
        val_count = sum(1 for _ in open(val_path))
        print(f"  ✅ Using existing split: {train_count} train, {val_count} val")
        return str(train_path), str(val_path)

    with open(MEL_MANIFEST) as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    val_size = max(1, int(len(lines) * val_ratio))
    val_lines = lines[:val_size]
    train_lines = lines[val_size:]

    with open(train_path, "w") as f:
        f.writelines(train_lines)
    with open(val_path, "w") as f:
        f.writelines(val_lines)

    print(f"  ✅ Split created: {len(train_lines)} train, {len(val_lines)} val")
    return str(train_path), str(val_path)


def main():
    parser = argparse.ArgumentParser(description="HiFi-GAN Phase 2 training")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of Phase 2 epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  🎵 HiFi-GAN Phase 2 — FastPitch Mel Fine-tuning")
    print(f"{'='*60}\n")

    if not MEL_MANIFEST.exists():
        print("❌ FastPitch mel manifest not found!")
        print("   Run first: python scripts/generate_fp_mels.py")
        sys.exit(1)

    # ── Step 1: Split data ────────────────────────────────────────────────
    train_manifest, val_manifest = create_train_val_split()

    # ── Step 2: Create datasets ───────────────────────────────────────────
    print("\n⏳ Setting up datasets...")
    train_ds = FastPitchMelDataset(train_manifest, n_samples=8192, min_duration=0.5, hop_length=256)
    val_ds = FastPitchMelDataset(val_manifest, n_samples=8192, min_duration=0.5, hop_length=256)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
        collate_fn=collate_fn, persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
        collate_fn=collate_fn, persistent_workers=True,
    )

    # ── Step 3: Load Phase 1 model ────────────────────────────────────────
    print("\n⏳ Loading Phase 1 HiFi-GAN checkpoint...")
    _apply_phase2_patches()
    from nemo.collections.tts.models import HifiGanModel
    from omegaconf import OmegaConf
    import lightning.pytorch as pl
    from nemo.utils.exp_manager import exp_manager

    phase1_ckpt = find_best_phase1_ckpt()
    if phase1_ckpt is None:
        print("❌ No Phase 1 HiFi-GAN checkpoint found")
        sys.exit(1)

    print(f"  📁 Loading from {phase1_ckpt.name}")

    # Load pretrained model structure, then load Phase 1 weights
    model = HifiGanModel.from_pretrained(model_name="tts_en_hifigan")
    ckpt_data = torch.load(str(phase1_ckpt), map_location="cpu")
    state_dict = ckpt_data.get("state_dict", ckpt_data)
    model.load_state_dict(state_dict, strict=False)
    print(f"  ✅ Phase 1 weights loaded")

    # ── Step 4: Setup trainer ─────────────────────────────────────────────
    cfg = OmegaConf.load(str(HIFIGAN_CONFIG))

    trainer = pl.Trainer(
        num_nodes=1,
        devices=1,
        accelerator="gpu",
        strategy="auto",
        precision="32-true",
        max_epochs=args.epochs,
        accumulate_grad_batches=1,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=100,
        check_val_every_n_epoch=10,
        benchmark=True,
    )

    # ── Step 5: Setup experiment manager ──────────────────────────────────
    exp_cfg = OmegaConf.create({
        "exp_dir": "outputs/hifigan_shona_phase2",
        "name": "HiFiGAN_Shona_Phase2",
        "create_tensorboard_logger": True,
        "create_checkpoint_callback": True,
        "checkpoint_callback_params": {
            "monitor": "val_loss",
            "save_top_k": 3,
            "mode": "min",
        },
        "resume_if_exists": True,
        "resume_ignore_no_checkpoint": True,
    })
    log_dir = exp_manager(trainer, exp_cfg)
    print(f"  📁 Experiment dir: {log_dir}")

    # ── Step 6: Assign custom dataloaders ─────────────────────────────────
    model.set_trainer(trainer)
    # Override the dataloaders directly
    model._train_dl = train_dl
    model._validation_dl = val_dl

    # ── Step 7: Train ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🚀 Starting Phase 2 training...")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples: {len(val_ds)}")
    print(f"{'='*60}\n")

    trainer.fit(model)

    print(f"\n{'='*60}")
    print(f"  ✅ Phase 2 training complete!")
    print(f"  📁 Checkpoints: {log_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
