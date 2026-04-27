#!/usr/bin/env python3
"""Generate FastPitch mel spectrograms for HiFi-GAN Phase 2 training.

Phase 1 HiFi-GAN was trained on ground-truth mels (audio → mel → reconstruct).
Phase 2 trains on FastPitch-generated mels so the vocoder learns to handle
the imperfections in synthesized spectrograms.

This script:
1. Loads FastPitch from the best checkpoint
2. Runs inference on every utterance in the training manifest
3. Saves the generated mel spectrograms as .pt files
4. Creates a new manifest pointing to (mel_file, audio_file) pairs

Usage:
    python scripts/generate_fp_mels.py
"""
import json
import os
import re
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# ── PyTorch 2.6 weights_only workaround ──────────────────────────────────────
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

PROJECT_DIR = Path(__file__).resolve().parent.parent
FASTPITCH_CONFIG = PROJECT_DIR / "configs" / "training" / "fastpitch_shona.yaml"
CKPT_DIR = PROJECT_DIR / "outputs" / "fastpitch_shona" / "FastPitch_Shona" / "checkpoints"
MANIFEST_PATH = PROJECT_DIR / "data" / "manifests" / "shona_train_manifest.jsonl"
MEL_OUTPUT_DIR = PROJECT_DIR / "data" / "fastpitch_mels"


def find_best_checkpoint():
    pattern = re.compile(r"FastPitch_Shona--val_loss=([\d.]+)-epoch=(\d+)\.ckpt$")
    best_path, best_loss = None, float("inf")
    for f in CKPT_DIR.glob("FastPitch_Shona--*.ckpt"):
        if "-last" in f.name:
            continue
        m = pattern.search(f.name)
        if m:
            loss = float(m.group(1))
            if loss < best_loss:
                best_loss = loss
                best_path = f
    return best_path


def main():
    print(f"\n{'='*60}")
    print(f"  🎤 FastPitch Mel Generation for HiFi-GAN Phase 2")
    print(f"{'='*60}\n")

    # ── Load FastPitch ────────────────────────────────────────────────────
    ckpt_path = find_best_checkpoint()
    if ckpt_path is None:
        print("❌ No FastPitch checkpoint found")
        sys.exit(1)

    print(f"⏳ Loading FastPitch from {ckpt_path.name}...")
    from omegaconf import OmegaConf
    from nemo.collections.tts.models import FastPitchModel

    cfg = OmegaConf.load(str(FASTPITCH_CONFIG))
    model = FastPitchModel(cfg=cfg.model)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  ✅ FastPitch loaded on {device}")

    # ── Load manifest ─────────────────────────────────────────────────────
    with open(MANIFEST_PATH) as f:
        entries = [json.loads(line) for line in f]
    print(f"  📂 {len(entries)} utterances to process")

    # ── Generate mels ─────────────────────────────────────────────────────
    MEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_manifest = PROJECT_DIR / "data" / "manifests" / "shona_fp_mels_manifest.jsonl"

    success = 0
    skipped = 0

    with open(output_manifest, "w") as mf:
        for entry in tqdm(entries, desc="Generating mels"):
            audio_path = entry["audio_filepath"]
            text = entry.get("text", "")

            if not text.strip():
                skipped += 1
                continue

            # Create unique filename from audio path
            audio_basename = Path(audio_path).stem
            mel_path = MEL_OUTPUT_DIR / f"{audio_basename}.pt"

            # Skip if already generated
            if mel_path.exists():
                record = {
                    "audio_filepath": audio_path,
                    "mel_filepath": str(mel_path),
                    "duration": entry.get("duration", 0),
                    "text": text,
                }
                mf.write(json.dumps(record) + "\n")
                success += 1
                continue

            try:
                with torch.no_grad():
                    tokens = model.parse(text)
                    tokens = tokens.to(device)
                    spectrogram = model.generate_spectrogram(tokens=tokens)
                    # Save as [n_mel, time] on CPU
                    mel = spectrogram.squeeze(0).cpu()
                    torch.save(mel, str(mel_path))

                record = {
                    "audio_filepath": audio_path,
                    "mel_filepath": str(mel_path),
                    "duration": entry.get("duration", 0),
                    "text": text,
                }
                mf.write(json.dumps(record) + "\n")
                success += 1

            except Exception as e:
                skipped += 1
                continue

    print(f"\n{'='*60}")
    print(f"  ✅ Done! {success} mels generated, {skipped} skipped")
    print(f"  📁 Mels: {MEL_OUTPUT_DIR}")
    print(f"  📄 Manifest: {output_manifest}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
