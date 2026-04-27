#!/usr/bin/env python3
"""Diagnose where the robotic sound comes from.

Test 1: Ground-truth audio → mel → HiFi-GAN → audio (tests vocoder quality)
Test 2: Text → FastPitch mel → HiFi-GAN → audio (tests full pipeline)

If Test 1 sounds good but Test 2 sounds robotic → FastPitch is the problem.
If Test 1 also sounds robotic → HiFi-GAN is the problem.
"""
import sys, re, wave
from pathlib import Path
import numpy as np
import torch
import torchaudio

_orig = torch.load
def _ul(*a,**kw): kw["weights_only"]=False; return _orig(*a,**kw)
torch.load = _ul

PROJECT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT / "outputs" / "hifigan_shona" / "HiFiGAN_Shona" / "checkpoints"
FP_CKPT_DIR = PROJECT / "outputs" / "fastpitch_shona" / "FastPitch_Shona" / "checkpoints"
OUT_DIR = PROJECT / "outputs" / "test_audio" / "diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_best(d, prefix):
    pat = re.compile(rf"{prefix}--val_loss=([\d.]+)-epoch=(\d+)\.ckpt$")
    best, bl = None, float("inf")
    for f in d.glob(f"{prefix}--*.ckpt"):
        if "-last" in f.name: continue
        m = pat.search(f.name)
        if m and float(m.group(1)) < bl:
            bl = float(m.group(1)); best = f
    return best

def save_wav(audio_np, path, sr=22050):
    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm = (audio_np * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())

print("⏳ Loading models...")
from omegaconf import OmegaConf
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load FastPitch
fp_ckpt = find_best(FP_CKPT_DIR, "FastPitch_Shona")
cfg = OmegaConf.load(str(PROJECT / "configs/training/fastpitch_shona.yaml"))
fp = FastPitchModel(cfg=cfg.model)
sd = torch.load(str(fp_ckpt), map_location="cpu").get("state_dict", {})
fp.load_state_dict(sd, strict=False)
fp.eval().to(device)
print(f"  ✅ FastPitch: {fp_ckpt.name}")

# Load fine-tuned HiFi-GAN
hg_ckpt = find_best(CKPT_DIR, "HiFiGAN_Shona")
hg = HifiGanModel.from_pretrained("tts_en_hifigan")
sd2 = torch.load(str(hg_ckpt), map_location="cpu").get("state_dict", {})
hg.load_state_dict(sd2, strict=False)
hg.eval().to(device)
print(f"  ✅ HiFi-GAN (Shona): {hg_ckpt.name}")

# Also load pretrained English HiFi-GAN for comparison
hg_en = HifiGanModel.from_pretrained("tts_en_hifigan")
hg_en.eval().to(device)
print(f"  ✅ HiFi-GAN (English pretrained)")

# Pick a few ground-truth audio samples
import json
manifest = PROJECT / "data/manifests/shona_train_manifest.jsonl"
with open(manifest) as f:
    entries = [json.loads(l) for l in f]
# Pick 3 medium-length samples
samples = [e for e in entries if 3.0 < e.get("duration", 0) < 6.0][:3]

print(f"\n{'='*60}")
print("  TEST 1: Ground-truth audio → mel → HiFi-GAN → audio")
print(f"{'='*60}")

for i, entry in enumerate(samples):
    audio, sr = torchaudio.load(entry["audio_filepath"])
    if sr != 22050:
        audio = torchaudio.functional.resample(audio, sr, 22050)
    audio = audio[0].to(device)

    # Save original
    save_wav(audio.cpu().numpy(), OUT_DIR / f"gt_{i+1}_original.wav")

    with torch.no_grad():
        # Ground-truth mel → Shona HiFi-GAN
        mel, _ = hg.audio_to_melspec_precessor(audio.unsqueeze(0), torch.tensor([audio.shape[0]]).to(device))
        recon = hg.convert_spectrogram_to_audio(spec=mel)
        save_wav(recon.squeeze().cpu().numpy(), OUT_DIR / f"gt_{i+1}_shona_hifigan.wav")

        # Ground-truth mel → English HiFi-GAN
        recon_en = hg_en.convert_spectrogram_to_audio(spec=mel)
        save_wav(recon_en.squeeze().cpu().numpy(), OUT_DIR / f"gt_{i+1}_english_hifigan.wav")

    print(f"  [{i+1}] {Path(entry['audio_filepath']).name} ({entry['duration']:.1f}s)")

print(f"\n{'='*60}")
print("  TEST 2: Text → FastPitch → HiFi-GAN → audio")
print(f"{'='*60}")

test_sentences = [
    "Mhoro, makadii?",
    "Zuva rakanaka nhasi.",
    "Zimbabwe inyika yakanaka chaizvo.",
]

for i, text in enumerate(test_sentences):
    with torch.no_grad():
        tokens = fp.parse(text).to(device)
        spec = fp.generate_spectrogram(tokens=tokens)

        # FastPitch mel → Shona HiFi-GAN
        audio_shona = hg.convert_spectrogram_to_audio(spec=spec)
        save_wav(audio_shona.squeeze().cpu().numpy(), OUT_DIR / f"fp_{i+1}_shona_hifigan.wav")

        # FastPitch mel → English HiFi-GAN
        audio_en = hg_en.convert_spectrogram_to_audio(spec=spec)
        save_wav(audio_en.squeeze().cpu().numpy(), OUT_DIR / f"fp_{i+1}_english_hifigan.wav")

    print(f"  [{i+1}] \"{text}\"")

print(f"\n{'='*60}")
print(f"  ✅ Done! Files saved to: {OUT_DIR}")
print(f"")
print(f"  Compare:")
print(f"  gt_*_original.wav      → ground-truth (reference)")
print(f"  gt_*_shona_hifigan.wav → GT mel → Shona HiFi-GAN (tests vocoder)")
print(f"  gt_*_english_hifigan.wav → GT mel → English HiFi-GAN (baseline)")
print(f"  fp_*_shona_hifigan.wav → FastPitch → Shona HiFi-GAN (full pipeline)")
print(f"  fp_*_english_hifigan.wav → FastPitch → English HiFi-GAN (pipeline baseline)")
print(f"{'='*60}")
