#!/usr/bin/env python3
"""Test audio quality from a FastPitch checkpoint.

Loads the best (or specified) FastPitch .ckpt, pairs it with a pretrained
HiFi-GAN vocoder, synthesizes Shona test sentences, and saves .wav files.

Usage:
    # Use best checkpoint automatically:
    python scripts/test_audio.py

    # Specify a checkpoint:
    python scripts/test_audio.py --checkpoint outputs/fastpitch_shona/FastPitch_Shona/checkpoints/FastPitch_Shona--val_loss=2.5572-epoch=14.ckpt

    # Also try Griffin-Lim (no vocoder download needed):
    python scripts/test_audio.py --griffin-lim
"""
import argparse
import os
import re
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ── PyTorch 2.6 weights_only workaround (same as training) ────────────────────
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

torch.set_float32_matmul_precision("high")

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_DIR / "outputs" / "fastpitch_shona" / "FastPitch_Shona" / "checkpoints"
CONFIG_PATH = PROJECT_DIR / "configs" / "training" / "fastpitch_shona.yaml"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "test_audio"

# ── Shona test sentences ─────────────────────────────────────────────────────
# A mix of short, medium, and long utterances to test different aspects
TEST_SENTENCES = [
    "Mhoro, makadii?",                                   # Hello, how are you?
    "Ndiri kunzwa tsitsi.",                               # I am feeling well.
    "Zuva rakanaka nhasi.",                               # It's a beautiful day today.
    "Ndinokuda zvikuru.",                                 # I love you very much.
    "Tinotenda Mwari wedu.",                              # We thank our God.
    "Mwana akadzidza kunyora tsamba.",                    # The child learned to write a letter.
    "Mvura yanaya zvakanyanya manheru ano.",              # It rained heavily this evening.
    "Zimbabwe inyika yakanaka chaizvo.",                  # Zimbabwe is a very beautiful country.
]


def find_best_checkpoint() -> Path:
    """Find the checkpoint with the lowest val_loss in the checkpoint directory."""
    if not CKPT_DIR.exists():
        print(f"❌ Checkpoint directory not found: {CKPT_DIR}")
        sys.exit(1)

    ckpts = list(CKPT_DIR.glob("*.ckpt"))
    # Exclude '-last.ckpt' duplicates
    ckpts = [c for c in ckpts if not c.name.endswith("-last.ckpt")]

    if not ckpts:
        print(f"❌ No checkpoints found in {CKPT_DIR}")
        sys.exit(1)

    # Parse val_loss from filename: FastPitch_Shona--val_loss=2.5572-epoch=14.ckpt
    def extract_val_loss(path: Path) -> float:
        match = re.search(r"val_loss=([\d.]+)", path.name)
        return float(match.group(1)) if match else float("inf")

    best = min(ckpts, key=extract_val_loss)
    return best


def extract_epoch(path: Path) -> str:
    """Extract epoch number from checkpoint filename."""
    match = re.search(r"epoch=(\d+)", path.name)
    return match.group(1) if match else "?"


def save_wav(audio_np: np.ndarray, filepath: Path, sample_rate: int = 22050):
    """Save a numpy audio array as a 16-bit WAV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (audio_np.clip(-1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def griffin_lim(spectrogram: np.ndarray, n_fft: int = 1024, hop_length: int = 256,
               n_iter: int = 60) -> np.ndarray:
    """Simple Griffin-Lim algorithm to convert mel spectrogram to audio.

    This is a rough approximation — HiFi-GAN will sound much better, but
    this works without downloading any extra model.
    """
    import librosa
    # spectrogram is log-mel, convert back to linear
    S = np.exp(spectrogram)
    audio = librosa.griffinlim(
        S, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft,
        window="hann", center=True, length=None
    )
    # Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    return audio


def main():
    parser = argparse.ArgumentParser(description="Test FastPitch audio quality")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to FastPitch .ckpt file (default: auto-find best)")
    parser.add_argument("--hifigan", type=str, default="tts_en_hifigan",
                        help="Pretrained HiFi-GAN model name or .nemo path (default: tts_en_hifigan)")
    parser.add_argument("--griffin-lim", action="store_true",
                        help="Use Griffin-Lim instead of HiFi-GAN (no download needed)")
    parser.add_argument("--sentences", type=str, nargs="*", default=None,
                        help="Custom sentences to synthesize (default: built-in Shona set)")
    args = parser.parse_args()

    # ── Find checkpoint ───────────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"❌ Checkpoint not found: {ckpt_path}")
            sys.exit(1)
    else:
        ckpt_path = find_best_checkpoint()

    epoch = extract_epoch(ckpt_path)
    print(f"\n{'='*60}")
    print(f"  🎤 FastPitch Audio Test")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Epoch      : {epoch}")
    print(f"  Vocoder    : {'Griffin-Lim' if args.griffin_lim else args.hifigan}")
    print(f"{'='*60}\n")

    # ── Load FastPitch from .ckpt ─────────────────────────────────────────────
    print("⏳ Loading FastPitch model...")
    from omegaconf import OmegaConf
    from nemo.collections.tts.models import FastPitchModel

    cfg = OmegaConf.load(str(CONFIG_PATH))
    model = FastPitchModel(cfg=cfg.model)
    # Load weights from checkpoint (Lightning checkpoint format)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # Handle both plain state_dict and Lightning wrapped checkpoints
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  ✅ FastPitch loaded on {device}")

    # ── Load vocoder ──────────────────────────────────────────────────────────
    vocoder = None
    if not args.griffin_lim:
        print(f"⏳ Loading HiFi-GAN vocoder ({args.hifigan})...")
        from nemo.collections.tts.models import HifiGanModel
        if Path(args.hifigan).exists():
            vocoder = HifiGanModel.restore_from(args.hifigan)
        else:
            vocoder = HifiGanModel.from_pretrained(model_name=args.hifigan)
        vocoder = vocoder.eval().to(device)
        print(f"  ✅ HiFi-GAN loaded")

    # ── Create output directory ───────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vocoder_name = "griffinlim" if args.griffin_lim else "hifigan"
    run_dir = OUTPUT_DIR / f"epoch{epoch}_{vocoder_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output: {run_dir}\n")

    # ── Synthesize ────────────────────────────────────────────────────────────
    sentences = args.sentences if args.sentences else TEST_SENTENCES
    sample_rate = 22050

    for i, text in enumerate(sentences, 1):
        print(f"  [{i}/{len(sentences)}] \"{text}\"")

        try:
            with torch.no_grad():
                # Parse text to token IDs
                parsed = model.parse(text)
                parsed = parsed.to(device)

                # Generate mel spectrogram
                spectrogram = model.generate_spectrogram(tokens=parsed)

                if args.griffin_lim:
                    # Griffin-Lim path
                    mel_np = spectrogram.squeeze().cpu().numpy()
                    audio_np = griffin_lim(mel_np, n_fft=1024, hop_length=256)
                else:
                    # HiFi-GAN vocoder path
                    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
                    audio_np = audio.squeeze().cpu().numpy()

            # Save WAV
            safe_name = re.sub(r'[^\w\s]', '', text)[:30].strip().replace(' ', '_').lower()
            wav_path = run_dir / f"{i:02d}_{safe_name}.wav"
            save_wav(audio_np, wav_path, sample_rate)

            duration = len(audio_np) / sample_rate
            print(f"           → {wav_path.name} ({duration:.2f}s)")

        except Exception as e:
            print(f"           ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    wav_count = len(list(run_dir.glob("*.wav")))
    print(f"\n{'='*60}")
    print(f"  ✅ Done! {wav_count}/{len(sentences)} sentences synthesized")
    print(f"  📁 Output: {run_dir}")
    print(f"")
    print(f"  Listen with:")
    print(f"    aplay {run_dir}/*.wav")
    print(f"  Or copy to your machine:")
    print(f"    scp blaquesoul@<host>:{run_dir}/*.wav .")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
