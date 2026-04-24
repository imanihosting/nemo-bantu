"""TTS synthesizer using FastPitch + HiFi-GAN checkpoints.

Loads models once at import time (singleton) and reuses them for all requests.
Uses the same checkpoint-loading approach as scripts/test_audio.py.
"""
import io
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── PyTorch 2.6 weights_only workaround ──────────────────────────────────────
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

PROJECT_DIR = Path(__file__).resolve().parent.parent
FASTPITCH_CONFIG = PROJECT_DIR / "configs" / "training" / "fastpitch_shona.yaml"
FASTPITCH_CKPT_DIR = PROJECT_DIR / "outputs" / "fastpitch_shona" / "FastPitch_Shona" / "checkpoints"
HIFIGAN_CKPT_DIR = PROJECT_DIR / "outputs" / "hifigan_shona" / "HiFiGAN_Shona" / "checkpoints"


@dataclass
class SynthesisResult:
    audio_bytes: bytes
    sample_rate: int
    format: str


def _find_best_checkpoint(ckpt_dir: Path, prefix: str) -> Optional[Path]:
    """Find the checkpoint with the lowest val_loss in a directory."""
    if not ckpt_dir.exists():
        return None
    pattern = re.compile(rf"{prefix}--val_loss=([\d.]+)-epoch=(\d+)\.ckpt$")
    best_path, best_loss = None, float("inf")
    for f in ckpt_dir.glob(f"{prefix}--*.ckpt"):
        # Skip the '-last' checkpoints (duplicates of the latest)
        if "-last" in f.name:
            continue
        m = pattern.search(f.name)
        if m:
            loss = float(m.group(1))
            if loss < best_loss:
                best_loss = loss
                best_path = f
    return best_path


def _silent_wav(duration_seconds: float = 0.6, sample_rate: int = 22050) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    samples = b"\x00\x00" * frame_count
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples)
    return output.getvalue()


class NemoSynthesizer:
    """Singleton-style TTS synthesizer using FastPitch + HiFi-GAN checkpoints."""

    _instance: Optional["NemoSynthesizer"] = None

    def __init__(self):
        self.sample_rate = 22050
        self.fastpitch = None
        self.hifigan = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

    @classmethod
    def get_instance(cls) -> "NemoSynthesizer":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_models()
        return cls._instance

    @classmethod
    def reload(cls) -> "NemoSynthesizer":
        """Clear cached models and reload from the latest checkpoints."""
        cls._instance = None
        import gc; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
        return cls.get_instance()

    def _load_models(self) -> None:
        """Load FastPitch and HiFi-GAN from training checkpoints."""
        try:
            from omegaconf import OmegaConf
            from nemo.collections.tts.models import FastPitchModel, HifiGanModel

            # ── Load FastPitch ────────────────────────────────────────────
            fp_ckpt = _find_best_checkpoint(FASTPITCH_CKPT_DIR, "FastPitch_Shona")
            if fp_ckpt is None:
                print("⚠️  No FastPitch checkpoint found")
                return

            print(f"⏳ Loading FastPitch from {fp_ckpt.name}...")
            cfg = OmegaConf.load(str(FASTPITCH_CONFIG))
            self.fastpitch = FastPitchModel(cfg=cfg.model)
            ckpt_data = torch.load(str(fp_ckpt), map_location="cpu")
            state_dict = ckpt_data.get("state_dict", ckpt_data)
            self.fastpitch.load_state_dict(state_dict, strict=False)
            self.fastpitch.eval().to(self.device)
            print(f"  ✅ FastPitch loaded ({fp_ckpt.name}) on {self.device}")

            # ── Load HiFi-GAN ─────────────────────────────────────────────
            hg_ckpt = _find_best_checkpoint(HIFIGAN_CKPT_DIR, "HiFiGAN_Shona")
            if hg_ckpt:
                print(f"⏳ Loading fine-tuned HiFi-GAN from {hg_ckpt.name}...")
                self.hifigan = HifiGanModel.from_pretrained(model_name="tts_en_hifigan")
                hg_data = torch.load(str(hg_ckpt), map_location="cpu")
                hg_state = hg_data.get("state_dict", hg_data)
                self.hifigan.load_state_dict(hg_state, strict=False)
            else:
                print("⏳ Loading pretrained English HiFi-GAN (no Shona checkpoint found)...")
                self.hifigan = HifiGanModel.from_pretrained(model_name="tts_en_hifigan")

            self.hifigan.eval().to(self.device)
            print(f"  ✅ HiFi-GAN loaded on {self.device}")

            self.is_loaded = True

        except Exception as exc:
            print(f"❌ Model loading failed: {exc}")
            import traceback
            traceback.print_exc()

    def synthesize(self, text: str, speed: float = 1.0) -> SynthesisResult:
        """Synthesize speech from raw Shona text."""
        if not self.is_loaded:
            return SynthesisResult(
                audio_bytes=_silent_wav(sample_rate=self.sample_rate),
                sample_rate=self.sample_rate,
                format="wav",
            )

        with torch.no_grad():
            tokens = self.fastpitch.parse(text)
            spectrogram = self.fastpitch.generate_spectrogram(tokens=tokens, pace=speed)
            audio = self.hifigan.convert_spectrogram_to_audio(spec=spectrogram)

        audio_np = audio.squeeze().detach().cpu().numpy()

        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            pcm16 = (audio_np.clip(-1.0, 1.0) * 32767.0).astype("int16").tobytes()
            wav_file.writeframes(pcm16)

        return SynthesisResult(
            audio_bytes=output.getvalue(),
            sample_rate=self.sample_rate,
            format="wav",
        )


def synthesize_from_phonemes(phonemes: str, language: str, voice: str, speed: float) -> SynthesisResult:
    """Called by the pipeline. Passes text directly to FastPitch (which has its own tokenizer)."""
    _ = language, voice  # reserved for future multi-voice support
    synth = NemoSynthesizer.get_instance()
    return synth.synthesize(text=phonemes, speed=speed)
