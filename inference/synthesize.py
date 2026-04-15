from dataclasses import dataclass
import io
import wave
from pathlib import Path
from typing import Optional

try:
    from nemo.collections.tts.models import FastPitchModel, HifiGanModel
except Exception:  # pragma: no cover - optional dependency
    FastPitchModel = None
    HifiGanModel = None


@dataclass
class SynthesisResult:
    audio_bytes: bytes
    sample_rate: int
    format: str


class NemoSynthesizer:
    def __init__(self, fastpitch_checkpoint: str, hifigan_checkpoint: str, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.fastpitch: Optional[object] = None
        self.hifigan: Optional[object] = None
        self.is_loaded = False
        self._try_load(fastpitch_checkpoint=fastpitch_checkpoint, hifigan_checkpoint=hifigan_checkpoint)

    def _try_load(self, fastpitch_checkpoint: str, hifigan_checkpoint: str) -> None:
        if FastPitchModel is None or HifiGanModel is None:
            return

        fp_path = Path(fastpitch_checkpoint) if fastpitch_checkpoint else None
        hg_path = Path(hifigan_checkpoint) if hifigan_checkpoint else None
        if not fp_path or not hg_path or not fp_path.exists() or not hg_path.exists():
            return

        self.fastpitch = FastPitchModel.restore_from(str(fp_path))
        self.hifigan = HifiGanModel.restore_from(str(hg_path))
        self.is_loaded = True

    def synthesize(self, text: str, speed: float) -> SynthesisResult:
        if not self.is_loaded:
            return SynthesisResult(audio_bytes=_silent_wav(sample_rate=self.sample_rate), sample_rate=self.sample_rate, format="wav")

        if hasattr(self.fastpitch, "parse"):  # NeMo text frontend path
            parsed = self.fastpitch.parse(text)
            spectrogram = self.fastpitch.generate_spectrogram(tokens=parsed, pace=speed)
            audio = self.hifigan.convert_spectrogram_to_audio(spec=spectrogram)
            audio_np = audio.squeeze().detach().cpu().numpy()
        else:
            return SynthesisResult(audio_bytes=_silent_wav(sample_rate=self.sample_rate), sample_rate=self.sample_rate, format="wav")

        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            pcm16 = (audio_np.clip(-1.0, 1.0) * 32767.0).astype("int16").tobytes()
            wav_file.writeframes(pcm16)
        return SynthesisResult(audio_bytes=output.getvalue(), sample_rate=self.sample_rate, format="wav")


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


def synthesize_from_phonemes(phonemes: str, language: str, voice: str, speed: float) -> SynthesisResult:
    # Phase-2 behavior: attempts NeMo inference and gracefully falls back to silence.
    _ = language, voice
    synth = NemoSynthesizer(
        fastpitch_checkpoint="models/fastpitch.nemo",
        hifigan_checkpoint="models/hifigan.nemo",
        sample_rate=22050,
    )
    return synth.synthesize(text=phonemes, speed=speed)
