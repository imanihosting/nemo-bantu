from frontend.normalizer import normalize_text
from inference.synthesize import SynthesisResult, NemoSynthesizer


def run_tts_pipeline(text: str, language: str, voice: str, speed: float) -> tuple[SynthesisResult, dict]:
    """Run the full TTS pipeline: normalize text → synthesize audio.

    Note: G2P is skipped because FastPitch uses its own character-level tokenizer.
    """
    normalized = normalize_text(text=text, language=language)
    synth = NemoSynthesizer.get_instance()
    result = synth.synthesize(text=normalized, speed=speed)
    debug_info = {"normalized_text": normalized, "language": language, "voice": voice}
    return result, debug_info
