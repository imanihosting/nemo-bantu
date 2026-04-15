from frontend.normalizer import normalize_text
from frontend.g2p import text_to_phonemes
from inference.synthesize import SynthesisResult, synthesize_from_phonemes


def run_tts_pipeline(text: str, language: str, voice: str, speed: float) -> tuple[SynthesisResult, dict]:
    normalized = normalize_text(text=text, language=language)
    phonemes = text_to_phonemes(text=normalized, language=language)
    result = synthesize_from_phonemes(phonemes=phonemes, language=language, voice=voice, speed=speed)
    debug_info = {"normalized_text": normalized, "phonemes": phonemes, "language": language, "voice": voice}
    return result, debug_info
