"""Routing shim for backward compatibility.

Existing callers do ``from frontend.g2p import text_to_phonemes`` and pass a
``language`` kwarg. We keep that contract and dispatch to the per-language
:class:`BantuG2P` registered in :mod:`frontend.registry`.
"""

from frontend.registry import SUPPORTED_LANGUAGES, get_pack


def text_to_phonemes(text: str, language: str) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}")
    return get_pack(language).g2p.phonemise(text).phonemes
