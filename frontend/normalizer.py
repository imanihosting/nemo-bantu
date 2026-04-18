"""Routing shim for backward compatibility.

Existing callers do ``from frontend.normalizer import normalize_text``. The
real work now lives in per-language :class:`BantuNormalizer` instances built
by :mod:`frontend.registry`.
"""

from frontend.registry import SUPPORTED_LANGUAGES, get_pack


def normalize_text(text: str, language: str) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}")
    return get_pack(language).normalizer.normalize(text)
