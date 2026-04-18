"""Shona text normalisation.

Bootstrap numerals 0-10 (the smallest set worth shipping); larger ranges and
abbreviation handling require native-speaker input. ``digit_words`` covers
single digits so multi-digit numbers degrade gracefully ("12" -> "imwe mbiri")
rather than emitting English fallback.
"""

from frontend.base.normalizer import BantuNormalizer


SHONA_DIGITS = {
    "0": "zero",
    "1": "imwe",
    "2": "mbiri",
    "3": "nhatu",
    "4": "ina",
    "5": "shanu",
    "6": "tanhatu",
    "7": "nomwe",
    "8": "rusere",
    "9": "pfumbamwe",
    "10": "gumi",
}

SHONA_ABBREVIATIONS: dict[str, str] = {
    # Seed only; expand with native-speaker review.
}


def build_normalizer() -> BantuNormalizer:
    return BantuNormalizer(
        language="shona",
        digit_words=SHONA_DIGITS,
        abbreviations=SHONA_ABBREVIATIONS,
    )
