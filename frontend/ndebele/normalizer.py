"""Ndebele text normalisation."""

from frontend.base.normalizer import BantuNormalizer


NDEBELE_DIGITS = {
    "0": "iqanda",
    "1": "kunye",
    "2": "kubili",
    "3": "kuthathu",
    "4": "kune",
    "5": "isihlanu",
    "6": "isithupha",
    "7": "isikhombisa",
    "8": "isificaminwembili",
    "9": "isificalolunye",
    "10": "itshumi",
}

NDEBELE_ABBREVIATIONS: dict[str, str] = {}


def build_normalizer() -> BantuNormalizer:
    return BantuNormalizer(
        language="ndebele",
        digit_words=NDEBELE_DIGITS,
        abbreviations=NDEBELE_ABBREVIATIONS,
    )
