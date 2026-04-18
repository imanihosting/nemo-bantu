"""Zulu text normalisation."""

from frontend.base.normalizer import BantuNormalizer


ZULU_DIGITS = {
    "0": "iqanda",
    "1": "kunye",
    "2": "kubili",
    "3": "kuthathu",
    "4": "kune",
    "5": "isihlanu",
    "6": "isithupha",
    "7": "isikhombisa",
    "8": "isishiyagalombili",
    "9": "isishiyagalolunye",
    "10": "ishumi",
}

ZULU_ABBREVIATIONS: dict[str, str] = {}


def build_normalizer() -> BantuNormalizer:
    return BantuNormalizer(
        language="zulu",
        digit_words=ZULU_DIGITS,
        abbreviations=ZULU_ABBREVIATIONS,
    )
