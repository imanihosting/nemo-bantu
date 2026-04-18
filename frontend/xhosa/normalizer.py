"""Xhosa text normalisation."""

from frontend.base.normalizer import BantuNormalizer


XHOSA_DIGITS = {
    "0": "iqanda",
    "1": "inye",
    "2": "zimbini",
    "3": "zintathu",
    "4": "zine",
    "5": "zintlanu",
    "6": "zintandathu",
    "7": "zisixhenxe",
    "8": "zisibhozo",
    "9": "zilithoba",
    "10": "lishumi",
}

XHOSA_ABBREVIATIONS: dict[str, str] = {}


def build_normalizer() -> BantuNormalizer:
    return BantuNormalizer(
        language="xhosa",
        digit_words=XHOSA_DIGITS,
        abbreviations=XHOSA_ABBREVIATIONS,
    )
