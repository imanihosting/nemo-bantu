import re


_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def _expand_digits(token: str) -> str:
    if not token.isdigit():
        return token
    return " ".join(_DIGIT_WORDS[ch] for ch in token)


def normalize_text(text: str, language: str) -> str:
    """Basic normalization scaffold. Extend per-language rules over time."""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    tokens = cleaned.split(" ")
    expanded = [_expand_digits(tok) for tok in tokens]
    return " ".join(expanded)
