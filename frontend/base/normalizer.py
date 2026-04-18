"""Shared text normalisation primitives.

Each language subclasses :class:`BantuNormalizer` and supplies its own
``digit_words``, ``abbreviations``, and (optionally) overrides for ordinal,
currency, or date handling. The base class handles the mechanical work of
case-folding, whitespace collapse, punctuation stripping, and dispatch.
"""

from dataclasses import dataclass, field
import re


@dataclass
class BantuNormalizer:
    language: str
    digit_words: dict[str, str] = field(default_factory=dict)
    abbreviations: dict[str, str] = field(default_factory=dict)
    keep_punctuation: str = ".,!?;:"

    _whitespace = re.compile(r"\s+")
    _token_split = re.compile(r"(\s+)")

    def normalize(self, text: str) -> str:
        cleaned = self._whitespace.sub(" ", text.strip().lower())
        tokens = self._token_split.split(cleaned)
        out: list[str] = []
        for tok in tokens:
            if not tok or tok.isspace():
                out.append(tok)
                continue
            out.append(self._normalize_token(tok))
        joined = "".join(out)
        return self._strip_punctuation(joined)

    def _normalize_token(self, token: str) -> str:
        if token in self.abbreviations:
            return self.abbreviations[token]
        if token.rstrip(self.keep_punctuation).isdigit():
            digits = token.rstrip(self.keep_punctuation)
            tail = token[len(digits):]
            return self._expand_number(digits) + tail
        return token

    def _expand_number(self, digits: str) -> str:
        if digits in self.digit_words:
            return self.digit_words[digits]
        return " ".join(self.digit_words.get(ch, ch) for ch in digits)

    def _strip_punctuation(self, text: str) -> str:
        keep = set(self.keep_punctuation)
        return "".join(ch for ch in text if ch.isalnum() or ch.isspace() or ch in keep or ch == "'")
