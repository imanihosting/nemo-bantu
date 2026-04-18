"""Lexicon-first, rule-based G2P scanner shared across Bantu languages.

Resolution order per word:
    1. Override dictionary (per-language hardcoded fixes)
    2. Lexicon file (``frontend/lexicons/<language>.txt``)
    3. Greedy longest-match grapheme scan against the language's
       :class:`PhonemeInventory`

The scanner emits IPA phoneme tokens separated by spaces; words are joined by
the :data:`WORD_BOUNDARY` token so a downstream tokenizer can recover word
boundaries during training.
"""

from dataclasses import dataclass
from pathlib import Path
import re

from frontend.base.phonemes import PhonemeInventory, WORD_BOUNDARY


LEXICON_ROOT = Path(__file__).resolve().parent.parent / "lexicons"
WORD_PATTERN = re.compile(r"[a-zA-Z'\u01c0\u01c1\u01c2\u01c3\u00e0-\u00ff]+")


@dataclass(frozen=True)
class G2PResult:
    """Phonemised output plus debug breadcrumbs for native-speaker review."""

    phonemes: str
    word_sources: list[tuple[str, str, str]]  # (word, source, phonemes)


class BantuG2P:
    def __init__(
        self,
        inventory: PhonemeInventory,
        overrides: dict[str, str] | None = None,
        lexicon_path: Path | None = None,
    ) -> None:
        self.inventory = inventory
        self.overrides = {k.lower(): v for k, v in (overrides or {}).items()}
        self._grapheme_table = inventory.grapheme_table()
        self._lexicon_path = lexicon_path or (LEXICON_ROOT / f"{inventory.language}.txt")
        self._lexicon = self._load_lexicon(self._lexicon_path)

    @staticmethod
    def _load_lexicon(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        entries: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                entries[parts[0].lower()] = parts[1]
        return entries

    def phonemise_word(self, word: str) -> tuple[str, str]:
        """Return ``(phonemes, source_label)`` for a single word.

        ``source_label`` is one of ``override`` | ``lexicon`` | ``rule``.
        """
        key = word.lower()
        if key in self.overrides:
            return self.overrides[key], "override"
        if key in self._lexicon:
            return self._lexicon[key], "lexicon"
        return self._scan(key), "rule"

    def _scan(self, word: str) -> str:
        pieces: list[str] = []
        i = 0
        while i < len(word):
            match = self._longest_grapheme_at(word, i)
            if match is None:
                # Unknown character — keep it verbatim so audit catches it
                # rather than silently dropping it. Native-speaker review will
                # decide whether to add a grapheme or normalise the input.
                pieces.append(word[i])
                i += 1
                continue
            grapheme, ipa = match
            pieces.append(ipa)
            i += len(grapheme)
        return " ".join(pieces)

    def _longest_grapheme_at(self, word: str, start: int) -> tuple[str, str] | None:
        for grapheme, ipa in self._grapheme_table:
            if word.startswith(grapheme, start):
                return grapheme, ipa
        return None

    def phonemise(self, text: str) -> G2PResult:
        words = WORD_PATTERN.findall(text)
        rendered: list[str] = []
        sources: list[tuple[str, str, str]] = []
        for word in words:
            phonemes, source = self.phonemise_word(word)
            rendered.append(phonemes)
            sources.append((word, source, phonemes))
        joined = f" {WORD_BOUNDARY} ".join(rendered)
        return G2PResult(phonemes=joined, word_sources=sources)
