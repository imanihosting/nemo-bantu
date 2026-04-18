"""Phoneme inventory primitives shared across Bantu languages.

Each language ships its own `PhonemeInventory` describing the IPA-mapped
graphemes it understands. The G2P scanner consumes this inventory to perform
greedy longest-match grapheme tokenisation.
"""

from dataclasses import dataclass, field


WORD_BOUNDARY = "|"
SILENCE = "_"
PAD = "<pad>"
UNK = "<unk>"


@dataclass(frozen=True)
class PhonemeInventory:
    """Per-language phoneme inventory.

    Attributes:
        language: ISO language tag (e.g. ``shona``).
        vowels: ordered grapheme -> IPA mapping for vowels.
        consonants: ordered grapheme -> IPA mapping for plain consonants.
        digraphs: multi-character graphemes mapped to IPA. The G2P scanner
            tries longer keys first so trigraphs like ``ngc`` (Nguni breathy
            nasal click) win over ``nc``/``ng``.
        clicks: subset of digraphs that represent click consonants. Empty for
            non-click languages (Shona). Tracked separately so we can audit
            click coverage in tests.
        whistled: subset of digraphs that represent whistled fricatives
            (Shona-specific ``sv``/``zv``). Empty otherwise.
        notes: free-form documentation for native-speaker review.
    """

    language: str
    vowels: dict[str, str]
    consonants: dict[str, str]
    digraphs: dict[str, str] = field(default_factory=dict)
    clicks: dict[str, str] = field(default_factory=dict)
    whistled: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def grapheme_table(self) -> list[tuple[str, str]]:
        """Return (grapheme, IPA) pairs sorted by descending grapheme length.

        Longest match first is required so that ``ngc`` (trigraph) is matched
        before ``ng`` (digraph) before ``n`` (single character). The scanner
        relies on this ordering being stable.
        """
        merged: dict[str, str] = {}
        merged.update(self.consonants)
        merged.update(self.vowels)
        merged.update(self.digraphs)
        merged.update(self.clicks)
        merged.update(self.whistled)
        return sorted(merged.items(), key=lambda pair: (-len(pair[0]), pair[0]))

    def all_phonemes(self) -> set[str]:
        """Set of every IPA phoneme this language can emit."""
        return {ipa for _, ipa in self.grapheme_table()}
