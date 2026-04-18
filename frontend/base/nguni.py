"""Shared phoneme building blocks for the Nguni branch (Zulu, Xhosa, Ndebele).

The Nguni languages share a five-vowel inventory plus three click places
(dental ǀ, alveolar ǃ, lateral ǁ) crossed with five manner series:
plain, aspirated, voiced, nasal, breathy-voiced nasal.

The click set is grapheme-encoded in standard Nguni orthography:

    place      | plain | aspirated | voiced | nasal | breathy-nasal
    ---------- | ----- | --------- | ------ | ----- | -------------
    dental ǀ   | c     | ch        | gc     | nc    | ngc
    alveolar ǃ | q     | qh        | gq     | nq    | ngq
    lateral ǁ  | x     | xh        | gx     | nx    | ngx

Greedy longest-match is mandatory: ``ngc`` must beat ``nc`` must beat ``c``.
The :class:`PhonemeInventory.grapheme_table` ordering enforces this.
"""


NGUNI_VOWELS = {
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",
}

NGUNI_PLAIN_CONSONANTS = {
    "p": "p", "b": "ɓ",        # b is implosive in Zulu/Xhosa
    "t": "t", "d": "d", "k": "k", "g": "ɡ",
    "f": "f", "v": "v", "s": "s", "z": "z", "h": "h",
    "m": "m", "n": "n", "r": "r", "l": "l", "y": "j", "w": "w", "j": "dʒ",
}

NGUNI_DIGRAPHS = {
    # Aspirated stops
    "ph": "pʰ", "th": "tʰ", "kh": "kʰ",
    # Voiced/breathy stops written with -h
    "bh": "b", "dh": "d", "gh": "ɡ",
    # Affricates / palatal series
    "ch": "tʃ", "sh": "ʃ", "tsh": "tʃʰ",
    # Palatal & velar nasals
    "ny": "ɲ", "ng'": "ŋ",
    # Prenasalised stops/fricatives
    "mb": "ᵐb", "mp": "ᵐp",
    "nd": "ⁿd", "nt": "ⁿt", "nz": "ⁿz", "ns": "ⁿs",
    "ng": "ᵑɡ", "nk": "ᵑk",
    "nj": "ⁿdʒ", "hl": "ɬ", "dl": "ɮ",
}

# Click consonants. Keys are Nguni orthography graphemes; values are IPA.
# Trigraphs (ngc/ngq/ngx) MUST come first so the scanner picks them before
# the digraphs/single chars. Order in the dict does not matter — the scanner
# sorts by descending length — but we list them this way for readability.
NGUNI_CLICKS = {
    # Dental ǀ
    "ngc": "ŋǀʱ", "nc": "ŋǀ", "gc": "ɡǀ", "ch": "ǀʰ", "c": "ǀ",
    # Alveolar ǃ
    "ngq": "ŋǃʱ", "nq": "ŋǃ", "gq": "ɡǃ", "qh": "ǃʰ", "q": "ǃ",
    # Lateral ǁ
    "ngx": "ŋǁʱ", "nx": "ŋǁ", "gx": "ɡǁ", "xh": "ǁʰ", "x": "ǁ",
}
