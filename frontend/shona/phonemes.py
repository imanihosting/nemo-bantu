"""Shona (sna) phoneme inventory.

Five-vowel system, no click consonants. Distinguishing features versus the
Nguni languages: whistled fricatives ``sv``/``zv`` (rare cross-linguistically)
and prenasalised stop series ``mb``/``nd``/``ng``/``nz``.

IPA mappings follow standard Bantu phonology references; native-speaker
review is required before production training.
"""

from frontend.base.phonemes import PhonemeInventory


VOWELS = {
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",
}

PLAIN_CONSONANTS = {
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "ɡ",
    "f": "f", "v": "v", "s": "s", "z": "z", "h": "h",
    "m": "m", "n": "n", "r": "r", "l": "l", "j": "dʒ", "y": "j", "w": "w",
    "c": "tʃ",
}

DIGRAPHS = {
    # Aspirated stops
    "ph": "pʰ", "th": "tʰ", "kh": "kʰ",
    # Voiced/breathy stops written with -h
    "bh": "b", "dh": "d", "gh": "ɡ",
    # Affricate / palatal series
    "ch": "tʃ", "sh": "ʃ", "zh": "ʒ",
    # Palatal nasal & velar nasal
    "ny": "ɲ", "ng'": "ŋ",
    # Prenasalised stops/fricatives — fundamental to Bantu phonology
    "mb": "ᵐb", "mp": "ᵐp",
    "nd": "ⁿd", "nt": "ⁿt", "nz": "ⁿz", "ns": "ⁿs",
    "ng": "ᵑɡ", "nk": "ᵑk",
    "nj": "ⁿdʒ", "nc": "ⁿtʃ",
}

# Shona's signature whistled fricatives. Modelled as labialised sibilants for
# now; phonetically they are closer to whistled coronal fricatives. Audit
# against trained model output.
WHISTLED = {
    "sv": "sʷ",
    "zv": "zʷ",
}

INVENTORY = PhonemeInventory(
    language="shona",
    vowels=VOWELS,
    consonants=PLAIN_CONSONANTS,
    digraphs=DIGRAPHS,
    clicks={},  # Shona has no clicks
    whistled=WHISTLED,
    notes="Five-vowel Bantu inventory. Whistled fricatives sv/zv are Shona-specific and need native-speaker validation.",
)
