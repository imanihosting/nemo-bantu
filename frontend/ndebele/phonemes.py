"""Ndebele (Northern, ISO nde) phoneme inventory.

Northern Ndebele (Zimbabwean) is descended from Zulu speakers and inherits
the Nguni click inventory. The five-vowel system and digraph set are
identical to Zulu in this scaffold.
"""

from frontend.base.nguni import (
    NGUNI_CLICKS,
    NGUNI_DIGRAPHS,
    NGUNI_PLAIN_CONSONANTS,
    NGUNI_VOWELS,
)
from frontend.base.phonemes import PhonemeInventory


_DIGRAPHS = {k: v for k, v in NGUNI_DIGRAPHS.items() if k not in {"ch"}}


INVENTORY = PhonemeInventory(
    language="ndebele",
    vowels=NGUNI_VOWELS,
    consonants=NGUNI_PLAIN_CONSONANTS,
    digraphs=_DIGRAPHS,
    clicks=NGUNI_CLICKS,
    whistled={},
    notes="Northern Ndebele (Zimbabwe) inherits the full Nguni click set from Zulu. Native-speaker review needed for click frequency vs. plain-consonant variants.",
)
