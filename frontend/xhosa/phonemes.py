"""Xhosa (xho) phoneme inventory.

Same click inventory as Zulu in this scaffold. Xhosa actually has a richer
click distribution and additional ejective and slack-voiced contrasts that
should be added during native-speaker review (see ``notes``).
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
    language="xhosa",
    vowels=NGUNI_VOWELS,
    consonants=NGUNI_PLAIN_CONSONANTS,
    digraphs=_DIGRAPHS,
    clicks=NGUNI_CLICKS,
    whistled={},
    notes="Xhosa shares the Nguni click set; additional ejective/slack-voice contrasts should be added after native-speaker review.",
)
