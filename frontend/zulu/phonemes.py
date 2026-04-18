"""Zulu (zul) phoneme inventory.

Standard Nguni five-vowel system with the full three-place click series.
"""

from frontend.base.nguni import (
    NGUNI_CLICKS,
    NGUNI_DIGRAPHS,
    NGUNI_PLAIN_CONSONANTS,
    NGUNI_VOWELS,
)
from frontend.base.phonemes import PhonemeInventory


# ``ch`` and ``xh`` collide between the affricate digraph table and the click
# table in Nguni orthography. In Zulu, ``ch`` and ``xh`` ARE clicks; the
# affricate ``ch`` interpretation belongs to other languages. Override here.
_DIGRAPHS = {k: v for k, v in NGUNI_DIGRAPHS.items() if k not in {"ch"}}


INVENTORY = PhonemeInventory(
    language="zulu",
    vowels=NGUNI_VOWELS,
    consonants=NGUNI_PLAIN_CONSONANTS,
    digraphs=_DIGRAPHS,
    clicks=NGUNI_CLICKS,
    whistled={},
    notes="Zulu: full Nguni click inventory (dental/alveolar/lateral × plain/aspirated/voiced/nasal/breathy-nasal). Implosive /ɓ/ for grapheme b.",
)
