"""Per-language frontend registry.

Single dispatch point used by inference, training, and tests. Adding a new
language means: create ``frontend/<lang>/`` with ``phonemes`` + ``normalizer``
modules, then register it here.
"""

from dataclasses import dataclass

from frontend.base.g2p import BantuG2P
from frontend.base.normalizer import BantuNormalizer
from frontend.base.phonemes import PhonemeInventory


@dataclass(frozen=True)
class LanguagePack:
    inventory: PhonemeInventory
    normalizer: BantuNormalizer
    g2p: BantuG2P


def _build(language: str) -> LanguagePack:
    if language == "shona":
        from frontend.shona import normalizer as norm
        from frontend.shona import phonemes
    elif language == "ndebele":
        from frontend.ndebele import normalizer as norm
        from frontend.ndebele import phonemes
    elif language == "zulu":
        from frontend.zulu import normalizer as norm
        from frontend.zulu import phonemes
    elif language == "xhosa":
        from frontend.xhosa import normalizer as norm
        from frontend.xhosa import phonemes
    else:
        raise ValueError(f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}")
    return LanguagePack(
        inventory=phonemes.INVENTORY,
        normalizer=norm.build_normalizer(),
        g2p=BantuG2P(inventory=phonemes.INVENTORY),
    )


SUPPORTED_LANGUAGES = frozenset({"shona", "ndebele", "zulu", "xhosa"})


_PACKS: dict[str, LanguagePack] = {}


def get_pack(language: str) -> LanguagePack:
    """Return the cached :class:`LanguagePack` for ``language``."""
    if language not in _PACKS:
        _PACKS[language] = _build(language)
    return _PACKS[language]
