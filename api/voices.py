"""Voice catalog.

Bootstrap catalog with one female voice per supported language. Wire to a
DB-backed catalog in Phase D.2 once tenant-scoped permissions are needed.
"""

from dataclasses import dataclass

from frontend.registry import SUPPORTED_LANGUAGES


@dataclass(frozen=True)
class Voice:
    voice_id: str
    language: str
    gender: str
    sample_rate: int


_CATALOG: dict[str, Voice] = {
    f"{lang}_female_1": Voice(
        voice_id=f"{lang}_female_1",
        language=lang,
        gender="female",
        sample_rate=22050,
    )
    for lang in sorted(SUPPORTED_LANGUAGES)
}


def list_voices() -> list[Voice]:
    return list(_CATALOG.values())


def get_voice(voice_id: str) -> Voice | None:
    return _CATALOG.get(voice_id)
