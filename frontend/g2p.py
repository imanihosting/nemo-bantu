from pathlib import Path
import re


SUPPORTED_LANGUAGES = {"shona", "ndebele", "zulu", "xhosa"}
PRENASALIZED_CLUSTERS = ("mb", "nd", "ng")


def _load_lexicon(language: str) -> dict[str, str]:
    lexicon_path = Path("frontend/lexicons") / f"{language}.txt"
    if not lexicon_path.exists():
        return {}
    entries: dict[str, str] = {}
    for line in lexicon_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            entries[parts[0].lower()] = parts[1]
    return entries


def _fallback_rule_based(word: str) -> str:
    # Preserve prenazalized clusters as single phones in fallback mode.
    pieces: list[str] = []
    i = 0
    while i < len(word):
        cluster = next((c for c in PRENASALIZED_CLUSTERS if word.startswith(c, i)), None)
        if cluster is not None:
            pieces.append(cluster)
            i += len(cluster)
            continue
        pieces.append(word[i])
        i += 1
    return " ".join(pieces)


def text_to_phonemes(text: str, language: str) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}")

    lexicon = _load_lexicon(language)
    words = re.findall(r"[a-zA-Z']+", text)
    phoneme_words = []
    for word in words:
        key = word.lower()
        phoneme_words.append(lexicon.get(key, _fallback_rule_based(key)))
    return " | ".join(phoneme_words)
