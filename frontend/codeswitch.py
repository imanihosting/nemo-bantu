"""Code-switching detection.

Southern African Bantu speakers routinely insert English words into Bantu
sentences ("ndiri busy" / "siya-shopping"). The TTS frontend needs to flag
those spans so they can be routed to an English G2P at training/inference
time. This module provides cheap heuristic detection — full LID modelling is
out of scope for the bootstrap.
"""

from dataclasses import dataclass
import re


# Compact stoplist of high-frequency English words that commonly code-switch
# into Southern African Bantu speech. Keep small; precision over recall.
ENGLISH_HINTS = frozenset({
    "the", "and", "but", "with", "for", "from", "okay", "ok", "please",
    "thank", "thanks", "sorry", "yes", "no", "now", "today", "tomorrow",
    "phone", "email", "internet", "google", "whatsapp", "facebook",
    "money", "bank", "shop", "shopping", "busy", "free", "weekend",
    "office", "meeting", "online", "school", "work", "home",
})

# Bantu orthography is overwhelmingly lowercase ASCII; non-Latin diacritics
# are rare. Detect "looks English" by the presence of English-only graphemes
# (q/x as plain consonants are valid in Nguni clicks, so we cannot use them
# as English markers without language context).
_ENGLISH_ONLY_PATTERNS = (
    re.compile(r"[A-Z]{2,}"),  # ALL-CAPS acronyms
)

# Bantu noun-class / locative prefixes that commonly attach directly to
# English loanwords in code-switched speech. e.g. "ishopping" (i- class 9),
# "eshopping" (e- locative), "kuoffice" (ku- locative). Stripping these and
# re-checking the residue against the stoplist catches the dominant pattern
# of Southern African code-switching, where speakers prefix-attach rather
# than hyphenate. Order matters: longer prefixes first so "ama-" beats "a-".
_BANTU_LOANWORD_PREFIXES = ("ama", "izi", "aba", "umu", "ku", "i", "e", "u", "o")


@dataclass(frozen=True)
class CodeSwitchSpan:
    word: str
    start: int
    end: int
    reason: str  # "stoplist" | "acronym"


def detect_code_switches(text: str, language: str) -> list[CodeSwitchSpan]:
    """Return spans likely to be English insertions inside ``text``.

    Cheap, deterministic, no model required. Designed to be conservative —
    false negatives are fine (the rule-based G2P will mispronounce them and
    the lexicon expansion phase will catch the bug). False positives are
    worse because they route Bantu words through English G2P.
    """
    _ = language  # reserved for future per-language tweaks
    spans: list[CodeSwitchSpan] = []
    for match in re.finditer(r"\S+", text):
        word = match.group(0)
        lowered = word.strip(".,!?;:'\"").lower()
        # Three ways a Bantu speaker can introduce an English loanword:
        #   1. Bare token: "shopping" — direct check.
        #   2. Hyphenated compound: "siya-online" — split on - or _.
        #   3. Prefix-fused noun-class: "ishopping" / "eshopping" / "kuoffice"
        #      — strip a known Bantu prefix and re-check the residue.
        # Pattern 3 is the dominant Nguni form ("ngiyakwenza ishopping") and
        # the most important to catch.
        candidates: set[str] = {lowered, *re.split(r"[-_]", lowered)}
        for prefix in _BANTU_LOANWORD_PREFIXES:
            if lowered.startswith(prefix) and len(lowered) > len(prefix):
                candidates.add(lowered[len(prefix):])
        if candidates & ENGLISH_HINTS:
            spans.append(CodeSwitchSpan(word=word, start=match.start(), end=match.end(), reason="stoplist"))
            continue
        if any(p.search(word) for p in _ENGLISH_ONLY_PATTERNS):
            spans.append(CodeSwitchSpan(word=word, start=match.start(), end=match.end(), reason="acronym"))
    return spans
