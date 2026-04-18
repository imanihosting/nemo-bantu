"""NeMo-compatible phoneme tokenizer for Bantu languages.

Wraps a :class:`PhonemeInventory` to produce stable integer IDs over the
union of all language phoneme sets. Used by the Phase C training run
(``configs/training/fastpitch_shona_phoneme.yaml``) instead of NeMo's
``BaseCharsTokenizer``.

The tokenizer is intentionally NeMo-shaped (``encode``/``decode``/``vocab``)
so it can be plugged into ``FastPitchModel`` config blocks as a
``_target_`` import, but it does not subclass NeMo's tokenizer base classes
to keep this module importable without ``nemo_toolkit`` installed.
"""

from frontend.base.phonemes import PAD, SILENCE, UNK, WORD_BOUNDARY
from frontend.registry import SUPPORTED_LANGUAGES, get_pack


_SPECIAL_TOKENS = (PAD, UNK, SILENCE, WORD_BOUNDARY, " ")


def _build_vocab(languages: tuple[str, ...]) -> list[str]:
    """Stable, sorted vocabulary across all requested languages.

    Special tokens always occupy the lowest IDs so PAD is 0; this matters for
    NeMo's collator which assumes pad_id is fixed across runs.
    """
    phonemes: set[str] = set()
    for lang in languages:
        phonemes.update(get_pack(lang).inventory.all_phonemes())
    return list(_SPECIAL_TOKENS) + sorted(phonemes)


class BantuPhonemeTokenizer:
    def __init__(self, languages: tuple[str, ...] = tuple(sorted(SUPPORTED_LANGUAGES))) -> None:
        self.languages = languages
        self._vocab = _build_vocab(languages)
        self._token_to_id = {tok: i for i, tok in enumerate(self._vocab)}
        self.pad_id = self._token_to_id[PAD]
        self.unk_id = self._token_to_id[UNK]

    @property
    def vocab(self) -> list[str]:
        return list(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode(self, phoneme_string: str) -> list[int]:
        """Encode a space-separated phoneme string into IDs.

        Unknown phonemes map to ``unk_id`` rather than raising — training
        will surface them via the validation harness.
        """
        ids: list[int] = []
        for tok in phoneme_string.split(" "):
            if not tok:
                continue
            ids.append(self._token_to_id.get(tok, self.unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._vocab[i] for i in ids if 0 <= i < len(self._vocab))

    def __call__(self, phoneme_string: str) -> list[int]:
        return self.encode(phoneme_string)
