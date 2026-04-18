import pytest

from frontend.codeswitch import detect_code_switches
from frontend.g2p import text_to_phonemes
from frontend.nemo_tokenizer import BantuPhonemeTokenizer
from frontend.normalizer import normalize_text
from frontend.registry import SUPPORTED_LANGUAGES, get_pack


# --- normalisation -----------------------------------------------------------

def test_normalize_expands_digits_in_target_language():
    # Phase B: digits expand to Bantu numerals, not English. "2" -> "mbiri" (Shona).
    normalized = normalize_text("ndine 2 mbudzi", language="shona")
    assert "mbiri" in normalized


def test_normalize_unsupported_language_rejected():
    with pytest.raises(ValueError):
        normalize_text("hello", language="klingon")


# --- G2P: prenasalised stops (Shona) ----------------------------------------

def test_g2p_basic_emits_phonemes():
    phonemes = text_to_phonemes("mhoro", language="shona")
    assert phonemes  # non-empty


def test_g2p_preserves_prenasalized_clusters():
    # "mb" should be a single prenasalised stop /ᵐb/, not split into m + b.
    phonemes = text_to_phonemes("mbira", language="shona")
    assert "ᵐb" in phonemes


def test_g2p_preserves_whistled_fricative_shona():
    phonemes = text_to_phonemes("zvakanaka", language="shona")
    assert "zʷ" in phonemes


# --- G2P: clicks (Nguni) ----------------------------------------------------

@pytest.mark.parametrize("language,word,expected_click", [
    ("zulu", "icala", "ǀ"),     # dental click
    ("zulu", "iqanda", "ǃ"),    # alveolar click
    ("zulu", "ixoxo", "ǁ"),     # lateral click
    ("xhosa", "iqaqa", "ǃ"),
    ("xhosa", "icici", "ǀ"),
    ("ndebele", "iqanda", "ǃ"),
])
def test_g2p_emits_click_consonants(language, word, expected_click):
    phonemes = text_to_phonemes(word, language=language)
    assert expected_click in phonemes, f"expected click {expected_click!r} in {language} {word!r} -> {phonemes!r}"


def test_g2p_aspirated_click_distinct_from_plain():
    # In Zulu, "xh" is aspirated lateral click ǁʰ, distinct from plain "x" -> ǁ.
    plain = text_to_phonemes("xoxa", language="zulu")
    aspirated = text_to_phonemes("xhosa", language="zulu")
    assert "ǁʰ" in aspirated
    assert "ǁʰ" not in plain


def test_g2p_nasal_breathy_click_trigraph_wins():
    # Trigraph "ngc" (breathy nasal dental click) must beat digraph "nc"/"ng".
    pack = get_pack("zulu")
    result = pack.g2p.phonemise("ngcono")
    assert "ŋǀʱ" in result.phonemes


# --- code-switching ---------------------------------------------------------

def test_codeswitch_detects_prefix_fused_loanword():
    # Authentic Zulu code-switching uses noun-class prefix fusion, not
    # hyphenation: "ngiyakwenza ishopping" (i- class 9), "eshopping" (e-
    # locative). The detector must strip the Bantu prefix before looking up
    # the English root.
    spans = detect_code_switches("ngiyakwenza ishopping namuhla", language="zulu")
    matched = [s.word.lower() for s in spans]
    assert any("ishopping" in w for w in matched), f"expected ishopping flagged in {matched}"


def test_codeswitch_detects_acronym():
    # 2+ consecutive uppercase letters trigger the acronym rule.
    spans = detect_code_switches("ndinoda USB nhasi", language="shona")
    assert any(s.reason == "acronym" for s in spans)


# --- tokenizer --------------------------------------------------------------

def test_tokenizer_round_trip_includes_clicks():
    tok = BantuPhonemeTokenizer()
    phonemes = text_to_phonemes("iqanda", language="zulu")
    ids = tok.encode(phonemes)
    decoded = tok.decode(ids)
    # The click token must survive a round trip — losing it during tokenisation
    # would silently corrupt training labels.
    assert "ǃ" in decoded


def test_tokenizer_vocab_covers_every_supported_language():
    tok = BantuPhonemeTokenizer()
    vocab = set(tok.vocab)
    for language in SUPPORTED_LANGUAGES:
        for ipa in get_pack(language).inventory.all_phonemes():
            assert ipa in vocab, f"phoneme {ipa!r} from {language} missing from tokenizer vocab"


def test_tokenizer_pad_id_is_zero():
    # NeMo's collator assumes pad_id=0; locking this in prevents silent breakage
    # if special-token order ever changes.
    tok = BantuPhonemeTokenizer()
    assert tok.pad_id == 0
