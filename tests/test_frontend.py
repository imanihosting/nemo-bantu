from frontend.normalizer import normalize_text
from frontend.g2p import text_to_phonemes


def test_normalize_digits():
    normalized = normalize_text("I have 2 cows", language="shona")
    assert "two" in normalized


def test_g2p_basic():
    phonemes = text_to_phonemes("mhoro", language="shona")
    assert len(phonemes) > 0


def test_g2p_preserves_prenasalized_clusters():
    phonemes = text_to_phonemes("mbira", language="shona")
    assert "mb" in phonemes
