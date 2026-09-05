"""Test di text_norm.py."""
from pipeline.text_norm import normalize, similarity


def test_normalize_basic():
    assert normalize("  Hello,  World!!  ") == "hello world"


def test_normalize_diacritics():
    assert normalize("Caffè") == "caffe"


def test_normalize_none():
    assert normalize(None) == ""


def test_similarity_identical():
    assert similarity("ciao", "ciao") == 1.0


def test_similarity_different():
    assert similarity("ciao", "mondo") < 0.5
