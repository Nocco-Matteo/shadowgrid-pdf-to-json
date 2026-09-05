"""Test di phase6_validate.py: cancelli grounding, coerenza, localizzazione."""
from __future__ import annotations

from pipeline.phase6_validate import (
    gate_grounding,
    gate_value_quote,
    localize_bbox,
)


def test_grounding_ok():
    assert gate_grounding("Contratto n. 44/B", "Il Contratto n. 44/B del 2024", 90)


def test_grounding_fail_hallucinated():
    assert not gate_grounding("pippo pluto", "Il Contratto n. 44/B del 2024", 90)


def test_grounding_empty_quote():
    assert not gate_grounding("", "testo", 90)


def test_value_quote_number_ok():
    assert gate_value_quote(44, "Contratto n. 44/B")


def test_value_quote_number_wrong():
    assert not gate_value_quote(45, "Contratto n. 44/B")


def test_value_quote_float_comma():
    assert gate_value_quote(1234.5, "Importo: 1.234,50 euro")


def test_value_quote_string_substring():
    assert gate_value_quote("ACME", "Società ACME S.r.l.")


def test_value_quote_string_wrong():
    assert not gate_value_quote("XYZ", "Società ACME S.r.l.")


def test_value_quote_both_none():
    assert gate_value_quote(None, None)


def test_value_quote_enum_via_token_overlap():
    # enum mappata: la quote contiene il termine che mappa al valore
    assert gate_value_quote("EUR", "Importo in euro")


def test_localize_bbox_found():
    regions = [
        {"text": "preambolo", "bbox": "[0,0,10,10]"},
        {"text": "Contratto n. 44/B", "bbox": "[10,10,100,30]"},
    ]
    bbox = localize_bbox("Contratto n. 44/B", regions)
    assert bbox == (10, 10, 100, 30)


def test_localize_bbox_not_found():
    regions = [{"text": "altro", "bbox": "[0,0,10,10]"}]
    assert localize_bbox("pippo", regions) is None
