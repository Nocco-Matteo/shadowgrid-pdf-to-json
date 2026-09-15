"""Test dello schema (Extracted[T], loose, flatten)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline.schema import ContractStrict, Extracted, flatten_extracted, loose


def test_extracted_both_none_ok():
    e = Extracted[str](value=None, quote=None)
    assert e.value is None and e.quote is None
    assert e.confidence == "high"


def test_extracted_both_present_ok():
    e = Extracted[str](value="abc", quote="abc")
    assert e.value == "abc"


def test_extracted_value_without_quote_raises():
    with pytest.raises(ValidationError):
        Extracted[str](value="abc", quote=None)


def test_extracted_quote_without_value_raises():
    with pytest.raises(ValidationError):
        Extracted[str](value=None, quote="abc")


def test_loose_has_no_constraints():
    L = loose(ContractStrict)
    # tutti i campi Optional -> istanza vuota valida
    instance = L()
    assert instance.model_dump() == {k: None for k in ContractStrict.model_fields}


def test_loose_name():
    L = loose(ContractStrict)
    assert L.__name__.endswith("Loose")


def test_flatten_extracted_simple():
    e = Extracted[str](value="42", quote="42", page=1, bbox=(0, 0, 10, 10))
    rows = flatten_extracted(e)
    assert len(rows) == 1
    assert rows[0]["field_path"] == "<root>"
    assert rows[0]["quote"] == "42"
    assert rows[0]["page"] == 1


def test_flatten_extracted_nested():
    from pipeline.schema import Party

    p = Party(
        name=Extracted[str](value="ACME", quote="ACME"),
        role=Extracted[str](value="buyer", quote="buyer"),
        vat_id=Extracted(value=None, quote=None),
    )
    rows = flatten_extracted(p)
    paths = [r["field_path"] for r in rows]
    assert "name" in paths
    assert "role" in paths
    # il null esplicito produce una riga (assenza DICHIARATA), con
    # value_json=None: distinta dal campo mai prodotto
    assert "vat_id" in paths
    vat_row = next(r for r in rows if r["field_path"] == "vat_id")
    assert vat_row["value_json"] is None
    assert vat_row["quote"] is None


def test_flatten_omitted_field_no_row():
    # campo mai presente nel payload -> nessuna riga (vs null esplicito)
    rows = flatten_extracted({"name": {"value": "ACME", "quote": "ACME"}})
    assert [r["field_path"] for r in rows] == ["name"]
