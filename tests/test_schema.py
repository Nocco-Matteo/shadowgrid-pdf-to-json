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


# ---------------------------------------------------------------------------
# guided_schema: schema per task con tutti i campi obbligatori
# ---------------------------------------------------------------------------


def test_guided_schema_flat_task_requires_only_its_fields():
    from pipeline.schema import guided_schema

    g = guided_schema(ContractStrict, ["contract_number", "currency"])
    assert set(g["properties"]) == {"contract_number", "currency"}
    assert set(g["required"]) == {"contract_number", "currency"}
    # il campo foglia è l'oggetto Extracted, non null
    assert g["properties"]["contract_number"] == {"$ref": "#/$defs/Extracted_str_"}
    leaf = g["$defs"]["Extracted_str_"]
    # confidence obbligatoria: omessa varrebbe il default "high", cioè la
    # risposta più rassicurante senza che nessuno l'abbia scelta
    assert leaf["required"] == ["value", "quote", "confidence"]
    assert all("default" not in p for p in leaf["properties"].values())


def test_guided_schema_keeps_value_types():
    """Regressione smoke: con Extracted[Any] il modello dava amount_eur come
    stringa "1.234,50" e lo schema strict finale falliva."""
    from pipeline.schema import guided_schema

    g = guided_schema(ContractStrict, ["amount_eur", "currency"])
    types = lambda ref: g["$defs"][ref.split("/")[-1]]["properties"]["value"]  # noqa: E731
    amount = types(g["properties"]["amount_eur"]["$ref"])
    assert {"type": "number"} in amount["anyOf"]
    currency = types(g["properties"]["currency"]["$ref"])
    assert any(o.get("enum") == ["EUR", "USD", "GBP"] for o in currency["anyOf"])


def test_guided_schema_list_item_task():
    from pipeline.schema import guided_schema

    g = guided_schema(ContractStrict, ["parties"], single_item=True)
    assert g["required"] == ["parties"]
    parties = g["properties"]["parties"]
    assert parties["type"] == "array" and parties["minItems"] == parties["maxItems"] == 1
    party = g["$defs"]["Party"]
    assert set(party["required"]) == {"name", "role", "vat_id"}
    assert party["properties"]["vat_id"] == {"$ref": "#/$defs/Extracted_Union_str__NoneType__"}


def test_guided_schema_accepts_declared_absence_and_rejects_omission():
    jsonschema = pytest.importorskip("jsonschema")
    from pipeline.schema import guided_schema

    g = guided_schema(ContractStrict, ["parties"], single_item=True)
    leaf = {"value": "x", "quote": "x", "page": 1, "bbox": None, "confidence": "high"}
    absent = {"value": None, "quote": None, "confidence": "high"}
    jsonschema.validate({"parties": [{"name": leaf, "role": leaf, "vat_id": absent}]}, g)
    with pytest.raises(jsonschema.ValidationError):  # vat_id omesso
        jsonschema.validate({"parties": [{"name": leaf, "role": leaf}]}, g)
    with pytest.raises(jsonschema.ValidationError):  # confidence omessa
        jsonschema.validate(
            {"parties": [{"name": leaf, "role": leaf,
                          "vat_id": {"value": None, "quote": None}}]}, g)
    with pytest.raises(jsonschema.ValidationError):  # campo intero a null
        jsonschema.validate({"parties": [{"name": None, "role": leaf, "vat_id": absent}]}, g)


def test_loose_unchanged_by_guided_schema():
    from pipeline.schema import guided_schema

    guided_schema(ContractStrict, ["parties"], single_item=True)
    assert loose(ContractStrict)().model_dump() == {k: None for k in ContractStrict.model_fields}


# ---------------------------------------------------------------------------
# La famiglia degli envelope "effetto"
# ---------------------------------------------------------------------------


def test_root_field_names_are_unique_across_schemas():
    """Le estrazioni di più envelope convivono nello stesso documento e la
    chiave è (doc_id, field_path, attempt): due schemi con lo stesso campo
    radice si sovrascriverebbero a vicenda."""
    from pipeline.schema import SCHEMAS

    seen: dict[str, str] = {}
    for name, model in SCHEMAS.items():
        for field in model.model_fields:
            assert field not in seen, f"{name} e {seen[field]} condividono '{field}'"
            seen[field] = name


def test_effect_family_shares_one_shape():
    """I sei envelope "effetto" del compendium differiscono SOLO per il campo
    che dice di chi è la regola: la base è `featureDefinitionBase`."""
    from pipeline.schema import SCHEMAS

    owners = {
        "race_traits": ("raceName", "subraceName"),
        "class_features": ("className", "subclassName"),
        "feat_effects": ("featName", None),
        "spell_effects": ("spellName", None),
        "item_features": ("itemName", None),
        "background_features": ("backgroundName", None),
    }
    shared = {"featureName", "grantedAtLevel", "effects"}
    for name, (owner, sub) in owners.items():
        doc = SCHEMAS[name]
        (list_field,) = doc.model_fields
        item = doc.model_fields[list_field].annotation.__args__[0]
        expected = shared | {owner} | ({sub} if sub else set())
        assert set(item.model_fields) == expected, name
        assert doc.seed_file.endswith(".json")


def test_every_effect_envelope_builds_a_guided_schema():
    """Lo schema guidato deve costruirsi per tutti: un envelope che non si
    decodifica non è estraibile."""
    from pipeline.schema import SCHEMAS, guided_schema

    for name, doc in SCHEMAS.items():
        if not getattr(doc, "seed_file", None):
            continue
        (list_field,) = doc.model_fields
        g = guided_schema(doc, [list_field], single_item=True)
        assert g["properties"][list_field]["maxItems"] == 1, name
        # gli effetti restano vincolati alla tassonomia chiusa del compendium
        assert any(k.startswith("cmp__effects__") for k in g["$defs"]), name


def test_effect_validator_runs_on_every_envelope():
    """Il controllo contro la tassonomia non è privilegio dei tratti razziali."""
    from pydantic import ValidationError

    from pipeline.schema import SCHEMAS

    for name in ("class_features", "feat_effects", "item_features"):
        doc = SCHEMAS[name]
        (list_field,) = doc.model_fields
        item = doc.model_fields[list_field].annotation.__args__[0]
        owner = next(f for f in item.model_fields if f.endswith("Name")
                     and f not in ("featureName",))
        good = {"value": "X", "quote": "X", "page": 1, "bbox": None, "confidence": "high"}
        with pytest.raises(ValidationError):
            item(**{owner: good, "featureName": good,
                    "effects": {"value": [{"kind": "inesistente"}], "quote": "q",
                                "page": 1, "bbox": None, "confidence": "high"}})
