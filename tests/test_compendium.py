"""compendium.py: schema/compendium.schema.json e seed raceTraits.json."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")

from pipeline import compendium as c  # noqa: E402

SEED = json.loads((Path(__file__).resolve().parents[1] / "seeds" / "raceTraits.json").read_text())


def test_seed_is_valid_after_allof_merge():
    """Lo schema su disco esprime `.extend()` come allOf di due oggetti chiusi:
    senza la fusione nessun seed è valido (nemmeno quello curato a mano)."""
    assert c.validate_envelope(SEED, "raceTraits.json") == []


def test_merged_definition_still_closed_and_required():
    bad = {"version": 1, "definitions": [dict(SEED["definitions"][0], inventato=1)]}
    assert c.validate_envelope(bad, "raceTraits.json")
    missing = {"version": 1, "definitions": [
        {k: v for k, v in SEED["definitions"][0].items() if k != "raceName"}]}
    assert any("raceName" in e for e in c.validate_envelope(missing, "raceTraits.json"))


def test_decoding_schema_accepts_seed_effects_and_rejects_unknown_kind():
    from jsonschema import Draft202012Validator

    v = Draft202012Validator(c.decoding_schema("featureEffect"))
    assert all(v.is_valid(e) for d in SEED["definitions"] for e in d["effects"])
    assert not v.is_valid({"kind": "flying_speed", "amount": 30})
    # niente vincoli pesanti per la grammatica, niente riferimenti annidati
    dumped = json.dumps(c.decoding_schema("featureEffect"))
    assert '"pattern"' not in dumped and '"oneOf"' not in dumped
    assert "#/$defs/effects/" not in dumped


def test_race_trait_id_rule_matches_most_seed_ids():
    auto = [c.race_trait_id(d["raceName"], d.get("subraceName"), d["featureName"])
            for d in SEED["definitions"]]
    assert "race_genasi_fire_resistance" in auto  # token ripetuto non duplicato
    matches = sum(a == d["id"] for a, d in zip(auto, SEED["definitions"], strict=True))
    assert matches >= 31  # le eccezioni sono curate a mano: l'export riusa l'id del seed


def test_export_roundtrip_reuses_seed_ids():
    doc = {"traits": [
        {"raceName": {"value": d["raceName"]}, "subraceName": {"value": d.get("subraceName")},
         "featureName": {"value": d["featureName"]},
         "grantedAtLevel": {"value": d["grantedAtLevel"]}, "effects": {"value": d["effects"]}}
        for d in SEED["definitions"]]}
    envelope, notes = c.export_race_traits(doc, ["players_handbook"], SEED)
    assert c.validate_envelope(envelope, "raceTraits.json") == []
    assert [d["id"] for d in envelope["definitions"]] == [d["id"] for d in SEED["definitions"]]
    assert notes == []
