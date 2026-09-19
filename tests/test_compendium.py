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


# ---------------------------------------------------------------------------
# Regole del loader: documentate nelle description, non codificate nello schema
# ---------------------------------------------------------------------------


def test_effect_needs_exactly_one_of_amount_or_alternative():
    """`{"kind": "save_bonus", "condition": "always"}` senza amount è uscito
    validato dalla prima run ed è finito nell'export di Dwarven Resilience: il
    cancello 6.2 verifica i numeri presenti, e lì non ce n'erano. Un effetto
    che il loader del compendium rifiuterebbe non deve uscire dalla pipeline."""
    assert c.validate_def({"kind": "save_bonus", "condition": "always"}, "featureEffect")
    assert c.validate_def({"kind": "save_bonus", "amount": 2,
                           "fromAbilityModifier": "con"}, "featureEffect")
    assert c.validate_def({"kind": "save_bonus", "amount": 2}, "featureEffect") == []
    assert c.validate_def({"kind": "save_bonus", "fromAbilityModifier": "con"},
                          "featureEffect") == []
    # amount presente soddisfa il vincolo (lo zero ha una regola sua:
    # v. test_zero_delta_is_absence_not_effect)
    assert c.validate_def({"kind": "speed_bonus", "amount": -5}, "featureEffect") == []


def test_damage_type_must_be_one_of_the_closed_set():
    """La prima run ha esportato `damageTypes: ['associated with your draconic
    ancestry']`: prosa della regola in un campo valore. Lo schema definisce
    damageTypeClosed ma resistance accetta ancora stringhe libere."""
    assert c.validate_def(
        {"kind": "resistance", "damageTypes": ["associated with your draconic ancestry"]},
        "featureEffect")
    assert c.validate_def({"kind": "resistance", "damageTypes": ["fire"]},
                          "featureEffect") == []
    assert c.validate_def({"kind": "resistance", "damageTypes": ["Poison"]},
                          "featureEffect") == []
    assert c.validate_def({"kind": "resistance", "damageTypes": ["fire", "the"]},
                          "featureEffect")


def test_loader_rules_reach_nested_effects_of_an_envelope():
    """Un envelope va controllato fino agli effects annidati, non solo in cima."""
    bad = {"version": 1, "definitions": [{
        "id": "race_x_y", "raceName": "X", "featureName": "Y", "grantedAtLevel": 1,
        "sources": ["players_handbook"],
        "effects": [{"kind": "resistance", "damageTypes": ["whatever the text says"]}],
    }]}
    errors = c.validate_envelope(bad, "raceTraits.json")
    assert any("damage" in e for e in errors)


def test_curated_seed_still_passes_the_loader_rules():
    """Le regole nuove non devono bocciare il seed curato a mano."""
    assert c.validate_envelope(SEED, "raceTraits.json") == []


def test_zero_delta_is_absence_not_effect():
    """`speed_bonus: 0` è quello che resta quando una razza va a 30 feet come
    tutti: nessun cambiamento. Alla run 3 l'elfo l'ha prodotto mentre umano,
    dragonide, mezzelfo, mezzorco e tiefling — stessa velocità, stessa frase —
    non hanno prodotto effetti: incoerenza del modello, non delle razze."""
    assert c.validate_def({"kind": "speed_bonus", "amount": 0}, "featureEffect")
    assert c.validate_def({"kind": "speed_bonus", "amount": -5}, "featureEffect") == []
    assert c.validate_def({"kind": "ac_bonus", "amount": 0}, "featureEffect")
    # una grandezza diversa da amount non è toccata dalla regola
    assert c.validate_def({"kind": "save_bonus", "fromAbilityModifier": "con"},
                          "featureEffect") == []
