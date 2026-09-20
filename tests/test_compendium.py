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


def test_strip_defaults_makes_explicit_default_equal_to_omitted():
    """`condition: "always"` è il default dichiarato dallo schema: scriverla o
    ometterla è la stessa cosa. Alla run 5 il modello ha cominciato a scriverla
    sempre e sette effetti GIUSTI sono stati contati sbagliati, nascondendo che
    il tasso di errore era sceso invece di salire."""
    explicit = [{"kind": "resistance", "damageTypes": ["fire"], "condition": "always"}]
    implicit = [{"kind": "resistance", "damageTypes": ["fire"]}]
    assert c.strip_defaults(explicit) == implicit
    assert c.strip_defaults(implicit) == implicit
    # una condition VERA non si tocca: è l'informazione, non il default
    real = [{"kind": "speed_bonus", "amount": -5, "condition": "wearing_heavy_armor"}]
    assert c.strip_defaults(real) == real
    # kind sconosciuto: nessuna proprietà da togliere, nessun errore
    assert c.strip_defaults([{"kind": "boh", "condition": "always"}]) == \
        [{"kind": "boh", "condition": "always"}]


def test_export_does_not_write_redundant_condition():
    """Il seed curato non scrive mai `condition: always`: se l'export la scrive,
    ogni riestrazione segnala 'effects diversi dal seed' su tratti identici."""
    doc = {"traits": [{
        "raceName": {"value": "Tiefling"}, "subraceName": {"value": None},
        "featureName": {"value": "Hellish Resistance"},
        "grantedAtLevel": {"value": None},
        "effects": {"value": [{"kind": "resistance", "damageTypes": ["fire"],
                               "condition": "always"}]},
    }]}
    seed = {"version": 1, "definitions": [{
        "id": "race_tiefling_hellish_resistance", "raceName": "Tiefling",
        "featureName": "Hellish Resistance", "grantedAtLevel": 1,
        "sources": ["players_handbook"],
        "effects": [{"kind": "resistance", "damageTypes": ["fire"]}]}]}
    envelope, notes = c.export_race_traits(doc, ["players_handbook"], seed)
    assert envelope["definitions"][0]["effects"] == [
        {"kind": "resistance", "damageTypes": ["fire"]}]
    assert not [n for n in notes if "diversi dal seed" in n]


def test_effect_kinds_are_generated_from_the_schema():
    """L'elenco dei kind nel prompt va generato, non scritto a mano.

    Scritto a mano ne conteneva 13 su 25, e mancava `resource_cost`, che e' il
    kind piu' frequente fra le capacita' di classe curate. L'istruzione diceva
    di rispondere null quando nessun kind corrisponde, quindi il modello
    rispondeva null: 35 capacita' su 61 sbagliavano solo per questo."""
    kinds = dict(c.effect_kinds())
    assert len(kinds) == 25, f"attesi 25 kind, trovati {len(kinds)}"
    # i kind la cui assenza era misurabile nel gold
    for k in ("resource_cost", "extra_attack", "save_evasion",
              "ability_score_bonus", "ability_substitution", "damage_die"):
        assert k in kinds, k
    # ogni kind porta con se' i campi che lo distinguono
    assert "resourceId" in kinds["resource_cost"]
    assert "byLevel" in kinds["extra_attack"]
    assert "damageTypes" in kinds["resistance"]
    # `kind` e `condition` non sono campi da elencare: il primo e' il nome
    # stesso, il secondo ha una regola sua nel prompt
    assert all("kind" not in f and "condition" not in f for f in kinds.values())


def test_effect_kinds_text_lists_every_kind():
    txt = c.effect_kinds_text()
    for k in dict(c.effect_kinds()):
        assert k in txt, k
