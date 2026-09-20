"""Schema race_traits (compendium raceTraits.json): Fase 4 -> 5 -> 6 -> export,
con estrattore stub e testo in stile Player's Handbook."""
from __future__ import annotations

import json
import re

import pytest

from pipeline import phase4_enumerate, phase5_extract, phase6_validate
from pipeline.cli import export_doc
from pipeline.compendium import validate_envelope
from pipeline.config import Settings
from pipeline.db import DB
from pipeline.ocr_clients import Region
from pipeline.schema import RaceTraitsDoc

PAGES = {
    1: ["Dwarf Traits",
        "Your dwarf character has an assortment of inborn abilities.",
        "Darkvision. Accustomed to life underground, you have superior vision in dark and dim conditions.",
        "Dwarven Resilience. You have advantage on saving throws against poison, "
        "and you have resistance against poison damage."],
    2: ["Hill Dwarf",
        "As a hill dwarf, you have keen senses.",
        "Dwarven Toughness. Your hit point maximum increases by 1, and it increases "
        "by 1 every time you gain a level."],
}


@pytest.fixture
def env(tmp_path):
    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/phb.pdf", "sha", 2)
    for page_no, texts in PAGES.items():
        regions = [Region(page_no=page_no, bbox=(0, i * 40, 1000, i * 40 + 30),
                          region_type="text", text=t, order_idx=i)
                   for i, t in enumerate(texts)]
        db.save_page_ocr_a("d", page_no, regions, "\n".join(texts))
    db.set_status("d", "reconciled")
    yield db, s
    db.close()


def leaf(value, quote, page):
    return {"value": value, "quote": quote, "page": page, "bbox": None, "confidence": "high"}


NULL = {"value": None, "quote": None, "page": None, "bbox": None, "confidence": "high"}


class Stub:
    """Risponde all'inventario (Fase 4) e ai task per elemento (Fase 5)."""

    def __init__(self, db):
        self.region_id = {r["text"].split(".")[0]: r["region_id"]
                          for pn in PAGES for r in db.get_regions("d", pn, engine="a")}
        self.prompts: list[str] = []
        self.schemas: list[dict] = []

    def extract(self, prompt, images_b64=None, guided_json_schema=None):
        self.prompts.append(prompt)
        self.schemas.append(guided_json_schema)
        if prompt.startswith("Elenca SOLO"):
            return {"items": [
                {"anchor": "Darkvision", "page": 1, "region_ids": [self.region_id["Darkvision"]],
                 "section": "Dwarf Traits"},
                {"anchor": "Dwarven Resilience", "page": 1,
                 "region_ids": [self.region_id["Dwarven Resilience"]], "section": "Dwarf Traits"},
                {"anchor": "Dwarven Toughness", "page": 2,
                 "region_ids": [self.region_id["Dwarven Toughness"]], "section": "Hill Dwarf"},
            ]}
        item = re.search(r"^ELEMENTO: (.+)$", prompt, re.M).group(1)
        if item == "Darkvision":
            trait = {"raceName": leaf("Dwarf", "Dwarf Traits", 1), "subraceName": NULL,
                     "featureName": leaf("Darkvision", "Darkvision", 1),
                     "grantedAtLevel": NULL, "effects": NULL}
        elif item == "Dwarven Resilience":
            trait = {"raceName": leaf("Dwarf", "Dwarf Traits", 1), "subraceName": NULL,
                     "featureName": leaf("Dwarven Resilience", "Dwarven Resilience", 1),
                     "grantedAtLevel": NULL,
                     "effects": leaf([{"kind": "resistance", "damageTypes": ["poison"]}],
                                     "you have resistance against poison damage", 1)}
        else:
            trait = {"raceName": leaf("Dwarf", "Hill Dwarf", 2),
                     "subraceName": leaf("Hill", "Hill Dwarf", 2),
                     "featureName": leaf("Dwarven Toughness", "Dwarven Toughness", 2),
                     "grantedAtLevel": NULL,
                     "effects": leaf([{"kind": "hp_bonus_per_level", "amount": 1}],
                                     "Your hit point maximum increases by 1", 2)}
        return {"traits": [trait]}


def test_race_traits_end_to_end(env, tmp_path):
    db, s = env
    stub = Stub(db)
    desc = RaceTraitsDoc.model_fields["traits"].description
    phase4_enumerate.run("d", "traits", "race_traits", db=db, settings=s, client=stub, description=desc)
    inv = json.loads(db.latest_extraction("d", "traits$inventory")["value_json"])
    assert [(it["anchor"], it["section"], it["section_page"]) for it in inv] == [
        ("Darkvision", "Dwarf Traits", 1), ("Dwarven Resilience", "Dwarf Traits", 1),
        ("Dwarven Toughness", "Hill Dwarf", 2)]
    assert "OGNI tratto che cambia qualcosa che la scheda calcola" in stub.prompts[0]

    phase5_extract.run("d", RaceTraitsDoc, db=db, settings=s, client=stub)
    item_prompt = next(p for p in stub.prompts if "ELEMENTO: Dwarven Toughness" in p)
    # il titolo della sottorazza è nel contesto, con la pagina; campi descritti
    assert "p.2 text: Hill Dwarf" in item_prompt and "SEZIONE: Hill Dwarf" in item_prompt
    assert "- effects: effetti della regola nella tassonomia" in item_prompt
    # lo schema guidato vincola effects alla tassonomia del compendium
    assert any(k.startswith("cmp__effects__resistance") for k in stub.schemas[-1]["$defs"])

    phase6_validate.run("d", RaceTraitsDoc, db=db, settings=s, client=stub)
    assert db.schema_status("d", "race_traits") == "validated"

    out = export_doc("d", RaceTraitsDoc, db, s, ["players_handbook"])
    envelope = json.loads(out.read_text())
    assert validate_envelope(envelope, "raceTraits.json") == []
    by_id = {d["id"]: d for d in envelope["definitions"]}
    # envelope AUTONOMO dal manuale: le sources sono solo quelle del libro
    # processato, non quelle ereditate da un seed curato altrove
    assert set(by_id) == {"race_dwarf_dwarven_resilience", "race_dwarf_hill_dwarven_toughness"}
    assert by_id["race_dwarf_dwarven_resilience"]["sources"] == ["players_handbook"]
    assert by_id["race_dwarf_hill_dwarven_toughness"]["subraceName"] == "Hill"
    assert by_id["race_dwarf_hill_dwarven_toughness"]["grantedAtLevel"] == 1
    notes = (out.parent / "raceTraits.json.notes.txt").read_text()
    assert "Darkvision" in notes


def test_invalid_effect_fails_strict_schema(env):
    db, s = env

    class BadEffect(Stub):
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            out = super().extract(prompt, images_b64, guided_json_schema)
            if "traits" in out and out["traits"][0]["effects"]["value"]:
                out["traits"][0]["effects"]["value"] = [{"kind": "resistance"}]  # damageTypes mancante
            return out

    stub = BadEffect(db)
    phase4_enumerate.run("d", "traits", "race_traits", db=db, settings=s, client=stub)
    phase5_extract.run("d", RaceTraitsDoc, db=db, settings=s, client=stub)
    phase6_validate.run("d", RaceTraitsDoc, db=db, settings=s, client=stub)
    assert db.schema_status("d", "race_traits") == "needs_review"


def test_invalid_effect_never_reaches_the_export(env):
    """Il cancello 6.3 è PER CAMPO: un effetto fuori tassonomia non arriva a
    `validated`, quindi l'export (che filtra su `validated`) non lo vede.

    Prima il controllo era solo sul documento intero: bocciava il documento ma
    lasciava le estrazioni validate, e l'export le prendeva lo stesso. Nella
    run 2 sono finiti nel seed 3 effetti invalidi con il log che dichiarava il
    file non valido."""
    db, s = env

    class BadEffect(Stub):
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            out = super().extract(prompt, images_b64, guided_json_schema)
            if "traits" in out and out["traits"][0]["effects"]["value"]:
                out["traits"][0]["effects"]["value"] = [
                    {"kind": "save_bonus", "condition": "always"}]  # né amount né ability
            return out

    stub = BadEffect(db)
    phase4_enumerate.run("d", "traits", "race_traits", db=db, settings=s, client=stub)
    phase5_extract.run("d", RaceTraitsDoc, db=db, settings=s, client=stub)
    phase6_validate.run("d", RaceTraitsDoc, db=db, settings=s, client=stub)

    bad = [r for r in db.get_extractions("d")
           if r["field_path"].endswith(".effects") and r["value_json"]
           and "save_bonus" in r["value_json"]]
    assert bad and all(r["status"] != "validated" for r in bad)

    out = export_doc("d", RaceTraitsDoc, db, s, ["players_handbook"])
    assert out.name == "raceTraits.json", "envelope invalido scritto col nome buono"
    assert validate_envelope(json.loads(out.read_text()), "raceTraits.json") == []


def test_page_windows_overlap_and_budget():
    from pipeline.phase4_enumerate import _page_windows

    blocks = [(n, "x" * 100) for n in range(1, 8)]
    windows = _page_windows(blocks, 320)
    assert [w[0] for w in windows] == [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
    assert all(len(text) <= 320 for _, text in windows)
    # pagina più lunga del budget: finestra da sola, nessun ciclo infinito
    assert [w[0] for w in _page_windows([(1, "x" * 900), (2, "y")], 320)] == [[1], [2]]


def test_enumerate_keeps_same_name_on_different_pages(env):
    """Tratti omonimi di razze diverse (es. 'Natural Armor') non si fondono."""
    db, s = env

    class Twice:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            return {"items": [
                {"anchor": "Dwarven Resilience", "page": 1, "region_ids": [], "section": None},
                {"anchor": "Dwarven Toughness", "page": 2, "region_ids": [], "section": None},
                {"anchor": "Dwarven Resilience", "page": 1, "region_ids": [], "section": None},
            ]}

    items = phase4_enumerate.run("d", "traits", "race_traits", db=db, settings=s, client=Twice())
    assert [(i.anchor, i.page) for i in items] == [("Dwarven Resilience", 1), ("Dwarven Toughness", 2)]


def test_reset_extraction_keeps_ocr(env):
    """Documento già estratto con un altro schema: si riparte da reconciled
    senza rifare l'OCR (e senza righe del vecchio schema nel documento)."""
    db, s = env
    db.upsert_extraction("d", "contract_number", '"44/B"', "x", 1, None, 1, "needs_review", schema_name="race_traits")
    db.record_task("d", "flat_p1_0", "ok", schema_name="race_traits")
    db.set_status("d", "reconciled")
    db.set_schema_status("d", "race_traits", "needs_review")
    db.reset_extraction("d")
    assert db.get_status("d") == "reconciled"
    assert db.get_extractions("d") == [] and db.task_status("d", "flat_p1_0") is None
    assert len(db.get_regions("d", 1, engine="a")) == len(PAGES[1])


def test_reset_keeps_the_inventory_when_asked(env):
    """Rifare la Fase 5 dopo un guasto non deve ripagare l'enumerazione.

    Azzerando l'envelope per ripulire 177 task falliti se n'e' andato anche
    l'inventario di 445 capacita', che erano quindici minuti di GPU."""
    db, s = env
    db.upsert_extraction("d", "traits$inventory", '[{"anchor": "X", "page": 1}]',
                         None, None, None, 1, "validated", schema_name="race_traits")
    db.upsert_extraction("d", "traits[0].featureName", '"X"', "X", 1, None, 1,
                         "validated", schema_name="race_traits")
    db.record_task("d", "enumerate:traits:pagine 1-2", "ok", schema_name="race_traits")
    db.record_task("d", "traits[0]", "failed", "guasto", schema_name="race_traits")
    db.set_schema_status("d", "race_traits", "needs_review")

    db.reset_extraction("d", schema_name="race_traits", keep_inventory=True)

    left = {r["field_path"] for r in db.get_extractions("d", schema_name="race_traits")}
    assert left == {"traits$inventory"}, "l'inventario non e' sopravvissuto"
    assert db.task_status("d", "enumerate:traits:pagine 1-2") == "ok"
    assert db.task_status("d", "traits[0]") is None, "il task fallito doveva sparire"

    # senza il flag si azzera tutto, come prima
    db.reset_extraction("d", schema_name="race_traits")
    assert db.get_extractions("d", schema_name="race_traits") == []
