"""Test di phase5_extract.py: esito task persistito, fallimenti, resume."""
from __future__ import annotations

import pytest

from pipeline.config import Settings
from pipeline.db import DB
from pipeline.ocr_clients import Region
from pipeline.phase5_extract import run
from pipeline.schema import ContractStrict


@pytest.fixture
def env(tmp_path):
    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.set_status("d", "reconciled")
    db.set_schema_status("d", "contract", "enumerated")
    db.save_page_ocr_a("d", 1, [
        Region(page_no=1, bbox=(0, 0, 100, 10), region_type="text",
               text="CONTRATTO N. 44/B", order_idx=0),
        Region(page_no=1, bbox=(0, 20, 100, 30), region_type="text",
               text="Data emissione: 2024-03-15", order_idx=1),
        Region(page_no=1, bbox=(0, 40, 100, 50), region_type="text",
               text="Importo: 1.234,50 EUR Valuta: EUR", order_idx=2),
        Region(page_no=1, bbox=(0, 60, 100, 70), region_type="text",
               text="Parte: ACME - buyer", order_idx=3),
    ], "CONTRATTO N. 44/B\nData emissione: 2024-03-15\n"
       "Importo: 1.234,50 EUR Valuta: EUR\nParte: ACME - buyer")
    yield db, s
    db.close()


def _payload():
    def leaf(value, quote):
        return {"value": value, "quote": quote, "page": 1,
                "bbox": [0, 0, 100, 10], "confidence": "high"}

    return {
        "contract_number": leaf("44/B", "CONTRATTO N. 44/B"),
        "issue_date": leaf("2024-03-15", "Data emissione: 2024-03-15"),
        "amount_eur": leaf(1234.50, "Importo: 1.234,50 EUR"),
        "currency": leaf("EUR", "Valuta: EUR"),
        "parties": [
            {"name": leaf("ACME", "ACME"), "role": leaf("buyer", "buyer")},
        ],
    }


def test_failed_task_blocks_and_is_recorded(env):
    """Un'estrazione fallita (server giù / JSON rotto) non avanza il documento:
    lo stato diventa needs_review e l'esito del task è persistito."""
    db, s = env

    class Flaky:
        def __init__(self):
            self.calls = 0

        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("extractor down")
            return _payload()

    flaky = Flaky()
    run("d", ContractStrict, db=db, settings=s, client=flaky)
    assert db.schema_status("d", "contract") == "needs_review"
    failed = db.failed_tasks("d")
    assert len(failed) == 1
    assert "extractor down" in failed[0]["error"]
    # nessuna estrazione scritta dal task fallito
    assert db.get_extractions("d") == []


def test_resume_retries_only_failed_tasks(env):
    """Al re-run da needs_review, i task ok sono saltati, i falliti ritentati."""
    db, s = env

    class FirstFailThenOk:
        def __init__(self):
            self.calls = 0

        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("down")
            return _payload()

    client = FirstFailThenOk()
    run("d", ContractStrict, db=db, settings=s, client=client)
    assert db.schema_status("d", "contract") == "needs_review"

    run("d", ContractStrict, db=db, settings=s, client=client)
    # il task è stato ritentato (2 chiamate totali) e ora è ok; lo stato resta
    # needs_review: la chiusura spetta alla Fase 6 (controlli di completezza)
    assert client.calls == 2
    assert db.failed_tasks("d") == []
    assert db.schema_status("d", "contract") == "needs_review"
    assert len(db.get_extractions("d", status="pending")) >= 6

    # la validazione sul percorso needs_review chiude il ciclo -> done
    from pipeline.phase6_validate import run as validate

    validate("d", ContractStrict, db=db, settings=s, client=client)
    assert db.schema_status("d", "contract") == "done"


def test_empty_response_is_failure_not_absence(env):
    """Una risposta {} (JSON valido ma vuota) è un fallimento del task, non
    'tutti i campi assenti'."""
    db, s = env

    class Empty:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            return {}

    run("d", ContractStrict, db=db, settings=s, client=Empty())
    assert db.schema_status("d", "contract") == "needs_review"
    assert len(db.failed_tasks("d")) >= 1


def test_partial_response_is_failure(env):
    """Una risposta parziale (solo alcuni campi del task) è un fallimento,
    non 'campi assenti': le omissioni non diventano null sintetici."""
    db, s = env

    class Partial:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            return {"contract_number": {
                "value": "44/B", "quote": "CONTRATTO N. 44/B",
                "page": 1, "bbox": [0, 0, 100, 10], "confidence": "high",
            }}

    run("d", ContractStrict, db=db, settings=s, client=Partial())
    assert db.schema_status("d", "contract") == "needs_review"
    failed = db.failed_tasks("d")
    assert failed and "campi omessi" in failed[0]["error"]
    # nessuna riga scritta dal task incompleto
    assert db.get_extractions("d") == []


def test_null_rows_persisted(env):
    """I null espliciti del modello producono righe pending con value NULL."""
    db, s = env

    class WithNull:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            payload = _payload()
            payload["parties"][0]["vat_id"] = {
                "value": None, "quote": None, "page": 1,
                "bbox": None, "confidence": "high",
            }
            return payload

    run("d", ContractStrict, db=db, settings=s, client=WithNull())
    vat = db.latest_extraction("d", "parties[0].vat_id")
    assert vat is not None
    assert vat["value_json"] is None
    assert vat["quote"] is None
    assert vat["status"] == "pending"


def test_missing_or_invalid_page_falls_back(env):
    """Regressione smoke: le regioni nel prompt non riportano la pagina, il
    modello può dare page=null o inventarla. Senza pagina valida il grounding
    di Fase 6 confronta con un testo vuoto e fallisce anche su quote corrette."""
    db, s = env

    class NoPage:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            payload = _payload()
            payload["contract_number"]["page"] = None
            payload["issue_date"]["page"] = 99  # pagina inesistente
            return payload

    run("d", ContractStrict, db=db, settings=s, client=NoPage())
    assert db.latest_extraction("d", "contract_number")["page_no"] == 1
    assert db.latest_extraction("d", "issue_date")["page_no"] == 1


def test_flat_fields_anchored_by_schema_labels(env):
    """Regressione smoke: i nomi Python dei campi (contract_number, amount_eur)
    non compaiono in un documento italiano; le etichette `anchors` dello schema
    sì. Senza ancora il task riceve l'intero documento (warning + contesto enorme
    sui PDF veri)."""
    from pipeline.phase5_extract import build_tasks

    db, s = env
    tasks = build_tasks("d", ContractStrict, db, s)
    flat = [t for t in tasks if t.item_field is None]
    assert [t.page_no for t in flat] == [1]
    assert set(flat[0].fields) == {"contract_number", "issue_date", "amount_eur", "currency"}


def test_field_anchors_fallback_to_name():
    from pipeline.schema import Party, field_anchors

    assert field_anchors(ContractStrict, "amount_eur")[0] == "importo"
    assert field_anchors(ContractStrict, "amount_eur")[-1] == "amount eur"
    assert field_anchors(Party, "vat_id") == ["vat id"]


# ---------------------------------------------------------------------------
# Contesto: le tabelle della pagina
# ---------------------------------------------------------------------------


def _doc_with_table(tmp_path):
    """Una pagina come quelle vere: la regola in prosa, il numero in tabella."""
    from pipeline.config import Settings
    from pipeline.db import DB

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "w")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/phb.pdf", "sha", 2)
    db.add_region("d", 1, (0, 0, 500, 20), "text", "DRAGONBORN TRAITS", "a", 0)
    db.add_region("d", 1, (0, 40, 500, 200), "table",
                  "<table><tr><td>Black</td><td>Acid</td></tr>"
                  "<tr><td>Blue</td><td>Lightning</td></tr></table>", "a", 1)
    db.add_region("d", 1, (0, 240, 500, 300), "text",
                  "Damage Resistance. You have resistance to the damage type "
                  "associated with your draconic ancestry.", "a", 2)
    db.set_page("d", 1, full_text="x")
    db.set_status("d", "reconciled")
    db.set_schema_status("d", "race_traits", "enumerated")
    return db, s


def test_item_context_includes_the_page_table(tmp_path):
    """Il numero della regola sta in tabella, non nel paragrafo.

    Nella run 5 Dragonborn/Damage Resistance è uscito `value=null`: la tabella
    Draconic Ancestry era la regione 721 della STESSA pagina del tratto, e il
    prompt conteneva solo la regione 728. Un'assenza dichiarata che i cancelli
    accettano e le note registrano come esclusione ragionevole — il dato si
    perde senza che niente lo segnali."""
    import json

    from pipeline.phase5_extract import build_prompt, build_tasks
    from pipeline.schema import RaceTraitsDoc

    db, s = _doc_with_table(tmp_path)
    db.upsert_extraction("d", "traits$inventory", json.dumps(
        [{"anchor": "Damage Resistance.", "page": 1, "region_ids": [3],
          "section": "DRAGONBORN TRAITS", "section_page": 1}]),
        None, None, None, 1, "validated", schema_name="race_traits")

    task = next(t for t in build_tasks("d", RaceTraitsDoc, db, s) if t.item_field)
    types = {r["type"] for r in task.regions}
    assert "table" in types, "la tabella della pagina non è nel contesto"
    prompt = build_prompt(task)
    assert "Lightning" in prompt and "rimanda a una tabella" in prompt
    db.close()


def test_page_tables_respect_the_context_budget(tmp_path):
    """Una tabella enorme (una lista di incantesimi) non deve mangiarsi il
    contesto: sopra il budget viene lasciata fuori, con una riga di log."""
    import json

    from pipeline.phase5_extract import build_tasks
    from pipeline.schema import RaceTraitsDoc

    db, s = _doc_with_table(tmp_path)
    # lontana dall'elemento: non entra dal margine di _select_by_ids, quindi
    # è il budget a doverla fermare
    db.add_region("d", 1, (0, 320, 500, 380), "text", "Languages. ...", "a", 3)
    db.add_region("d", 1, (0, 400, 500, 900), "table", "<table>" + "x" * 50_000, "a", 4)
    s.item_table_context_chars = 1000
    db.upsert_extraction("d", "traits$inventory", json.dumps(
        [{"anchor": "Damage Resistance.", "page": 1, "region_ids": [3],
          "section": "DRAGONBORN TRAITS", "section_page": 1}]),
        None, None, None, 1, "validated", schema_name="race_traits")

    task = next(t for t in build_tasks("d", RaceTraitsDoc, db, s) if t.item_field)
    tables = [r for r in task.regions if r["type"] == "table"]
    assert len(tables) == 1 and "Lightning" in tables[0]["text"]
    db.close()


def test_item_spans_to_the_next_anchor(tmp_path):
    """Con gli elementi a INTESTAZIONE l'elemento non è una regione sola.

    Nel Player's Handbook una capacità di classe è una riga 'RAGE' seguita dai
    paragrafi che la descrivono. Prendendo l'anchor più un margine restavano
    fuori il bonus ai danni, la resistenza e la durata — tutti i numeri della
    regola. Il confine non si indovina: l'inventario è ordinato, quindi ogni
    elemento finisce dove comincia il successivo."""
    import json

    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase5_extract import build_tasks
    from pipeline.schema import ClassFeaturesDoc

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "w")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/phb.pdf", "sha", 1)
    body = [
        "RAGE",
        "In battle, you fight with primal ferocity.",
        "You gain a bonus to the damage roll, as shown in the Barbarian table.",
        "You have resistance to bludgeoning, piercing, and slashing damage.",
        "UNARMORED DEFENSE",
        "Your Armor Class equals 10 + your Dexterity modifier.",
    ]
    for i, t in enumerate(body):
        db.add_region("d", 1, (0, i * 60, 500, i * 60 + 40), "text", t, "a", i)
    db.set_page("d", 1, full_text="\n".join(body))
    db.set_status("d", "reconciled")
    db.set_schema_status("d", "class_features", "enumerated")
    db.upsert_extraction("d", "features$inventory", json.dumps([
        {"anchor": "RAGE", "page": 1, "region_ids": [1], "section": None},
        {"anchor": "UNARMORED DEFENSE", "page": 1, "region_ids": [5], "section": None},
    ]), None, None, None, 1, "validated", schema_name="class_features")

    tasks = [t for t in build_tasks("d", ClassFeaturesDoc, db, s) if t.item_field]
    rage = next(t for t in tasks if t.anchor == "RAGE")
    text = " ".join(r["text"] or "" for r in rage.regions)
    assert "bonus to the damage roll" in text, "il corpo della regola è fuori contesto"
    assert "resistance to bludgeoning" in text
    assert "Armor Class equals" not in text, "lo span invade la capacità successiva"
    db.close()
