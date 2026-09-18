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
    db.set_status("d", "enumerated")
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
    assert db.get_status("d") == "needs_review"
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
    assert db.get_status("d") == "needs_review"

    run("d", ContractStrict, db=db, settings=s, client=client)
    # il task è stato ritentato (2 chiamate totali) e ora è ok; lo stato resta
    # needs_review: la chiusura spetta alla Fase 6 (controlli di completezza)
    assert client.calls == 2
    assert db.failed_tasks("d") == []
    assert db.get_status("d") == "needs_review"
    assert len(db.get_extractions("d", status="pending")) >= 6

    # la validazione sul percorso needs_review chiude il ciclo -> done
    from pipeline.phase6_validate import run as validate

    validate("d", ContractStrict, db=db, settings=s, client=client)
    assert db.get_status("d") == "done"


def test_empty_response_is_failure_not_absence(env):
    """Una risposta {} (JSON valido ma vuota) è un fallimento del task, non
    'tutti i campi assenti'."""
    db, s = env

    class Empty:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            return {}

    run("d", ContractStrict, db=db, settings=s, client=Empty())
    assert db.get_status("d") == "needs_review"
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
    assert db.get_status("d") == "needs_review"
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
