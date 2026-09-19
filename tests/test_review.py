"""Test di phase7_review.py: coda sugli ultimi tentativi e chiusura del ciclo."""
from __future__ import annotations

import json

import pytest

from pipeline.config import Settings
from pipeline.db import DB
from pipeline.phase7_review import (
    apply_correction,
    finalize_review,
    review_queue,
)
from pipeline.schema import ContractStrict


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # gold/corrections finisce in tmp, non nel repo
    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    d = DB(s)
    d.upsert_document("d", "/a.pdf", "sha", 1)
    d.set_status("d", "reconciled")
    d.set_schema_status("d", "contract", "needs_review")
    d.set_page("d", 1, full_text="Contratto n. 44/B del 2024")
    yield d
    d.close()


def _upsert(db, field, value, quote, attempt, status, confidence="high"):
    db.upsert_extraction("d", field, json.dumps(value), quote, 1, None,
                         attempt, status, confidence, schema_name="contract")


def test_queue_includes_rejected(db):
    """Un valore incoerente (rejected) deve stare in coda, non sparire."""
    _upsert(db, "contract_number", "44/B", "Contratto n. 44/B", 1, "validated")
    _upsert(db, "currency", "USD", "Valuta: EUR", 1, "rejected")
    _upsert(db, "issue_date", "2024-03-15", "Data emissione: 2024-03-15", 1,
            "needs_review")
    queue = review_queue("d", db=db)
    paths = {it["field_path"] for it in queue}
    assert paths == {"currency", "issue_date"}


def test_queue_uses_latest_attempt_only(db):
    """Una correzione (nuovo tentativo validated) sostituisce il vecchio in coda."""
    _upsert(db, "contract_number", "99/Z", "pippo", 1, "needs_review")
    _upsert(db, "contract_number", "44/B", "Contratto n. 44/B", 2, "validated")
    assert review_queue("d", db=db) == []


def test_correction_does_not_resurrect_old_attempt(db):
    _upsert(db, "contract_number", "99/Z", "pippo", 1, "needs_review")
    apply_correction("d", "contract_number", "44/B", db=db)
    row = db.latest_extraction("d", "contract_number")
    assert row["attempt"] == 2
    assert row["status"] == "validated"
    assert json.loads(row["value_json"]) == "44/B"
    # il vecchio tentativo esiste ancora ma non è in coda
    assert len(db.get_extractions("d")) == 2
    assert review_queue("d", db=db) == []


def test_finalize_rejects_zero_extractions(db):
    """needs_review senza estrazioni (es. risposta vuota marcata a monte):
    la finalizzazione non può portare a done."""
    assert not finalize_review("d", ContractStrict, db=db)
    assert db.schema_status("d", "contract") == "needs_review"


def test_finalize_rejects_pending(db):
    """Una riga ancora pending blocca la finalizzazione: niente assenze
    sintetizzate per lavoro non concluso."""
    _upsert(db, "contract_number", "44/B", "Contratto n. 44/B", 1, "validated")
    _upsert(db, "issue_date", "2024-03-15", "Data: 2024-03-15", 1, "pending")
    assert not finalize_review("d", ContractStrict, db=db)
    assert db.schema_status("d", "contract") == "needs_review"


def test_finalize_needs_full_document(db):
    """Coda vuota ma documento incompleto: niente done."""
    # solo un campo valido, gli altri mai estratti: lo schema strict su un
    # documento ricostruito con _fill_missing passa, ma qui simuliamo un campo
    # rejected ancora in coda
    _upsert(db, "contract_number", "44/B", "Contratto n. 44/B", 1, "validated")
    _upsert(db, "currency", "USD", "Valuta: EUR", 1, "rejected")
    assert not finalize_review("d", ContractStrict, db=db)
    assert db.schema_status("d", "contract") == "needs_review"


def test_finalize_goes_done(db):
    _upsert(db, "contract_number", "44/B", "Contratto n. 44/B", 1, "validated")
    _upsert(db, "issue_date", "2024-03-15", "Data emissione: 2024-03-15", 1, "validated")
    _upsert(db, "amount_eur", "1234.5", "Importo: 1.234,50 EUR", 1, "validated")
    _upsert(db, "currency", "EUR", "Valuta: EUR", 1, "validated")
    _upsert(db, "parties[0].name", "ACME", "ACME", 1, "validated")
    _upsert(db, "parties[0].role", "buyer", "buyer", 1, "validated")
    assert finalize_review("d", ContractStrict, db=db)
    assert db.schema_status("d", "contract") == "done"


def test_finalize_blocked_by_failed_tasks(db):
    """Un task di estrazione fallito blocca la finalizzazione (anche a coda vuota)."""
    db.record_task("d", "flat_p1_0", "failed", "connection refused", schema_name="contract")
    _upsert(db, "contract_number", "44/B", "Contratto n. 44/B", 1, "validated")
    assert not finalize_review("d", ContractStrict, db=db)
    assert db.schema_status("d", "contract") == "needs_review"


def test_finalize_after_corrections(db):
    """Il flusso completo: rejected -> correzione umana -> done."""
    _upsert(db, "contract_number", "44/B", "Contratto n. 44/B", 1, "validated")
    _upsert(db, "currency", "USD", "Valuta: EUR", 1, "rejected")
    _upsert(db, "issue_date", "2024-03-15", "Data emissione: 2024-03-15", 1, "validated")
    _upsert(db, "amount_eur", "1234.5", "Importo: 1.234,50 EUR", 1, "validated")
    _upsert(db, "parties[0].name", "ACME", "ACME", 1, "validated")
    _upsert(db, "parties[0].role", "buyer", "buyer", 1, "validated")
    assert len(review_queue("d", db=db)) == 1
    apply_correction("d", "currency", "EUR", db=db)
    assert review_queue("d", db=db) == []
    assert finalize_review("d", ContractStrict, db=db)
    assert db.schema_status("d", "contract") == "done"


def test_finalize_from_validated(db):
    db.set_status("d", "reconciled")
    db.set_schema_status("d", "contract", "validated")
    _upsert(db, "contract_number", "44/B", "Contratto n. 44/B", 1, "validated")
    _upsert(db, "issue_date", "2024-03-15", "Data emissione: 2024-03-15", 1, "validated")
    _upsert(db, "amount_eur", "1234.5", "Importo: 1.234,50 EUR", 1, "validated")
    _upsert(db, "currency", "EUR", "Valuta: EUR", 1, "validated")
    _upsert(db, "parties[0].name", "ACME", "ACME", 1, "validated")
    _upsert(db, "parties[0].role", "buyer", "buyer", 1, "validated")
    assert finalize_review("d", ContractStrict, db=db)
    assert db.schema_status("d", "contract") == "done"


def test_conflict_resolution_via_review(db):
    """Una correzione su un conflitto OCR aggiorna anche il testo canonico."""
    from pipeline.ocr_clients import Region

    db.save_page_ocr_a("d", 1, [
        Region(page_no=1, bbox=(0, 0, 10, 10), region_type="text",
               text="testo A", order_idx=0)
    ], "testo A")
    rid = db.get_regions("d", 1, engine="a")[0]["region_id"]
    db.add_conflict("d", rid, "testo A", "testo B", "testo A", "divergent_low")

    queue = review_queue("d", db=db)
    assert queue and queue[0]["field_path"] == f"region_conflict:{rid}"

    apply_correction("d", f"region_conflict:{rid}", "testo corretto", db=db)
    c = db.get_conflicts("d")[0]
    assert c["resolver"] == "human"
    assert c["resolved_text"] == "testo corretto"
    # il testo canonico segue la risoluzione umana
    assert db.get_page_text("d", 1) == "testo corretto"
    assert review_queue("d", db=db) == []


def test_conflict_correction_invalidates_dependents(db):
    """Una correzione OCR umana invalida le estrazioni validate sul testo
    vecchio: rientrano in coda, non restano validate col valore stantio."""
    from pipeline.ocr_clients import Region

    db.save_page_ocr_a("d", 1, [
        Region(page_no=1, bbox=(0, 0, 100, 10), region_type="text",
               text="Contratto 44/B", order_idx=0)
    ], "Contratto 44/B")
    rid = db.get_regions("d", 1, engine="a")[0]["region_id"]
    db.add_conflict("d", rid, "Contratto 44/B", "Contratto 99/Z",
                    "Contratto 44/B", "divergent_low")
    # valore già validato sul testo precedente (citazione dalla regione)
    _upsert(db, "contract_number", "44/B", "Contratto 44/B", 1, "validated")

    apply_correction("d", f"region_conflict:{rid}", "Contratto 99/Z", db=db)

    # canonico aggiornato...
    assert db.get_page_text("d", 1) == "Contratto 99/Z"
    # ...e il valore dipendente non resta validato: torna in coda
    row = db.latest_extraction("d", "contract_number")
    assert row["status"] == "needs_review"
    assert row["confidence"] == "low"
    queue = review_queue("d", db=db)
    assert [it["field_path"] for it in queue] == ["contract_number"]
