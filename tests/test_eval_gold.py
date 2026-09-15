"""Test di phase8_eval.py: valutazione su gold stub."""
from __future__ import annotations

import json

import pytest

from pipeline.config import Settings
from pipeline.db import DB
from pipeline.phase8_eval import evaluate, print_metrics


@pytest.fixture
def gold(tmp_path):
    g = tmp_path / "gold"
    (g / "annotations").mkdir(parents=True)
    (g / "sealed.txt").write_text("doc_a\n")
    (g / "annotations" / "doc_a.json").write_text(json.dumps({
        "contract_number": {"value": "44/B"},
        "amount_eur": {"value": 1234.5},
    }))
    return g


@pytest.fixture
def db(tmp_path):
    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    d = DB(s)
    yield d
    d.close()


def test_evaluate_perfect(db, gold):
    db.upsert_document("doc_a", "/a.pdf", "sha", 1)
    db.upsert_extraction("doc_a", "contract_number", '"44/B"', "44/B", 1, None, 1, "validated", "high")
    db.upsert_extraction("doc_a", "amount_eur", "1234.5", "1.234,50", 1, None, 1, "validated", "high")
    m = evaluate(gold, db=db, split="sealed")
    assert m.silent_error_rate == 0.0
    assert all(f.precision == 1.0 and f.recall == 1.0 for f in m.fields)


def test_evaluate_silent_error(db, gold):
    db.upsert_document("doc_a", "/a.pdf", "sha", 1)
    # valore sbagliato ma confidence high -> errore silenzioso
    db.upsert_extraction("doc_a", "contract_number", '"99/Z"', "99/Z", 1, None, 1, "validated", "high")
    db.upsert_extraction("doc_a", "amount_eur", "1234.5", "1.234,50", 1, None, 1, "validated", "high")
    m = evaluate(gold, db=db, split="sealed")
    assert m.silent_error_rate > 0.0


def test_print_metrics(db, gold, capsys):
    db.upsert_document("doc_a", "/a.pdf", "sha", 1)
    db.upsert_extraction("doc_a", "contract_number", '"44/B"', "44/B", 1, None, 1, "validated", "high")
    db.upsert_extraction("doc_a", "amount_eur", "1234.5", "1.234,50", 1, None, 1, "validated", "high")
    m = evaluate(gold, db=db, split="sealed")
    print_metrics(m)
    out = capsys.readouterr().out
    assert "Silent error rate" in out


def test_evaluate_penalizes_invented_fields(db, gold):
    """Un campo predetto MAI atteso dal gold è un falso positivo: senza questo
    controllo, precision/recall restano 1.0 anche con elementi inventati."""
    db.upsert_document("doc_a", "/a.pdf", "sha", 1)
    db.upsert_extraction("doc_a", "contract_number", '"44/B"', "44/B", 1, None, 1, "validated", "high")
    db.upsert_extraction("doc_a", "amount_eur", "1234.5", "1.234,50", 1, None, 1, "validated", "high")
    # campo completamente inventato (con confidence alta -> errore silenzioso)
    db.upsert_extraction("doc_a", "parties[0].name", '"Pippo Inventato"', "Pippo", 1,
                         None, 1, "validated", "high")
    m = evaluate(gold, db=db, split="sealed")
    by_path = {f.field_path: f for f in m.fields}
    assert by_path["parties[0].name"].precision == 0.0
    assert by_path["parties[0].name"].n == 0
    # la parte corretta resta corretta
    assert by_path["contract_number"].precision == 1.0
    assert by_path["contract_number"].recall == 1.0
    # l'invenzione high-confidence conta come errore silenzioso
    assert m.silent_error_rate > 0.0


def test_evaluate_uses_latest_attempt(db, gold):
    """Un retry che peggiora le cose: vale l'ultimo tentativo, non il vecchio
    validated."""
    db.upsert_document("doc_a", "/a.pdf", "sha", 1)
    db.upsert_extraction("doc_a", "contract_number", '"44/B"', "44/B", 1, None, 1, "validated", "high")
    db.upsert_extraction("doc_a", "contract_number", '"99/Z"', "pippo", 1, None, 2,
                         "needs_review", "high")
    db.upsert_extraction("doc_a", "amount_eur", "1234.5", "1.234,50", 1, None, 1, "validated", "high")
    m = evaluate(gold, db=db, split="sealed")
    by_path = {f.field_path: f for f in m.fields}
    assert by_path["contract_number"].recall == 0.0
