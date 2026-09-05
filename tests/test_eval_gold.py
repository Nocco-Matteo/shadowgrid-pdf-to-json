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
