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


def test_gold_items_resolve_by_identity_not_position(tmp_path):
    """Gli indici di lista non sono stabili fra run: `Damage Resistance` era
    traits[9] nella run 1 e traits[12] nella run 3. Un gold indicizzato per
    posizione confronterebbe tratti diversi, e il metro misurerebbe rumore.

    L'identità è (pagina, anchor), con l'anchor confrontata per prefisso perché
    anche quella cambia lunghezza fra una run e l'altra."""
    import json

    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase8_eval import evaluate

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "w")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    # in questa run il tratto cercato sta in posizione 2, non 0
    db.upsert_extraction("d", "traits$inventory", json.dumps([
        {"anchor": "Speed.", "page": 1},
        {"anchor": "Darkvision.", "page": 1},
        {"anchor": "Hellish Resistance. You have resistance to fire damage.", "page": 3},
    ]), None, None, None, 1, "validated", "high")
    db.upsert_extraction("d", "traits[2].featureName", '"Hellish Resistance"',
                         "Hellish Resistance", 3, None, 1, "validated", "high")

    gold = tmp_path / "gold"
    (gold / "annotations").mkdir(parents=True)
    (gold / "dev.txt").write_text("d\n")
    (gold / "annotations" / "d.json").write_text(json.dumps({"traits": [
        {"_anchor": "Hellish Resistance.", "_page": 3,
         "featureName": {"value": "Hellish Resistance"}},
    ]}))

    m = evaluate(gold, db=db, settings=s, split="dev")
    got = {f.field_path: f for f in m.fields}
    assert "traits[2].featureName" in got, "anchor non risolta sull'indice reale"
    assert got["traits[2].featureName"].exact_match == 1.0
    db.close()


def test_gold_item_missing_from_run_counts_as_recall_loss(tmp_path):
    """Un tratto annotato ma mai enumerato non deve sparire dal conteggio."""
    import json

    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase8_eval import evaluate

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "w")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.upsert_extraction("d", "traits$inventory", json.dumps(
        [{"anchor": "Speed.", "page": 1}]), None, None, None, 1, "validated", "high")

    gold = tmp_path / "gold"
    (gold / "annotations").mkdir(parents=True)
    (gold / "dev.txt").write_text("d\n")
    (gold / "annotations" / "d.json").write_text(json.dumps({"traits": [
        {"_anchor": "Stout Resilience.", "_page": 9,
         "featureName": {"value": "Stout Resilience"}},
    ]}))

    m = evaluate(gold, db=db, settings=s, split="dev")
    missing = [f for f in m.fields if "mancante" in f.field_path]
    assert missing and missing[0].recall == 0.0
    db.close()


def test_compare_annotations_finds_disagreements_and_skips_blanks():
    """Doppia annotazione: i campi non ancora compilati non sono disaccordi,
    quelli compilati e diversi sì."""
    from pipeline.phase8_eval import PLACEHOLDER, compare_annotations

    a = {"traits": [{"_anchor": "Fleet of Foot.", "_page": 20,
                     "raceName": {"value": "Elf"}, "subraceName": {"value": "Wood"},
                     "featureName": {"value": "Fleet of Foot"}}]}
    b = {"traits": [{"_anchor": "Fleet of Foot.", "_page": 20,
                     "raceName": {"value": "Wood Elf"}, "subraceName": {"value": None},
                     "featureName": {"value": PLACEHOLDER}}]}
    diffs = compare_annotations(a, b)
    assert len(diffs) == 2  # featureName non compilato: non conta
    assert any("raceName" in d for d in diffs)
    assert any("subraceName" in d for d in diffs)
    assert compare_annotations(a, a) == []
