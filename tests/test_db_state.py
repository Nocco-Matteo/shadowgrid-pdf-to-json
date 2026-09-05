"""Test di db.py: transizioni, idempotenza."""
from __future__ import annotations

import pytest

from pipeline.config import Settings
from pipeline.db import DB, EDGES, STATES


@pytest.fixture
def db(tmp_path):
    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    d = DB(s)
    yield d
    d.close()


def test_upsert_and_status(db):
    db.upsert_document("doc_x", "/a.pdf", "sha1", 3)
    assert db.get_status("doc_x") == "ingested"


def test_dedup_by_sha(db):
    db.upsert_document("doc_x", "/a.pdf", "sha1", 3)
    existing = db.find_by_sha256("sha1")
    assert existing is not None
    assert existing["doc_id"] == "doc_x"


def test_valid_transition(db):
    db.upsert_document("doc_x", "/a.pdf", "sha1", 3)
    db.transition("doc_x", "ingested", "rasterized")
    assert db.get_status("doc_x") == "rasterized"


def test_invalid_transition_raises(db):
    db.upsert_document("doc_x", "/a.pdf", "sha1", 3)
    with pytest.raises(ValueError):
        db.transition("doc_x", "ingested", "validated")  # salta troppi stati


def test_transition_wrong_source_raises(db):
    db.upsert_document("doc_x", "/a.pdf", "sha1", 3)
    with pytest.raises(ValueError):
        db.transition("doc_x", "rasterized", "ocr_a")  # non è in rasterized


def test_failed_from_any_state(db):
    for src in STATES:
        if src in {"done", "needs_review", "failed"}:
            continue
        assert "failed" in EDGES[src]


def test_set_page_idempotent(db):
    db.upsert_document("doc_x", "/a.pdf", "sha1", 3)
    db.set_page("doc_x", 1, image_path="/p1.png", dpi=300, deskew_angle=0.0)
    db.set_page("doc_x", 1, full_text="hello")  # non sovrascrive image_path
    p = db.get_page("doc_x", 1)
    assert p["image_path"] == "/p1.png"
    assert p["full_text"] == "hello"


def test_add_region_and_get(db):
    db.upsert_document("doc_x", "/a.pdf", "sha1", 3)
    rid = db.add_region("doc_x", 1, (0, 0, 10, 10), "text", "ciao", "a", 0)
    regions = db.get_regions("doc_x", 1, engine="a")
    assert len(regions) == 1
    assert regions[0]["region_id"] == rid


def test_extraction_upsert_conflict(db):
    db.upsert_document("doc_x", "/a.pdf", "sha1", 3)
    db.upsert_extraction("doc_x", "f", '"v"', "q", 1, None, 1, "pending")
    db.upsert_extraction("doc_x", "f", '"v2"', "q2", 1, None, 1, "validated")  # stesso attempt -> update
    rows = db.get_extractions("doc_x")
    assert len(rows) == 1
    assert rows[0]["status"] == "validated"
