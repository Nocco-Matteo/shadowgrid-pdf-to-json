"""Test di phase4_enumerate.py con client stub."""
from __future__ import annotations

import pytest

from pipeline.config import Settings
from pipeline.db import DB
from pipeline.phase4_enumerate import run


class StubExtractor:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def extract(self, prompt, images_b64=None, guided_json_schema=None):
        self.calls += 1
        return self.payload


@pytest.fixture
def db(tmp_path):
    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    d = DB(s)
    d.upsert_document("doc_a", "/a.pdf", "sha", 1)
    d.set_status("doc_a", "reconciled")
    d.set_page("doc_a", 1, full_text="Contratto n. 44/B del 2024. Parti: ACME, Beta.")
    yield d
    d.close()


def test_enumerate_valid_anchor(db):
    client = StubExtractor({"items": [
        {"anchor": "Contratto n. 44/B", "page": 1, "region_ids": []},
    ]})
    items = run("doc_a", "parties", db=db, client=client)
    assert len(items) == 1
    assert items[0].anchor == "Contratto n. 44/B"
    # stato transitito a enumerated
    assert db.get_status("doc_a") == "enumerated"


def test_enumerate_drops_invented_anchor(db):
    client = StubExtractor({"items": [
        {"anchor": "Pippo Pluto Inesistente", "page": 1, "region_ids": []},
        {"anchor": "Contratto n. 44/B", "page": 1, "region_ids": []},
    ]})
    items = run("doc_a", "parties", db=db, client=client)
    assert len(items) == 1
    assert items[0].anchor == "Contratto n. 44/B"


def test_enumerate_dedup(db):
    client = StubExtractor({"items": [
        {"anchor": "Contratto n. 44/B", "page": 1, "region_ids": []},
        {"anchor": "contratto n. 44/b", "page": 1, "region_ids": []},
    ]})
    items = run("doc_a", "parties", db=db, client=client)
    assert len(items) == 1


def test_enumerate_coverage_mismatch_needs_review(db):
    client = StubExtractor({"items": [
        {"anchor": "Contratto n. 44/B", "page": 1, "region_ids": []},
    ]})
    run("doc_a", "parties", db=db, client=client, expected_count=5)
    assert db.get_status("doc_a") == "needs_review"
