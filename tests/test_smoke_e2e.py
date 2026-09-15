"""Smoke test end-to-end senza GPU: PDF sintetico reale, client OCR/estrattore stub.

Copre: ingest -> rasterize (PyMuPDF+cv2 reali) -> OCR A (stub) -> OCR B +
reconcile (stub) -> enumerate (stub) -> extract (stub) -> validate (cancelli
reali) -> eval su gold finto. Saltato automaticamente se pymupdf/opencv
non sono installati.

Run: pytest tests/test_smoke_e2e.py -q
"""

from __future__ import annotations

import json

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("cv2")

from pipeline import (  # noqa: E402
    phase1_ingest,
    phase2_ocr_a,
    phase3_ocr_b,
    phase4_enumerate,
    phase5_extract,
    phase6_validate,
    phase8_eval,
)
from pipeline.config import get_settings  # noqa: E402
from pipeline.db import DB  # noqa: E402
from pipeline.ocr_clients import Region  # noqa: E402
from pipeline.schema import ContractStrict  # noqa: E402

PAGE_TEXT = [
    "CONTRATTO N. 44/B",
    "Data emissione: 2024-03-15",
    "Importo: 1.234,50 EUR",
    "Valuta: EUR",
    "Parti:",
    "Mario Rossi - Acquirente - P.IVA 01234567890",
    "Lucia Bianchi - Venditore",
]


def _regions(page_no: int = 1) -> list[Region]:
    return [
        Region(
            page_no=page_no,
            bbox=(100.0, 100.0 + i * 40, 2000.0, 130.0 + i * 40),
            region_type="text",
            text=t,
            order_idx=i,
        )
        for i, t in enumerate(PAGE_TEXT)
    ]


class _StubOCR:
    def ocr_page(self, image_path, page_no):
        return _regions(page_no)


class _StubResolver:
    def read_crop(self, image_path):
        return ""


def _extraction_payload() -> dict:
    def leaf(value, quote):
        return {"value": value, "quote": quote, "page": 1,
                "bbox": [100.0, 100.0, 2000.0, 130.0], "confidence": "high"}

    return {
        "contract_number": leaf("44/B", "CONTRATTO N. 44/B"),
        "issue_date": leaf("2024-03-15", "Data emissione: 2024-03-15"),
        "amount_eur": leaf(1234.50, "Importo: 1.234,50 EUR"),
        "currency": leaf("EUR", "Valuta: EUR"),
        "parties": [
            {
                "name": leaf("Mario Rossi", "Mario Rossi - Acquirente"),
                "role": leaf("Acquirente", "Mario Rossi - Acquirente - P.IVA 01234567890"),
                "vat_id": leaf("01234567890", "P.IVA 01234567890"),
            },
            {
                "name": leaf("Lucia Bianchi", "Lucia Bianchi - Venditore"),
                "role": leaf("Venditore", "Lucia Bianchi - Venditore"),
                "vat_id": {"value": None, "quote": None, "page": 1,
                           "bbox": None, "confidence": "high"},
            },
        ],
    }


class _StubExtractor:
    def __init__(self):
        self.prompts: list[str] = []

    def extract(self, prompt, images_b64=None, guided_json_schema=None):
        self.prompts.append(prompt)
        if "Elenca SOLO" in prompt:  # prompt di enumerazione (Fase 4)
            # il modello legge gli id regione dal prompt (REGIONI (id | ...))
            import re

            region_lines = re.findall(r"\[(\d+)\] \w+: (.*)", prompt)
            return {"items": [
                {"anchor": "Mario Rossi", "page": 1,
                 "region_ids": [int(rid) for rid, txt in region_lines if "Mario Rossi" in txt]},
                {"anchor": "Lucia Bianchi", "page": 1,
                 "region_ids": [int(rid) for rid, txt in region_lines if "Lucia Bianchi" in txt]},
            ]}
        payload = _extraction_payload()
        if "ELEMENTO:" in prompt:  # task di singolo elemento lista (Fase 5)
            anchor = next(
                ln.split("ELEMENTO:", 1)[1].strip()
                for ln in prompt.splitlines() if "ELEMENTO:" in ln
            )
            item = next(p for p in payload["parties"]
                        if p["name"]["value"].split()[0] in anchor)
            return {"parties": [item]}
        return payload


def _make_pdf(path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    for i, line in enumerate(PAGE_TEXT):
        page.insert_text((72, 72 + i * 20), line, fontsize=12)
    doc.save(str(path))
    doc.close()


def test_smoke_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DB_PATH", str(tmp_path / "pipeline.db"))
    monkeypatch.setenv("PIPELINE_WORK_DIR", str(tmp_path / "work"))

    pdf = tmp_path / "contratto.pdf"
    _make_pdf(pdf)

    s = get_settings()
    db = DB(s)
    stub_ocr = _StubOCR()
    stub_x = _StubExtractor()

    doc_id = phase1_ingest.ingest(pdf, db=db, settings=s)
    phase1_ingest.rasterize(doc_id, db=db, settings=s)
    assert db.get_status(doc_id) == "rasterized"
    assert db.get_page(doc_id, 1)["image_path"]

    phase2_ocr_a.run(doc_id, db=db, settings=s, client=stub_ocr)
    assert db.get_status(doc_id) == "ocr_a"
    assert "44/B" in db.get_page(doc_id, 1)["full_text"]

    phase3_ocr_b.run(doc_id, db=db, settings=s,
                     client_b=stub_ocr, resolver=_StubResolver())
    assert db.get_status(doc_id) == "reconciled"

    items = phase4_enumerate.run(doc_id, "parties", db=db, settings=s, client=stub_x)
    assert len(items) == 2
    assert db.get_status(doc_id) == "enumerated"

    phase5_extract.run(doc_id, ContractStrict, db=db, settings=s, client=stub_x)
    assert db.get_status(doc_id) == "extracted"
    pending = db.get_extractions(doc_id, status="pending")
    assert len(pending) >= 8  # 4 piatti + 2x3 parti (null inclusi) + inventory
    # i task di elemento lista ricevono SOLO le regioni dell'inventario
    # (region_ids), non l'intera pagina
    item_prompts = [p for p in stub_x.prompts if "ELEMENTO:" in p]
    assert item_prompts
    assert all("CONTRATTO N. 44/B" not in p for p in item_prompts)

    phase6_validate.run(doc_id, ContractStrict, db=db, settings=s, client=stub_x)
    assert db.get_status(doc_id) == "validated"

    validated = {r["field_path"]: r for r in db.get_extractions(doc_id, status="validated")}
    assert json.loads(validated["contract_number"]["value_json"]) == "44/B"
    assert json.loads(validated["amount_eur"]["value_json"]) == 1234.5
    assert json.loads(validated["parties[1].name"]["value_json"]) == "Lucia Bianchi"
    assert validated["contract_number"]["bbox"] is not None

    # Eval su gold finto: i path dei campi lista devono coincidere
    gold = tmp_path / "gold"
    (gold / "annotations").mkdir(parents=True)
    (gold / "sealed.txt").write_text(doc_id + "\n")
    (gold / "annotations" / f"{doc_id}.json").write_text(json.dumps({
        "contract_number": {"value": "44/B"},
        "issue_date": {"value": "2024-03-15"},
        "amount_eur": {"value": 1234.5},
        "currency": {"value": "EUR"},
        "parties": [
            {"name": {"value": "Mario Rossi"}, "role": {"value": "Acquirente"},
             "vat_id": {"value": "01234567890"}},
            {"name": {"value": "Lucia Bianchi"}, "role": {"value": "Venditore"},
             "vat_id": {"value": None}},
        ],
    }))
    m = phase8_eval.evaluate(gold, db=db, split="sealed")
    assert m.silent_error_rate == 0.0
    assert all(f.exact_match == 1.0 for f in m.fields)


def test_resume_run_from_needs_review(tmp_path, monkeypatch):
    """Resume e2e: rilanciare la pipeline su un documento in needs_review non
    deve sollevare (rasterize/ocr saltano via skip_if_done; extract ritenta i
    task falliti; validate chiude con finalize_review -> done)."""
    monkeypatch.setenv("PIPELINE_DB_PATH", str(tmp_path / "pipeline.db"))
    monkeypatch.setenv("PIPELINE_WORK_DIR", str(tmp_path / "work"))

    pdf = tmp_path / "contratto.pdf"
    _make_pdf(pdf)

    s = get_settings()
    db = DB(s)
    doc_id = phase1_ingest.ingest(pdf, db=db, settings=s)
    phase1_ingest.rasterize(doc_id, db=db, settings=s)
    phase2_ocr_a.run(doc_id, db=db, settings=s, client=_StubOCR())
    phase3_ocr_b.run(doc_id, db=db, settings=s,
                     client_b=_StubOCR(), resolver=_StubResolver())
    phase4_enumerate.run(doc_id, "parties", db=db, settings=s,
                         client=_StubExtractor())

    # primo tentativo: estrattore giù -> needs_review con task fallito
    class Down:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            raise ConnectionError("down")

    phase5_extract.run(doc_id, ContractStrict, db=db, settings=s, client=Down())
    assert db.get_status(doc_id) == "needs_review"

    # rilancio della sequenza completa, come farebbe `run` al resume
    phase1_ingest.rasterize(doc_id, db=db, settings=s)   # skip (needs_review)
    phase2_ocr_a.run(doc_id, db=db, settings=s, client=_StubOCR())   # skip
    phase3_ocr_b.run(doc_id, db=db, settings=s, client_b=_StubOCR(),
                     resolver=_StubResolver())           # skip
    phase4_enumerate.run(doc_id, "parties", db=db, settings=s,
                         client=_StubExtractor())
    stub_x = _StubExtractor()
    phase5_extract.run(doc_id, ContractStrict, db=db, settings=s, client=stub_x)
    phase6_validate.run(doc_id, ContractStrict, db=db, settings=s, client=stub_x)
    assert db.get_status(doc_id) == "done"
