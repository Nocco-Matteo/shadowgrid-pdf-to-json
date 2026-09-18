"""Giro completo della CLI `run` con lo schema del compendium.

PDF vero (PyMuPDF) -> rasterize vera (PyMuPDF + OpenCV) -> OCR A/B stub che
leggono il testo del PDF -> Fase 4/5/6 con estrattore stub che risponde SOLO a
partire dal prompt (id regione, pagine, sezioni) -> export raceTraits.json.
Server vLLM sostituiti da no-op: qui si verifica il cablaggio, non i modelli."""
from __future__ import annotations

import json
import logging
import re

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("cv2")
pytest.importorskip("jsonschema")

from pipeline import (  # noqa: E402
    cli,
    phase2_ocr_a,
    phase3_ocr_b,
    phase4_enumerate,
    phase5_extract,
    phase6_validate,
)
from pipeline.compendium import validate_envelope  # noqa: E402
from pipeline.ocr_clients import Region  # noqa: E402
from pipeline.vllm_runner import VLLMRunner  # noqa: E402

PAGES = [
    ["Dwarf Traits",
     "Your dwarf character has an assortment of inborn abilities.",
     "Darkvision. You have superior vision in dark and dim conditions.",
     "Dwarven Resilience. You have advantage on saving throws against poison,",
     "and you have resistance against poison damage."],
    ["Hill Dwarf",
     "As a hill dwarf, you have keen senses.",
     "Dwarven Toughness. Your hit point maximum increases by 1, and it",
     "increases by 1 every time you gain a level."],
    ["Armor", "Padded 5 gp 11 + Dex modifier", "Leather 10 gp 11 + Dex modifier"],
]
TRAITS = {"Darkvision": None,
          "Dwarven Resilience": ([{"kind": "resistance", "damageTypes": ["poison"]}],
                                 "you have resistance against poison damage"),
          "Dwarven Toughness": ([{"kind": "hp_bonus_per_level", "amount": 1}],
                                "Your hit point maximum increases by 1")}


def _make_pdf(path):
    doc = fitz.open()
    for lines in PAGES:
        page = doc.new_page()
        for i, line in enumerate(lines):
            page.insert_text((72, 72 + i * 20), line, fontsize=11)
    doc.save(str(path))
    doc.close()


class OCRFromPDF:
    """OCR stub: una regione per riga di testo del PDF (come farebbe l'OCR)."""

    def __init__(self, *a, **k):
        pass

    def ocr_page(self, image_path, page_no):
        return [Region(page_no=page_no, bbox=(72, 60 + i * 20, 500, 76 + i * 20),
                       region_type="text", text=t, order_idx=i)
                for i, t in enumerate(PAGES[page_no - 1])]

    def read_crop(self, image_path):
        return ""


REGION_RE = re.compile(r"^\[(\d+)\] (?:p\.(\d+) )?text: (.*)$", re.M)


class PromptOnlyExtractor:
    """Risponde leggendo SOLO il prompt: se il cablaggio non passa id,
    pagine o sezioni giuste, le risposte sono sbagliate e il test fallisce."""

    prompts: list[str] = []

    def __init__(self, *a, **k):
        pass

    def extract(self, prompt, images_b64=None, guided_json_schema=None):
        PromptOnlyExtractor.prompts.append(prompt)
        if prompt.startswith("Elenca SOLO"):
            items, page, section = [], None, None
            for block in re.split(r"^=== PAGE (\d+) ===$", prompt, flags=re.M)[1:]:
                if block.isdigit():
                    page = int(block)
                    continue
                for rid, _, text in REGION_RE.findall(block):
                    if text in ("Dwarf Traits", "Hill Dwarf"):
                        section = text
                    name = text.split(".")[0]
                    if name in TRAITS:
                        items.append({"anchor": name, "page": page,
                                      "region_ids": [int(rid)], "section": section})
            return {"items": items}
        name = re.search(r"^ELEMENTO: (.+)$", prompt, re.M).group(1)
        section = re.search(r"^SEZIONE: (.+)$", prompt, re.M).group(1)
        pages = {text: int(p) for _, p, text in REGION_RE.findall(prompt)}

        def leaf(value, quote):
            page = next(p for text, p in pages.items() if quote in text)
            return {"value": value, "quote": quote, "page": page, "bbox": None,
                    "confidence": "high"}

        null = {"value": None, "quote": None, "page": None, "bbox": None, "confidence": "high"}
        effects = TRAITS[name]
        return {"traits": [{
            "raceName": leaf("Dwarf", section),
            "subraceName": leaf("Hill", section) if section == "Hill Dwarf" else null,
            "featureName": leaf(name, name),
            "grantedAtLevel": null,
            "effects": leaf(*effects) if effects else null,
        }]}


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DB_PATH", str(tmp_path / "runs" / "pipeline.db"))
    monkeypatch.setenv("PIPELINE_WORK_DIR", str(tmp_path / "runs" / "work"))
    monkeypatch.setattr(VLLMRunner, "start", lambda self, model, **k: "http://stub/v1")
    monkeypatch.setattr(VLLMRunner, "start_extractor", lambda self, **k: "http://stub/v1")
    monkeypatch.setattr(VLLMRunner, "stop", lambda self: None)
    monkeypatch.setattr(phase2_ocr_a, "PaddleOCRVLClient", OCRFromPDF)
    monkeypatch.setattr(phase3_ocr_b, "DeepSeekOCRClient", OCRFromPDF)
    monkeypatch.setattr(phase3_ocr_b, "ResolverClient", OCRFromPDF)
    for mod in (phase4_enumerate, phase5_extract, phase6_validate):
        monkeypatch.setattr(mod, "ExtractorClient", PromptOnlyExtractor)
    PromptOnlyExtractor.prompts = []
    pdf = tmp_path / "phb.pdf"
    _make_pdf(pdf)
    return tmp_path, pdf


def _export(tmp_path):
    out = list((tmp_path / "runs" / "export").glob("*/raceTraits.json"))
    assert len(out) == 1, out
    return json.loads(out[0].read_text()), out[0]


def test_cli_run_race_traits_end_to_end(pipeline_env):
    tmp_path, pdf = pipeline_env
    assert cli.main(["run", str(pdf), "--source", "players_handbook"]) == 0

    envelope, out = _export(tmp_path)
    assert validate_envelope(envelope, "raceTraits.json") == []
    got = {d["id"]: d for d in envelope["definitions"]}
    assert set(got) == {"race_dwarf_dwarven_resilience", "race_dwarf_hill_dwarven_toughness"}
    assert got["race_dwarf_dwarven_resilience"]["effects"] == [
        {"kind": "resistance", "damageTypes": ["poison"]}]
    assert got["race_dwarf_hill_dwarven_toughness"]["subraceName"] == "Hill"
    assert "Darkvision" in (out.parent / "raceTraits.json.notes.txt").read_text()

    from pipeline.db import DB
    db = DB()
    doc_id = out.parent.name
    assert db.get_status(doc_id) == "validated"
    assert len(db.get_pages(doc_id)) == 3
    assert all(p["deskew_angle"] == 0.0 for p in db.get_pages(doc_id))  # pagina dritta
    # nessun campo del contratto: lo schema usato è davvero quello del compendium
    assert all(r["field_path"].startswith("traits") for r in db.get_extractions(doc_id))


def test_cli_run_refuses_mixed_schema_until_reset(pipeline_env, caplog):
    tmp_path, pdf = pipeline_env
    from pipeline import phase1_ingest
    from pipeline.db import DB

    db = DB()
    doc_id = phase1_ingest.ingest(pdf, db=db)
    # estrazione precedente con lo schema del contratto (come sul PDF vero)
    cli.main(["--schema", "race_traits", "rasterize", str(pdf)])
    db.upsert_extraction(doc_id, "contract_number", '"44/B"', "x", 1, None, 1, "needs_review")

    with caplog.at_level(logging.ERROR):
        cli.main(["run", str(pdf), "--source", "players_handbook"])
    assert "reset-extraction --doc-id " + doc_id in caplog.text
    assert not (tmp_path / "runs" / "export").exists()

    cli.main(["reset-extraction", "--doc-id", doc_id])
    cli.main(["run", str(pdf), "--source", "players_handbook"])
    envelope, _ = _export(tmp_path)
    assert len(envelope["definitions"]) == 2
