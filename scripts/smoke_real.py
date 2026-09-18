#!/usr/bin/env python3
"""Smoke test REALE — modelli veri su GPU, da lanciare dentro WSL2 sul target.

Diversamente da tests/test_smoke_e2e.py (stub, gira ovunque), questo script:
  - avvia davvero vLLM per ciascun modello (caricamento lento, ~5-15 min l'uno);
  - rasterizza un PDF vero (sintetico generato al volo, o --pdf <file>);
  - OCR A con PaddleOCR-VL, opzionale OCR B + reconcile con DeepSeek-OCR-2;
  - enumerate + extract con l'estrattore AWQ;
  - validate reale (grounding fuzzy sul testo OCR vero);
  - stampa un report finale per campo.

Uso (dentro WSL2, repo clonato, venv attivo):
    python scripts/smoke_real.py                  # completo
    python scripts/smoke_real.py --skip-ocr-b     # senza secondo OCR (più veloce)
    python scripts/smoke_real.py --pdf mio.pdf    # PDF reale invece del sintetico

Prerequisiti: nvidia-smi OK in WSL2, `pip install -r requirements.txt` con i
pacchetti CUDA scommentati, modelli in `pipeline.yaml` raggiungibili (HF cache
o download automatico al primo vllm serve).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import (  # noqa: E402
    phase1_ingest,
    phase2_ocr_a,
    phase3_ocr_b,
    phase4_enumerate,
    phase5_extract,
    phase6_validate,
)
from pipeline.config import get_settings  # noqa: E402
from pipeline.db import DB  # noqa: E402
from pipeline.schema import ContractStrict  # noqa: E402
from pipeline.vllm_runner import VLLMRunner  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    stream=sys.stderr)
log = logging.getLogger("smoke_real")

SMOKE_TEXT = [
    "CONTRATTO N. 44/B",
    "Data emissione: 2024-03-15",
    "Importo: 1.234,50 EUR",
    "Valuta: EUR",
    "Parti:",
    "Mario Rossi - Acquirente - P.IVA 01234567890",
    "Lucia Bianchi - Venditore",
]


def make_synthetic_pdf(out: Path) -> Path:
    import fitz

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    for i, line in enumerate(SMOKE_TEXT):
        page.insert_text((72, 72 + i * 22), line, fontsize=13)
    doc.save(str(out))
    doc.close()
    return out


def check_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        print("ATTENZIONE: nvidia-smi non trovato. vLLM non funzionerà senza GPU.")
        return
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"GPU: {out.stdout.strip() or 'non rilevata'}")


def banner(msg: str) -> None:
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}", flush=True)


def report(db: DB, doc_id: str) -> int:
    """Stampa lo stato finale e le estrazioni (ultimo tentativo per campo).
    Ritorna exit code."""
    status = db.get_status(doc_id)
    banner(f"REPORT — {doc_id} — stato: {status}")
    print(f"{'field_path':30} {'status':13} {'conf':5} value")
    latest = {}
    for e in db.get_extractions(doc_id):
        if e["field_path"] not in latest or e["attempt"] > latest[e["field_path"]]["attempt"]:
            latest[e["field_path"]] = e
    n_val = n_rev = 0
    for fp in sorted(latest):
        e = latest[fp]
        if e["status"] == "validated":
            n_val += 1
        elif e["status"] in ("needs_review", "rejected"):
            n_rev += 1
        print(f"{fp:30} {e['status']:13} {e['confidence'] or '-':5} {e['value_json']}")
    conflicts = db.get_conflicts(doc_id)
    if conflicts:
        print(f"\nConflitti OCR: {len(conflicts)}")
        for c in conflicts:
            print(f"  region {c['region_id']} [{c['resolver']}]: "
                  f"A={c['text_a']!r} B={c['text_b']!r} -> {c['resolved_text']!r}")
    print(f"\nEstrazioni validate: {n_val}, da revisionare/scartate: {n_rev}")
    ok = status in ("validated", "done") and n_val > 0
    print("\nSMOKE TEST:", "OK" if ok else "FALLITO — vedi log sopra")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test reale (GPU + vLLM)")
    ap.add_argument("--pdf", type=Path, default=None,
                    help="PDF reale; default: sintetico generato al volo")
    ap.add_argument("--skip-ocr-b", action="store_true",
                    help="salta il secondo OCR (metà tempo di smoke)")
    ap.add_argument("--list-field", dest="list_fields", nargs="*",
                    default=["parties"])
    args = ap.parse_args()

    banner("SMOKE TEST REALE — shadowgrid-pdf-to-json")
    check_gpu()

    s = get_settings()
    db = DB(s)

    pdf = args.pdf or make_synthetic_pdf(Path(s.work_dir) / "smoke" / "contratto_smoke.pdf")
    log.info("PDF: %s", pdf)

    runner = VLLMRunner(s)
    doc_id = phase1_ingest.ingest(pdf, db=db, settings=s)

    try:
        banner("Fase 1 — rasterize")
        phase1_ingest.rasterize(doc_id, db=db, settings=s)

        banner(f"Fase 2 — OCR A ({s.model_a})")
        s.ocr_a_url = runner.start(s.model_a, port=8080, phase="ocr_a")
        try:
            phase2_ocr_a.run(doc_id, db=db, settings=s)
        finally:
            runner.stop()
        ft = db.get_page(doc_id, 1)["full_text"] or ""
        log.info("full_text pagina 1: %d caratteri", len(ft))

        if not args.skip_ocr_b:
            banner(f"Fase 3 — OCR B + reconcile ({s.model_b})")
            s.ocr_b_url = runner.start(s.model_b, port=8080, phase="ocr_b")
            try:
                phase3_ocr_b.run(doc_id, db=db, settings=s)
            finally:
                runner.stop()
        else:
            if db.get_status(doc_id) == "ocr_a":
                db.transition(doc_id, "ocr_a", "reconciled")

        banner(f"Fasi 4-6 — enumerate + extract + validate ({s.extractor_model})")
        # La validazione avviene con il server ancora attivo: i retry di
        # Fase 6 rilanciano l'estrattore via HTTP (come in cli.cmd_run).
        s.extractor_url = runner.start_extractor(port=8080, phase="extract")
        try:
            for lf in args.list_fields or []:
                phase4_enumerate.run(doc_id, lf, db=db, settings=s)
            phase5_extract.run(doc_id, ContractStrict, db=db, settings=s)
            phase6_validate.run(doc_id, ContractStrict, db=db, settings=s)
        finally:
            runner.stop()
    except Exception as e:
        log.exception("Smoke fallito: %s", e)
        print("\nHint: controlla runs/vllm_*.log; VRAM libera; modelli in pipeline.yaml")
        runner.stop()
        return 1

    return report(db, doc_id)


if __name__ == "__main__":
    raise SystemExit(main())
