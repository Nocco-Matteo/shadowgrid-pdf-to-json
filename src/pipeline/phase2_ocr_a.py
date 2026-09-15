"""Fase 2 — OCR primario (PaddleOCR-VL-1.6).

Una predict() per pagina -> regioni con tipo (testo/tabella/formula/timbro),
bbox, testo, ordine di lettura. Tabelle conservate come struttura (HTML del
blocco). Scrive in regions. Concatena il testo di TUTTE le regioni (tabelle
incluse) in pages.full_text e pages.canonical_text (riferimento per fuzzy
Fase 4/6): una citazione autentica presente solo in tabella deve superare il
grounding.

Sistema di coordinate unico: le bbox sono riferite all'immagine salvata su
disco (già deskewed), che è la stessa usata da OCR B, dai crop di risoluzione
dei conflitti e dalla UI di revisione. Niente riproiezioni tra fasi.

Kill-safe: ogni pagina viene scritta in una sola transazione (regioni +
full_text + canonical); una pagina è completa se e solo se full_text IS NOT
NULL. Un'interruzione a metà pagina lascia la pagina incompleta e al resume
viene ri-OCR-ata per intero.
"""

from __future__ import annotations

import logging

from .config import Settings, get_settings
from .db import DB, skip_if_done
from .ocr_clients import PaddleOCRVLClient

log = logging.getLogger(__name__)


def run(
    doc_id: str,
    db: DB | None = None,
    settings: Settings | None = None,
    client: PaddleOCRVLClient | None = None,
) -> None:
    s = settings or get_settings()
    db = db or DB(s)
    if skip_if_done(db, doc_id, "rasterized", "ocr_a"):
        log.info("OCR A già fatto: %s", doc_id)
        return

    client = client or PaddleOCRVLClient(s.ocr_a_url)

    for page in db.get_pages(doc_id):
        page_no = page["page_no"]
        if page["full_text"] is not None:
            continue  # pagina già completata (commit atomico, kill-safe)

        regions = client.ocr_page(page["image_path"], page_no)
        # full_text = tutte le regioni in ordine di lettura, tabelle incluse
        # (il loro contenuto strutturato resta citabile per il grounding)
        full_text = "\n".join(r.text for r in regions)
        db.save_page_ocr_a(doc_id, page_no, regions, full_text)
        log.info("OCR A page %d: %d regioni", page_no, len(regions))

    db.transition(doc_id, "rasterized", "ocr_a")


__all__ = ["run"]
