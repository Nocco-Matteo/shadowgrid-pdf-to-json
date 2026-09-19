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

import json
import logging

from .config import Settings, get_settings
from .db import DB, skip_if_done
from .geometry import reading_order
from .ocr_clients import PaddleOCRVLClient, _image_height

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

    client = client or PaddleOCRVLClient(s.ocr_a_url, s.model_a)

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


def reorder(doc_id: str, db: DB | None = None, settings: Settings | None = None) -> int:
    """Riapplica l'ordine di lettura alle regioni GIÀ in database.

    L'ordine dipende solo dalle bbox, che sono già salvate: dopo un cambio di
    `geometry.reading_order` non serve rifare l'OCR, che su un manuale sono ore
    di GPU per due modelli. Riordina entrambi gli engine e ricostruisce
    `pages.full_text` e `pages.canonical_text`, che sono concatenazioni delle
    regioni nell'ordine di lettura.

    NON tocca le estrazioni: il testo cambia sotto di loro, quindi vanno
    azzerate a parte (`reset-extraction`) e rifatte da Fase 4.

    Ritorna il numero di pagine il cui ordine è cambiato.
    """
    db = db or DB(settings or get_settings())
    changed = 0
    for page in db.get_pages(doc_id):
        page_no = page["page_no"]
        height = _image_height(page["image_path"]) if page["image_path"] else None
        page_changed = False
        for engine in ("a", "b"):
            rows = db.get_regions(doc_id, page_no, engine=engine)
            if len(rows) < 2:
                continue
            boxes = [tuple(json.loads(r["bbox"])) if r["bbox"] else None for r in rows]
            flow = [bool((r["text_canonical"] or r["text"] or "").strip()) for r in rows]
            order = reading_order(boxes, height, in_flow=flow)
            if order == list(range(len(rows))):
                continue
            db.set_region_order(doc_id, [rows[i]["region_id"] for i in order])
            page_changed = True
        if page_changed:
            db.rebuild_page_full_text(doc_id, page_no)
            db.rebuild_page_canonical_text(doc_id, page_no)
            changed += 1
    log.info("Riordino %s: %d pagine cambiate", doc_id, changed)
    return changed


__all__ = ["run", "reorder"]
