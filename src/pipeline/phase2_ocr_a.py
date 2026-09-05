"""Fase 2 — OCR primario (PaddleOCR-VL-1.6).

Una predict() per pagina -> regioni con tipo (testo/tabella/formula/timbro),
bbox, testo, ordine di lettura. Riproietta le bbox sulle coordinate dell'immagine
originale invertendo la rotazione di deskew. Tabelle conservate come struttura.
Scrivi in regions. Concatena il testo in pages.full_text (riferimento per fuzzy Fase 6).
"""

from __future__ import annotations

import logging

from .config import Settings, get_settings
from .db import DB
from .geometry import invert_deskew_bbox
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
    if db.get_status(doc_id) == "ocr_a":
        log.info("OCR A già fatto: %s", doc_id)
        return
    if db.get_status(doc_id) != "rasterized":
        raise ValueError(f"OCR A richiede stato rasterized, trovato {db.get_status(doc_id)}")

    client = client or PaddleOCRVLClient(s.ocr_a_url)

    for page in db.get_pages(doc_id):
        page_no = page["page_no"]
        existing = db.get_regions(doc_id, page_no, engine="a")
        if existing:
            continue  # idempotente per pagina

        regions = client.ocr_page(page["image_path"], page_no)
        # Riproietta bbox invertendo il deskew
        deskew = page["deskew_angle"] or 0.0
        # Dimensioni immagine non salvate direttamente; le ricaviamo dal PNG
        img_w, img_h = _image_size(page["image_path"])

        full_text_parts: list[str] = []
        for r in regions:
            bbox_orig = None
            if r.bbox is not None:
                bbox_orig = invert_deskew_bbox(r.bbox, deskew, img_w, img_h)
            db.add_region(
                doc_id=doc_id,
                page_no=page_no,
                bbox=bbox_orig,
                region_type=r.region_type,
                text=r.text,
                engine="a",
                order_idx=r.order_idx,
            )
            if r.region_type == "text":
                full_text_parts.append(r.text)

        db.set_page_full_text(doc_id, page_no, "\n".join(full_text_parts))
        log.info("OCR A page %d: %d regioni", page_no, len(regions))

    db.transition(doc_id, "rasterized", "ocr_a")


def _image_size(path: str) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        # Fallback se PIL non disponibile
        return 0, 0


__all__ = ["run"]
