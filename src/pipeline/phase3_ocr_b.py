"""Fase 3 — Secondo OCR e riconciliazione.

Spegni server A, avvia DeepSeek-OCR 2 (o dots.ocr), engine='b'.
Allineamento segmentazioni:
  - primo passo per IoU bbox (soglia 0.5);
  - non appaiate -> allineamento per contenuto con rapidfuzz.
Confronto similarità normalizzata:
  - >= 0.95 -> accetta testo A;
  - < 0.95  -> conflitto.
Risoluzione: ritaglia bbox unione + ~10px margine, upscale 2x, rileggi con terzo
passaggio mirato. Maggioranza 2-su-3. Se tutti divergono -> confidence='low' e coda umana.

Il testo vincitore NON viene solo archiviato in region_conflicts: la colonna
regions.text_canonical (e pages.canonical_text, ricostruito per pagina) diventa
la rappresentazione canonica usata da tutte le fasi a valle (Fase 4/5/6 e
revisione). Gli originali A/B restano in regions.text.

Kill-safe: le regioni B di una pagina vengono scritte in una transazione unica
marcata da pages.ocr_b_done; conflitto e testo canonico vengono scritti nella
STESSA transazione; i conflitti già registrati non vengono risolti una seconda
volta al resume (ma le risoluzioni persistite vengono riapplicate al canonico,
per riparare interruzioni di versioni precedenti).
"""

from __future__ import annotations

import logging
from collections import Counter

from .config import Settings, get_settings
from .db import DB
from .geometry import expand, iou, union_bbox
from .ocr_clients import DeepSeekOCRClient, Region, ResolverClient
from .text_norm import similarity

log = logging.getLogger(__name__)


def run(
    doc_id: str,
    db: DB | None = None,
    settings: Settings | None = None,
    client_b: DeepSeekOCRClient | None = None,
    resolver: ResolverClient | None = None,
) -> None:
    s = settings or get_settings()
    db = db or DB(s)
    st = db.get_status(doc_id)
    if st in ("reconciled", "enumerated", "extracted", "validated", "done",
              "needs_review"):
        log.info("OCR B/reconcile già fatto: %s", doc_id)
        return
    if st not in ("ocr_a", "ocr_b"):
        raise ValueError(f"OCR B richiede stato ocr_a, trovato {st}")

    client_b = client_b or DeepSeekOCRClient(s.ocr_b_url, s.model_b)
    resolver = resolver or ResolverClient(s.ocr_b_url, s.model_b)

    # Conflitti già registrati in run precedenti (idempotenza del resume)
    recorded = {c["region_id"] for c in db.get_conflicts(doc_id)}
    # Ripara risoluzioni registrate ma non applicate al testo canonico
    # (interruzioni tra le due scritture in versioni precedenti): riapplica
    # le risoluzioni persistite, in un'unica transazione ciascuna.
    for c in db.get_conflicts(doc_id):
        if c["resolver"] in ("majority", "human") and c["region_id"]:
            db.apply_conflict_resolution(
                doc_id, c["region_id"], None, None,
                c["resolved_text"], c["resolver"],
            )

    total_regions = 0
    conflict_count = 0

    for page in db.get_pages(doc_id):
        page_no = page["page_no"]
        if page["ocr_b_done"]:
            regions_b = _rows_to_regions(
                db.get_regions(doc_id, page_no, engine="b"), page_no
            )  # già scritta (commit atomico), niente re-OCR
        else:
            regions_b = client_b.ocr_page(page["image_path"], page_no)
            db.save_page_ocr_b(doc_id, page_no, regions_b)
            regions_b = _rows_to_regions(
                db.get_regions(doc_id, page_no, engine="b"), page_no
            )

        regions_a = _rows_to_regions(db.get_regions(doc_id, page_no, engine="a"), page_no)
        total_regions += len(regions_a)
        conflict_count += _reconcile_page(
            db, doc_id, page_no, regions_a, regions_b, page["image_path"], resolver, s,
            recorded,
        )
        # Il testo canonico della pagina riflette le risoluzioni majority
        db.rebuild_page_canonical_text(doc_id, page_no)

    rate = conflict_count / total_regions if total_regions else 0.0
    if rate > s.conflict_rate_warn:
        log.warning(
            "Conflitti %.1f%% (> %.0f%%): problema probabile a monte (Fase 1)",
            rate * 100, s.conflict_rate_warn * 100,
        )

    if db.get_status(doc_id) == "ocr_a":
        db.transition(doc_id, "ocr_a", "ocr_b")
    db.transition(doc_id, "ocr_b", "reconciled")


def _rows_to_regions(rows, page_no: int) -> list[Region]:
    import json

    out = []
    for r in rows:
        bbox = tuple(json.loads(r["bbox"])) if r["bbox"] else None
        out.append(Region(page_no=page_no, bbox=bbox, region_type=r["region_type"],
                          text=r["text"], order_idx=r["order_idx"],
                          region_id=r["region_id"]))
    return out


def _reconcile_page(
    db: DB,
    doc_id: str,
    page_no: int,
    regions_a: list[Region],
    regions_b: list[Region],
    image_path: str,
    resolver: ResolverClient,
    s: Settings,
    recorded: set[int],
) -> int:
    """Riconcilia le due segmentazioni di una pagina. Ritorna il numero di conflitti."""
    paired_b = [False] * len(regions_b)
    conflicts = 0

    for ra in regions_a:
        # 1) allineamento per IoU
        best_j, best_iou = -1, 0.0
        for j, rb in enumerate(regions_b):
            if paired_b[j]:
                continue
            if ra.bbox and rb.bbox:
                v = iou(ra.bbox, rb.bbox)
                if v > best_iou:
                    best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= s.iou_align_threshold:
            paired_b[best_j] = True
            rb = regions_b[best_j]
            if similarity(ra.text, rb.text) >= s.text_conflict_threshold:
                continue  # accetta A
            # conflitto -> risoluzione
            conflicts += 1
            _resolve_conflict(db, doc_id, page_no, ra, rb, image_path, resolver, s,
                              recorded)
            continue

        # 2) non appaiata per IoU -> allineamento per contenuto
        best_j, best_sim = -1, 0.0
        for j, rb in enumerate(regions_b):
            if paired_b[j]:
                continue
            v = similarity(ra.text, rb.text)
            if v > best_sim:
                best_sim, best_j = v, j
        if best_j >= 0 and best_sim >= s.text_conflict_threshold:
            paired_b[best_j] = True
            continue
        # regione A senza corrispondenza: resta A (motore primario)

    return conflicts


def _resolve_conflict(
    db: DB,
    doc_id: str,
    page_no: int,
    ra: Region,
    rb: Region,
    image_path: str,
    resolver: ResolverClient,
    s: Settings,
    recorded: set[int],
) -> None:
    """Ritaglia bbox unione + margine, upscale 2x, rileggi. Maggioranza 2-su-3."""
    if ra.region_id and ra.region_id in recorded:
        return  # già risolto in un run precedente (resume)

    import cv2

    bbox_u = union_bbox(ra.bbox, rb.bbox) if ra.bbox and rb.bbox else (ra.bbox or rb.bbox)
    if bbox_u is None:
        # senza bbox non possiamo ritagliare; teniamo A e marchiamo conflitto low
        db.add_conflict(doc_id, ra.region_id or 0, ra.text, rb.text, ra.text, "fallback_a")
        return

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    bbox_e = expand(bbox_u, s.conflict_margin_px, w, h)
    x0, y0, x1, y1 = [int(v) for v in bbox_e]
    crop = img[y0:y1, x0:x1]
    crop_up = cv2.resize(crop, None, fx=s.conflict_upscale, fy=s.conflict_upscale,
                         interpolation=cv2.INTER_CUBIC)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        cv2.imwrite(tf.name, crop_up)
        crop_path = tf.name
    try:
        text_c = resolver.read_crop(crop_path)
    finally:
        import os

        os.unlink(crop_path)

    # Maggioranza 2-su-3
    votes = [ra.text, rb.text, text_c]
    counter = Counter(votes)
    winner, count = counter.most_common(1)[0]
    if count >= 2:
        resolved = winner
        resolver_name = "majority"
    else:
        # tutti divergenti: fallback al primario, ma il conflitto finisce nella
        # coda umana (Fase 7 pesca i resolver divergent_low/fallback_a)
        resolved = ra.text
        resolver_name = "divergent_low"

    # Conflitto + testo canonico (regione e pagina) in UNA transazione:
    # un'interruzione non può lasciare il conflitto registrato con un
    # canonico stantio che al resume non verrebbe più corretto.
    db.apply_conflict_resolution(
        doc_id, ra.region_id or 0, ra.text, rb.text, resolved, resolver_name,
        page_no=page_no,
    )


__all__ = ["run"]
