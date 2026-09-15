"""Fase 1 — Ingestione e rasterizzazione.

- sha256 del file sorgente -> doc_id. Deduplica: se già visto, salta.
- PyMuPDF: page.get_pixmap(dpi=300). Per scansioni degradate prova 400 DPI.
- Deskew: stima angolo (cv2.minAreaRect sul testo binarizzato, fallback Hough).
  Ruota solo se |angolo| > 0.3°.
- Normalizzazione: grayscale + cv2.createCLAHE. Non binarizzare aggressivamente.
- Salva PNG. Registra l'angolo di deskew (serve per riproiettare le bbox).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .db import DB

log = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_id_from_sha(sha: str) -> str:
    # Prefisso per distinguere eventuali altri namespace; usa i primi 16 byte.
    return f"doc_{sha[:32]}"


# ---------------------------------------------------------------------------
# Rasterizzazione (PyMuPDF). Import lazy per permettere test senza dipendenze.
# ---------------------------------------------------------------------------


def _open_pdf(path: Path):
    import fitz  # PyMuPDF

    return fitz.open(path)


def rasterize_page(page: Any, dpi: int, out_path: Path) -> tuple[int, int]:
    import fitz  # noqa: F401

    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat)
    pix.save(str(out_path))
    return pix.width, pix.height


# ---------------------------------------------------------------------------
# Deskew + normalizzazione (OpenCV). Import lazy.
# ---------------------------------------------------------------------------


def estimate_deskew_angle(img_gray: Any) -> float:
    """Stima l'angolo di skew con cv2.minAreaRect sul contorno del testo binarizzato.
    Fallback: trasformata di Hough sulle linee di testo."""
    import cv2

    # Binarizzazione Otsu inversa -> testo bianco su nero
    _, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Rumore piccolo via morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(bw, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pts: list[tuple[float, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        # minAreaRect restituisce angoli in [-90, 0); normalizza
        if angle < -45:
            angle = 90 + angle
        pts.append((angle, area))
    if not pts:
        return _hough_angle(img_gray)
    # Media pesata per area
    total = sum(a for _, a in pts)
    return sum(ang * a for ang, a in pts) / total if total > 0 else 0.0


def _hough_angle(img_gray: Any) -> float:
    import cv2
    import numpy as np

    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for ln in lines[:, 0]:
        x1, y1, x2, y2 = ln
        if x2 - x1 == 0:
            continue
        angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if not angles:
        return 0.0
    # Le linee di testo sono ~orizzontali -> riporta la mediana in [-45, 45)
    import statistics

    return (statistics.median(angles) + 45) % 90 - 45


def rotate_image(img: Any, angle: float) -> Any:
    import cv2

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def normalize_image(img: Any) -> Any:
    """grayscale + CLAHE. Non binarizzare aggressivamente (i VLM leggono meglio il grayscale)."""
    import cv2

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# Fase
# ---------------------------------------------------------------------------


def ingest(path: Path, db: DB | None = None, settings: Settings | None = None) -> str:
    """Calcola sha256, deduplica, registra il documento. Ritorna doc_id."""
    s = settings or get_settings()
    db = db or DB(s)
    path = Path(path)
    sha = sha256_of(path)
    existing = db.find_by_sha256(sha)
    if existing:
        log.info("Documento già noto: %s (skip)", existing["doc_id"])
        return existing["doc_id"]
    doc_id = doc_id_from_sha(sha)
    db.upsert_document(doc_id, str(path), sha, n_pages=None)
    log.info("Ingested %s -> %s", path, doc_id)
    return doc_id


def rasterize(
    doc_id: str,
    db: DB | None = None,
    settings: Settings | None = None,
    degraded: bool = False,
) -> None:
    """Rasterizza tutte le pagine non ancora processate, deskew+normalizza, salva PNG."""
    from .db import skip_if_done

    s = settings or get_settings()
    db = db or DB(s)
    if skip_if_done(db, doc_id, "ingested", "rasterized"):
        log.info("Già rasterizzato: %s", doc_id)
        return
    doc = db.get_document(doc_id)

    dpi = s.dpi_degraded if degraded else s.dpi
    work = Path(s.work_dir) / doc_id
    work.mkdir(parents=True, exist_ok=True)

    import cv2  # noqa: F401  (verifica disponibilità)

    pdf = _open_pdf(Path(doc["path"]))
    n_pages = len(pdf)
    # Aggiorna n_pages
    db.conn.execute("UPDATE documents SET n_pages=? WHERE doc_id=?", (n_pages, doc_id))

    for i, page in enumerate(pdf, start=1):
        existing = db.get_page(doc_id, i)
        if existing and existing["image_path"]:
            continue  # idempotente per pagina
        out = work / f"page_{i:04d}.png"
        w, h = rasterize_page(page, dpi, out)
        # Deskew + normalizza
        img = cv2.imread(str(out), cv2.IMREAD_COLOR)
        angle = estimate_deskew_angle(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if abs(angle) > s.deskew_min_angle:
            img = rotate_image(img, angle)
        else:
            angle = 0.0
        img_norm = normalize_image(img)
        cv2.imwrite(str(out), img_norm)
        db.set_page(doc_id, i, image_path=str(out), dpi=dpi, deskew_angle=angle)
        log.info("Rasterized page %d/%d (deskew=%.2f°)", i, n_pages, angle)

    db.transition(doc_id, "ingested", "rasterized")


def sample_pages(doc_id: str, k: int = 20, db: DB | None = None) -> list[str]:
    """Punto di controllo umano: ritorna i path di k pagine a campione da ispezionare."""
    s = get_settings()
    db = db or DB(s)
    pages = db.get_pages(doc_id)
    step = max(1, len(pages) // k)
    return [p["image_path"] for p in pages[::step][:k]]


__all__ = ["ingest", "rasterize", "sample_pages", "sha256_of", "doc_id_from_sha"]
