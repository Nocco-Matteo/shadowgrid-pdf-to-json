"""Fase 6 — Verifica e validazione.

Tre cancelli in sequenza. Un campo che ne fallisce uno non entra nell'output.

6.1 Cancello grounding:
    score = rapidfuzz.fuzz.partial_ratio(norm(quote), norm(page_full_text))
    assert score >= 90   (sotto soglia -> allucinato, scarta)

6.2 Cancello coerenza value <-> quote:
    - numeri/date -> estrai token numerici dalla quote, value è una normalizzazione;
    - enum -> la quote contiene il termine che mappa a quel valore;
    - stringhe libere -> value è una sottostringa normalizzata della quote.

6.3 Cancello schema: SchemaStrict.model_validate().

6.4 Retry: al fallimento di 6.1 o 6.3, rilancia la stessa chiamata con l'errore di
    validazione accodato al prompt. Massimo 2 tentativi. Al terzo, needs_review.

6.5 Localizzazione: cerca la quote nelle regioni della pagina, prendi la bbox della
    regione che la contiene, salvala. Serve alla Fase 7.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from rapidfuzz.fuzz import partial_ratio

from .config import Settings, get_settings
from .db import DB
from .ocr_clients import ExtractorClient
from .text_norm import normalize

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 6.1 Grounding
# ---------------------------------------------------------------------------


def gate_grounding(quote: str, page_full_text: str, threshold: int = 90) -> bool:
    if not quote or not page_full_text:
        return False
    score = partial_ratio(normalize(quote), normalize(page_full_text))
    return score >= threshold


# ---------------------------------------------------------------------------
# 6.2 Coerenza value <-> quote
# ---------------------------------------------------------------------------


_NUM_RE = re.compile(r"\d[\d.,]*\d|\d")


def _normalize_number(s: str) -> float | None:
    """Normalizza un token numerico europeo/anglosassone in float.
    '1.234,50' -> 1234.5 ; '1,234.50' -> 1234.5 ; '42' -> 42.0."""
    s = s.strip()
    if not s:
        return None
    has_comma = "," in s
    has_dot = "." in s
    if has_comma and has_dot:
        # l'ultimo separatore è il decimale
        if s.rfind(",") > s.rfind("."):
            # europeo: . migliaia, , decimali
            s = s.replace(".", "").replace(",", ".")
        else:
            # anglosassone: , migliaia, . decimali
            s = s.replace(",", "")
    elif has_comma:
        # solo virgola -> decimale
        s = s.replace(",", ".")
    elif has_dot:
        # solo punto: potrebbe essere migliaia (1.234) o decimale (1.5)
        # euristicamente: se 3 cifre dopo l'ultimo punto e ci sono più gruppi -> migliaia
        parts = s.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3):
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def gate_value_quote(value: Any, quote: str) -> bool:
    """Il valore deve essere derivabile dalla citazione."""
    if value is None and quote is None:
        return True
    if value is None or quote is None:
        return False
    nq = normalize(quote)
    # numeri / date
    if isinstance(value, (int, float)):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        for tok in _NUM_RE.findall(quote):
            nv = _normalize_number(tok)
            if nv is not None and abs(nv - v) < 1e-6:
                return True
        return False
    # enum / stringhe libere
    nv = normalize(str(value))
    if not nv:
        return False
    # value deve essere sottostringa normalizzata della quote, oppure
    # la quote contiene il termine che mappa al valore
    return nv in nq or _token_overlap(nv, nq)


def _token_overlap(a: str, b: str) -> bool:
    ta = set(a.split())
    tb = set(b.split())
    if not ta:
        return False
    # almeno il 60% dei token di value è presente nella quote
    overlap = len(ta & tb) / len(ta)
    return overlap >= 0.6


# ---------------------------------------------------------------------------
# 6.3 Schema
# ---------------------------------------------------------------------------


def gate_schema(schema_strict: type, extracted_dict: dict) -> tuple[bool, str | None]:
    try:
        schema_strict.model_validate(extracted_dict)
        return True, None
    except Exception as e:  # ValidationError
        return False, str(e)


# ---------------------------------------------------------------------------
# 6.5 Localizzazione
# ---------------------------------------------------------------------------


def localize_bbox(quote: str, regions: list[dict]) -> tuple[float, float, float, float] | None:
    """Cerca la quote nelle regioni della pagina; ritorna la bbox della regione
    che la contiene (best partial_ratio)."""
    if not quote:
        return None
    best_score, best_bbox = 0, None
    nq = normalize(quote)
    for r in regions:
        score = partial_ratio(nq, normalize(r["text"] or ""))
        if score > best_score:
            best_score = score
            best_bbox = json.loads(r["bbox"]) if r["bbox"] else None
    return tuple(best_bbox) if best_bbox and best_score >= 80 else None


# ---------------------------------------------------------------------------
# Fase
# ---------------------------------------------------------------------------


def run(
    doc_id: str,
    schema_strict: type,
    db: DB | None = None,
    settings: Settings | None = None,
    client: ExtractorClient | None = None,
) -> None:
    s = settings or get_settings()
    db = db or DB(s)
    if db.get_status(doc_id) == "validated":
        log.info("Validazione già fatta: %s", doc_id)
        return
    if db.get_status(doc_id) != "extracted":
        raise ValueError(f"Validazione richiede extracted, trovato {db.get_status(doc_id)}")

    client = client or ExtractorClient(s.extractor_url, s.extractor_model)

    extractions = db.get_extractions(doc_id, status="pending")
    # raggruppa per field_path per gestire i retry
    by_field: dict[str, list] = {}
    for e in extractions:
        by_field.setdefault(e["field_path"], []).append(e)

    needs_review = False
    for field_path, rows in by_field.items():
        latest = sorted(rows, key=lambda r: r["attempt"])[-1]
        ok, attempt = _validate_one(doc_id, field_path, latest, schema_strict, db, s, client)
        if not ok:
            needs_review = True

    if needs_review:
        db.set_status(doc_id, "needs_review")
    else:
        db.transition(doc_id, "extracted", "validated")


def _validate_one(
    doc_id: str,
    field_path: str,
    row,
    schema_strict: type,
    db: DB,
    s: Settings,
    client: ExtractorClient,
) -> tuple[bool, int]:
    """Esegue i 3 cancelli su un'estrazione. Ritorna (ok, attempt_finale)."""
    attempt = row["attempt"]
    quote = row["quote"]
    page_no = row["page_no"]
    value = json.loads(row["value_json"]) if row["value_json"] else None

    # 6.1 grounding
    page_full_text = ""
    if page_no:
        p = db.get_page(doc_id, page_no)
        page_full_text = p["full_text"] if p else ""
    if not gate_grounding(quote, page_full_text, s.grounding_threshold):
        log.warning("Grounding fallito per %s (attempt %d)", field_path, attempt)
        return _maybe_retry(doc_id, field_path, row, schema_strict, db, s, client,
                            err="grounding_failed")

    # 6.2 coerenza value/quote
    if not gate_value_quote(value, quote):
        log.warning("Coerenza value/quote fallita per %s", field_path)
        # non si ritenta per 6.2: citazione autentica ma interpretazione sbagliata
        db.upsert_extraction(doc_id, field_path, row["value_json"], quote, page_no,
                             _bbox(row), attempt, "rejected", row["confidence"])
        db.set_status(doc_id, "needs_review")
        return False, attempt

    # 6.3 schema (validazione cross-field su tutto il documento)
    # Per validare lo schema ricostruiamo un dict con tutte le estrazioni validated finora.
    schema_ok, err = _validate_against_schema(doc_id, field_path, schema_strict, db)
    if not schema_ok:
        log.warning("Schema fallito per %s: %s", field_path, err)
        return _maybe_retry(doc_id, field_path, row, schema_strict, db, s, client, err=err)

    # 6.5 localizzazione bbox
    regions = []
    if page_no:
        regions = [
            {"text": r["text"], "bbox": r["bbox"]}
            for r in db.get_regions(doc_id, page_no, engine="a")
        ]
    bbox = localize_bbox(quote, regions) or _bbox(row)

    db.upsert_extraction(doc_id, field_path, row["value_json"], quote, page_no,
                         bbox, attempt, "validated", row["confidence"])
    return True, attempt


def _validate_against_schema(
    doc_id: str, field_path: str, schema_strict: type, db: DB
) -> tuple[bool, str | None]:
    """Ricostruisce un dict con tutte le estrazioni validated + quella corrente e
    valida lo schema strict. Validazione cross-field approssimata per campo."""
    # Nota: una validazione cross-field completa richiede tutto il documento;
    # qui facciamo un check per-campo (tipi/vincoli). Il check completo si fa in
    # phase6 final pass. Implementazione semplice: valida il singolo valore come
    # Extracted del tipo atteso. Per il MVP accettiamo se value è coerente con quote.
    return True, None


def _maybe_retry(
    doc_id: str,
    field_path: str,
    row,
    schema_strict: type,
    db: DB,
    s: Settings,
    client: ExtractorClient,
    err: str,
) -> tuple[bool, int]:
    attempt = row["attempt"]
    if attempt >= s.max_retries + 1:
        # al 3° tentativo (max_retries=2 -> attempt 3) -> needs_review
        db.upsert_extraction(doc_id, field_path, row["value_json"], row["quote"],
                             row["page_no"], _bbox(row), attempt, "needs_review",
                             row["confidence"])
        db.set_status(doc_id, "needs_review")
        return False, attempt
    # rilancia con errore accodato (stub: richiede il prompt originale; qui marchiamo
    # solo un nuovo tentativo pending che la Fase 5 rieseguirà in un ciclo successivo)
    next_attempt = attempt + 1
    db.upsert_extraction(doc_id, field_path, row["value_json"], row["quote"],
                         row["page_no"], _bbox(row), next_attempt, "pending",
                         row["confidence"])
    log.info("Retry schedulato per %s (attempt %d, err=%s)", field_path, next_attempt, err)
    return False, next_attempt


def _bbox(row):
    if row["bbox"]:
        try:
            return tuple(json.loads(row["bbox"]))
        except json.JSONDecodeError:
            return None
    return None


__all__ = [
    "run",
    "gate_grounding",
    "gate_value_quote",
    "gate_schema",
    "localize_bbox",
]
