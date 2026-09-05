"""Fase 4 — Enumerazione delle liste.

Esiste perché lo schema è nidificato. Saltarla è la causa numero uno di elementi
mancanti o duplicati.

Per ogni campo di tipo lista, una chiamata dedicata che restituisce solo l'inventario:
  {"items": [{"anchor": "Contratto n. 44/B", "page": 3, "region_ids": [12, 13]}]}

- anchor è una stringa verbatim che identifica univocamente l'inizio dell'elemento.
- Verifica ogni anchor contro il testo OCR. Anchor non trovata -> elemento scartato.
- Controllo di copertura indipendente: conta righe tabella / occorrenze pattern
  numerazione / intestazioni sezione. Se il conteggio non torna -> revisione.
- Deduplica per anchor normalizzata.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from rapidfuzz.fuzz import partial_ratio

from .config import Settings, get_settings
from .db import DB
from .ocr_clients import ExtractorClient
from .text_norm import normalize

log = logging.getLogger(__name__)


@dataclass
class ListItem:
    anchor: str
    page: int
    region_ids: list[int]


def run(
    doc_id: str,
    list_field_path: str,
    db: DB | None = None,
    settings: Settings | None = None,
    client: ExtractorClient | None = None,
    expected_count: int | None = None,
) -> list[ListItem]:
    """Enumera gli elementi di un campo lista. Ritorna gli item validati.

    `expected_count` è il controllo di copertura indipendente (es. numero di righe
    di una tabella contato a parte). Se fornito e discordante -> il documento va
    in needs_review (la funzione ritorna gli item trovati ma logga un warning e
    imposta lo stato a needs_review).
    """
    s = settings or get_settings()
    db = db or DB(s)
    if db.get_status(doc_id) != "reconciled":
        # ammesso anche da 'enumerated' per ri-esecuzione
        st = db.get_status(doc_id)
        if st not in {"reconciled", "enumerated"}:
            raise ValueError(f"Enumerate richiede reconciled, trovato {st}")

    client = client or ExtractorClient(s.extractor_url, s.extractor_model)

    # Costruisci il prompt di enumerazione usando il full_text di tutte le pagine
    pages = db.get_pages(doc_id)
    context = "\n\n".join(
        f"=== PAGE {p['page_no']} ===\n{p['full_text'] or ''}" for p in pages
    )
    prompt = (
        f"Elenca SOLO gli elementi della lista '{list_field_path}' presenti nel documento.\n"
        f"Per ogni elemento restituisci anchor (stringa verbatim che identifica univocamente "
        f"l'inizio dell'elemento), page (numero pagina) e region_ids (lista di id regione).\n"
        f"Non estrarre i valori dei campi, solo l'inventario.\n"
        f"Output JSON: {{\"items\": [{{\"anchor\": \"...\", \"page\": 3, \"region_ids\": [12]}}]}}\n\n"
        f"{context}\n"
    )
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "anchor": {"type": "string"},
                        "page": {"type": "integer"},
                        "region_ids": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["anchor", "page"],
                },
            }
        },
        "required": ["items"],
    }
    raw = client.extract(prompt, guided_json_schema=schema)
    items_raw = raw.get("items", [])

    # Verifica anchor contro testo OCR (fuzzy partial_ratio >= 90)
    items: list[ListItem] = []
    seen_anchors: set[str] = set()
    for it in items_raw:
        anchor = it.get("anchor", "")
        page_no = it.get("page")
        if not anchor:
            continue
        page_row = db.get_page(doc_id, page_no) if page_no else None
        full_text = page_row["full_text"] if page_row else ""
        score = partial_ratio(normalize(anchor), normalize(full_text or ""))
        if score < 90:
            log.warning("Anchor non trovata nel testo (score=%d): %r -> scartata", score, anchor)
            continue
        key = normalize(anchor)
        if key in seen_anchors:
            continue  # deduplica
        seen_anchors.add(key)
        items.append(ListItem(anchor=anchor, page=page_no, region_ids=it.get("region_ids", [])))

    # Controllo di copertura indipendente
    if expected_count is not None and expected_count != len(items):
        log.error(
            "Copertura discordante per %s: attesi %d, trovati %d -> needs_review",
            list_field_path, expected_count, len(items),
        )
        db.set_status(doc_id, "needs_review")

    # Salva l'inventario come estrazione speciale (field_path = list_field_path + "$inventory")
    db.upsert_extraction(
        doc_id=doc_id,
        field_path=f"{list_field_path}$inventory",
        value_json=json.dumps([{"anchor": it.anchor, "page": it.page} for it in items],
                              ensure_ascii=False),
        quote=None,
        page_no=None,
        bbox=None,
        attempt=1,
        status="validated",
        confidence="high",
    )

    # Stato: reconciled -> enumerated (solo se non già needs_review)
    if db.get_status(doc_id) == "reconciled":
        db.transition(doc_id, "reconciled", "enumerated")

    return items


def count_pattern(text: str, pattern: str) -> int:
    """Utility per il controllo di copertura: conta occorrenze di un pattern regex."""
    return len(re.findall(pattern, text))


__all__ = ["run", "ListItem", "count_pattern"]
