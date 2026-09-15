"""Fase 4 — Enumerazione delle liste.

Esiste perché lo schema è nidificato. Saltarla è la causa numero uno di elementi
mancanti o duplicati.

Per ogni campo di tipo lista, una chiamata dedicata che restituisce solo l'inventario:
  {"items": [{"anchor": "Contratto n. 44/B", "page": 3, "region_ids": [12, 13]}]}

- anchor è una stringa verbatim che identifica univocamente l'inizio dell'elemento.
- Verifica ogni anchor contro il testo OCR. Anchor non trovata -> elemento scartato.
- Il prompt elenca le regioni con i loro id (il modello può selezionarli);
  i region_ids restituiti vengono validati contro le regioni reali della pagina
  (un id inventato non può selezionare contesto inesistente) e PERSISTITI
  nell'inventario: la Fase 5 li usa per restringere il contesto dell'elemento.
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

    Rieseguibile anche da needs_review (es. dopo un mismatch di copertura):
    l'inventario viene ricalcolato da zero.
    """
    s = settings or get_settings()
    db = db or DB(s)
    st = db.get_status(doc_id)
    if st is None:
        raise ValueError(f"Documento {doc_id} sconosciuto")
    if st in {"extracted", "validated", "done"}:
        return []  # già oltre l'enumerazione
    if st not in {"reconciled", "enumerated", "needs_review"}:
        raise ValueError(f"Enumerate richiede reconciled, trovato {st}")

    client = client or ExtractorClient(s.extractor_url, s.extractor_model)

    # Contesto: testo canonico per pagina + regioni con i loro ID. Senza gli
    # ID visibili nel prompt il modello non può selezionare region_ids in modo
    # affidabile (inventerebbe numeri plausibili).
    pages = db.get_pages(doc_id)
    page_parts = []
    for p in pages:
        lines = [f"=== PAGE {p['page_no']} ===", "REGIONI (id | tipo | testo):"]
        for r in db.get_canonical_regions(doc_id, p["page_no"]):
            lines.append(f"[{r['region_id']}] {r['region_type']}: {r['text']}")
        page_parts.append("\n".join(lines))
    context = "\n\n".join(page_parts)
    prompt = (
        f"Elenca SOLO gli elementi della lista '{list_field_path}' presenti nel documento.\n"
        f"Per ogni elemento restituisci anchor (stringa verbatim che identifica univocamente "
        f"l'inizio dell'elemento), page (numero pagina) e region_ids (lista di id regione).\n"
        f"Gli id regione sono quelli tra parentesi quadre nell'elenco REGIONI: usa SOLO "
        f"quelli elencati per la pagina dell'elemento.\n"
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
        page_text = db.get_page_text(doc_id, page_no) if page_no else ""
        score = partial_ratio(normalize(anchor), normalize(page_text or ""))
        if score < 90:
            log.warning("Anchor non trovata nel testo (score=%d): %r -> scartata", score, anchor)
            continue
        # region_ids validati contro le regioni reali della pagina
        valid_ids: set[int] = set()
        if page_no:
            valid_ids = {r["region_id"] for r in db.get_regions(doc_id, page_no, engine="a")}
        region_ids = [rid for rid in (it.get("region_ids") or []) if rid in valid_ids]
        if it.get("region_ids") and not region_ids:
            log.warning("region_ids inventati per %r: nessun id valido -> contesto pagina",
                        anchor)
        key = normalize(anchor)
        if key in seen_anchors:
            continue  # deduplica
        seen_anchors.add(key)
        items.append(ListItem(anchor=anchor, page=page_no, region_ids=region_ids))

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
        value_json=json.dumps(
            [{"anchor": it.anchor, "page": it.page, "region_ids": it.region_ids}
             for it in items],
            ensure_ascii=False,
        ),
        quote=None,
        page_no=None,
        bbox=None,
        attempt=1,
        status="validated",
        confidence="high",
    )

    # Stato: reconciled -> enumerated (solo se non già needs_review)
    st = db.get_status(doc_id)
    if st == "reconciled":
        db.transition(doc_id, "reconciled", "enumerated")
    elif st == "needs_review" and expected_count == len(items):
        # la ripetizione ha risolto il motivo della revisione
        db.set_status(doc_id, "enumerated")

    return items


def count_pattern(text: str, pattern: str) -> int:
    """Utility per il controllo di copertura: conta occorrenze di un pattern regex."""
    return len(re.findall(pattern, text))


__all__ = ["run", "ListItem", "count_pattern"]
