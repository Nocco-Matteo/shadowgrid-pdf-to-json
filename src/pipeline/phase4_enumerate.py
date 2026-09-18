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
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

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
    section: str | None = None       # titolo della sezione (es. la razza)
    section_page: int | None = None


def estimate_tokens(text: str) -> float:
    """Stima PRUDENTE dei token (ripiego senza tokenizer): i tokenizer Qwen
    spezzano cifre e simboli uno per uno, quindi una tabella di dadi/numeri
    costa ~1 token ogni 1.2 caratteri contro ~4 della prosa. Calibrata sul
    tokenizer di Qwen3.8: non sottostima nemmeno le tabelle (margine 30%)."""
    digits = sum(c.isdigit() for c in text)
    symbols = sum(not c.isalnum() and not c.isspace() for c in text)
    letters = sum(c.isalpha() for c in text)
    return 1.3 * (digits + symbols + letters / 3.0)


@lru_cache(maxsize=2)
def _hf_tokenizer(model: str):
    from transformers import AutoTokenizer

    # dalla cache HF (vLLM ha già scaricato il modello): niente rete
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)


def token_counter(model: str) -> Callable[[str], float]:
    """Conteggio token col tokenizer dell'estrattore (dalla cache HF, la stessa
    di vLLM); se non disponibile, la stima prudente."""
    try:
        tok = _hf_tokenizer(model)
    except Exception as e:  # transformers assente, modello non in cache, offline...
        log.warning("Tokenizer di %s non disponibile (%s): stima prudente dei token", model, e)
        return estimate_tokens
    return lambda text: len(tok.encode(text, add_special_tokens=False))


def _page_windows(
    page_blocks: list[tuple[int, str]],
    max_size: float,
    size: Callable[[str], float] = len,
) -> list[tuple[list[int], str]]:
    """Finestre di pagine consecutive entro `max_size` (misurato con `size`),
    con una pagina di sovrapposizione: un titolo di sezione a fine finestra
    resta visibile agli elementi della successiva. Una pagina più grande del
    budget fa finestra da sola."""
    sizes = [size(text) + size("\n\n") for _, text in page_blocks]
    windows: list[tuple[list[int], str]] = []
    i = 0
    while i < len(page_blocks):
        nos, parts, total = [], [], 0.0
        j = i
        while j < len(page_blocks) and (not parts or total + sizes[j] <= max_size):
            nos.append(page_blocks[j][0])
            parts.append(page_blocks[j][1])
            total += sizes[j]
            j += 1
        windows.append((nos, "\n\n".join(parts)))
        if j >= len(page_blocks):
            break
        i = j - 1 if j - 1 > i else j  # sovrapposizione di una pagina
    return windows


def _check_item(db: DB, doc_id: str, it: dict, window_pages: list[int]) -> ListItem | None:
    """Valida un elemento dell'inventario contro il testo OCR."""
    anchor = it.get("anchor", "")
    page_no = it.get("page")
    if not anchor:
        return None
    if page_no not in window_pages:
        log.warning("Elemento %r su pagina %s fuori dalla finestra -> scartato", anchor, page_no)
        return None
    page_text = db.get_page_text(doc_id, page_no) or ""
    score = partial_ratio(normalize(anchor), normalize(page_text))
    if score < 90:
        log.warning("Anchor non trovata nel testo (score=%d): %r -> scartata", score, anchor)
        return None
    # region_ids validati contro le regioni reali della pagina
    valid_ids = {r["region_id"] for r in db.get_regions(doc_id, page_no, engine="a")}
    region_ids = [rid for rid in (it.get("region_ids") or []) if rid in valid_ids]
    if it.get("region_ids") and not region_ids:
        log.warning("region_ids inventati per %r: nessun id valido -> contesto pagina", anchor)
    # section: la pagina più vicina (<= quella dell'elemento) che la contiene
    section, section_page = it.get("section") or None, None
    if section:
        for pn in sorted((p for p in window_pages if p <= page_no), reverse=True):
            if partial_ratio(normalize(section), normalize(db.get_page_text(doc_id, pn) or "")) >= 90:
                section_page = pn
                break
        if section_page is None:
            log.warning("Sezione %r di %r non trovata nel testo -> ignorata", section, anchor)
            section = None
    return ListItem(anchor=anchor, page=page_no, region_ids=region_ids,
                    section=section, section_page=section_page)


def run(
    doc_id: str,
    list_field_path: str,
    db: DB | None = None,
    settings: Settings | None = None,
    client: ExtractorClient | None = None,
    expected_count: int | None = None,
    description: str | None = None,
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

    # Contesto: regioni con i loro ID, per finestre di pagine che stanno nel
    # contesto dell'estrattore (un manuale intero in un prompt lo sfora). Senza
    # gli ID visibili il modello non può selezionare region_ids in modo
    # affidabile (inventerebbe numeri plausibili).
    pages = db.get_pages(doc_id)
    page_blocks = []
    for p in pages:
        lines = [f"=== PAGE {p['page_no']} ===", "REGIONI (id | tipo | testo):"]
        for r in db.get_canonical_regions(doc_id, p["page_no"]):
            lines.append(f"[{r['region_id']}] {r['region_type']}: {r['text']}")
        page_blocks.append((p["page_no"], "\n".join(lines)))
    count = token_counter(s.extractor_model)
    # contesto dell'estrattore - risposta - istruzioni - margine
    budget = (s.enumerate_window_tokens or
              s.extractor_max_model_len - s.extractor_max_tokens - 1024)
    windows = _page_windows(page_blocks, budget, count)

    what = f" ({description})" if description else ""
    header = (
        f"Elenca SOLO gli elementi della lista '{list_field_path}'{what} presenti "
        f"nelle pagine qui sotto.\n"
        f"Per ogni elemento restituisci anchor (stringa verbatim che identifica "
        f"univocamente l'inizio dell'elemento), page (numero pagina), region_ids "
        f"(lista di id regione) e section (titolo verbatim della sezione a cui "
        f"l'elemento appartiene, null se non c'è).\n"
        f"Gli id regione sono quelli tra parentesi quadre nell'elenco REGIONI: usa SOLO "
        f"quelli elencati per la pagina dell'elemento.\n"
        f"Non estrarre i valori dei campi, solo l'inventario. Nessun elemento: items=[].\n"
        f"Output JSON: {{\"items\": [{{\"anchor\": \"...\", \"page\": 3, "
        f"\"region_ids\": [12], \"section\": \"...\"}}]}}\n\n"
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
                        "section": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                    "required": ["anchor", "page", "region_ids", "section"],
                },
            }
        },
        "required": ["items"],
    }

    # Verifica anchor contro testo OCR (fuzzy partial_ratio >= 90)
    items: list[ListItem] = []
    seen: set[tuple[str, int]] = set()
    failed_windows = 0
    for w, (page_nos, context) in enumerate(windows, start=1):
        task = f"enumerate:{list_field_path}:pagine {page_nos[0]}-{page_nos[-1]}"
        try:
            raw = client.extract(header + context, guided_json_schema=schema)
        except Exception as e:
            # una finestra fallita non ferma il documento, ma non passa in
            # silenzio: task fallito -> needs_review (niente done)
            log.error("Enumerate %s fallita: %s", task, e)
            db.record_task(doc_id, task, "failed", str(e))
            failed_windows += 1
            continue
        db.record_task(doc_id, task, "ok")
        items_raw = raw.get("items", [])
        n_before = len(items)
        for it in items_raw:
            item = _check_item(db, doc_id, it, page_nos)
            if item is None:
                continue
            key = (normalize(item.anchor), item.page)
            if key in seen:
                continue  # deduplica (anche tra finestre sovrapposte)
            seen.add(key)
            items.append(item)
        if len(windows) > 1:
            log.info("Enumerate %s: finestra %d/%d (pagine %d-%d): %d elementi",
                     list_field_path, w, len(windows), page_nos[0], page_nos[-1],
                     len(items) - n_before)

    if failed_windows:
        log.error("Enumerate %s: %d/%d finestre fallite -> needs_review",
                  list_field_path, failed_windows, len(windows))
        db.set_status(doc_id, "needs_review")

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
            [{"anchor": it.anchor, "page": it.page, "region_ids": it.region_ids,
              "section": it.section, "section_page": it.section_page}
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
