"""Fase 5 — Estrazione campo per campo.

Avvia vLLM con Qwen3.8-27B in AWQ 4-bit.
- Raggruppa i campi in task: campi con stessa fonte vanno insieme, max 5-8 per chiamata.
  Per gli elementi di lista, un task per elemento.
- Contesto ristretto: solo le regioni pertinenti + una regione di margine sopra/sotto.
  Mai il documento intero. Se non sai quali regioni siano pertinenti, aggiungi un passo
  di routing (BM25 + embedding sulle regioni, top-k); per moduli strutturati la posizione
  è di solito stabile e puoi ancorarla a un'etichetta.
- Parametri: temperature=0, seed fisso, output vincolato allo SchemaLoose del task (response_format json_schema).
- Struttura del prompt, in quest'ordine:
    1. le regioni, ciascuna con il suo region_id;
    2. la definizione dei campi da estrarre;
    3. la regola che quote va copiata carattere per character dal testo fornito;
    4. l'istruzione esplicita che null è la risposta corretta quando il dato non è presente.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .config import Settings, get_settings
from .db import DB
from .ocr_clients import ExtractorClient
from .schema import _list_inner_type, field_anchors, flatten_extracted, guided_schema
from .text_norm import normalize

log = logging.getLogger(__name__)


@dataclass
class Task:
    name: str
    fields: list[str]  # field paths
    regions: list[dict]  # [{region_id, text, type}]
    page_no: int | None
    schema_strict: type  # BaseModel strict per il task
    images_b64: list[str] | None = None
    item_field: str | None = None   # campo lista (task di un singolo elemento)
    item_index: int | None = None
    anchor: str | None = None


def _regions_payload(db: DB, doc_id: str, page_no: int) -> list[dict]:
    # testo canonico (riconciliato): è quello che vedrà anche la validazione
    return [
        {"region_id": r["region_id"], "text": r["text"],
         "type": r["region_type"], "page": page_no}
        for r in db.get_canonical_regions(doc_id, page_no)
    ]


def _field_page(db: DB, doc_id: str, pages, labels: list[str]) -> int | None:
    """Pagina della prima regione che contiene un'etichetta del campo (anchor
    euristica). Le etichette si provano in ordine di priorità."""
    targets = [t for t in (normalize(label) for label in labels) if t]
    for target in targets:
        for p in pages:
            for r in db.get_canonical_regions(doc_id, p["page_no"]):
                if target in normalize(r["text"] or ""):
                    return p["page_no"]
    return None


def _select_by_ids(regions: list[dict], ids: list[int], margin: int = 1) -> list[dict]:
    """Regioni con region_id in `ids` + `margin` adiacenti (per posizione)."""
    if not ids:
        return regions
    pos = {r["region_id"]: i for i, r in enumerate(regions)}
    keep: set[int] = set()
    for rid in ids:
        if rid in pos:
            keep.update(range(max(0, pos[rid] - margin),
                              min(len(regions), pos[rid] + margin + 1)))
    return [regions[i] for i in sorted(keep)] or regions


def _select_for_fields(regions: list[dict], labels_by_field: list[list[str]],
                       margin: int = 1) -> list[dict]:
    """Unione delle selezioni per-etichetta; fallback: tutte le regioni passate."""
    keep: set[int] = set()
    for labels in labels_by_field:
        found = next((lb for lb in labels
                      if any(normalize(lb) in normalize(r["text"] or "") for r in regions)),
                     labels[-1])
        sel = select_regions_for_label(regions, found, margin)
        ids = {r["region_id"] for r in sel}
        keep.update(i for i, r in enumerate(regions) if r["region_id"] in ids)
    return [regions[i] for i in sorted(keep)] or regions


def build_tasks(
    doc_id: str,
    schema_strict: type,
    db: DB,
    settings: Settings,
) -> list[Task]:
    """Contesto ristretto: mai il documento intero quando si può ancorare.

    - Campi piatti: raggruppati per pagina-anchor (nome campo trovato nel testo
      di una regione), max `max_fields_per_task` per chiamata. Campi non
      ancorabili -> contesto di tutte le pagine (fallback loggato).
    - Liste: un task per elemento dell'inventario di Fase 4, ristretto ai
      region_ids dichiarati + una regione di margine sopra/sotto.
    """
    tasks: list[Task] = []
    pages = db.get_pages(doc_id)
    s = settings

    flat_fields = []
    list_fields = []
    for name, fi in schema_strict.model_fields.items():
        if _is_list_of_model(fi.annotation):
            list_fields.append(name)
        else:
            flat_fields.append(name)

    # Campi piatti: gruppi per pagina anchor, poi chunk per max_fields_per_task
    by_page: dict[int | None, list[str]] = {}
    for f in flat_fields:
        by_page.setdefault(_field_page(db, doc_id, pages, field_anchors(schema_strict, f)),
                           []).append(f)

    for page_no, fields in sorted(by_page.items(), key=lambda kv: (kv[0] is None, kv[0])):
        if page_no is None:
            log.warning("Campi non ancorabili a una pagina: %s -> contesto intero", fields)
            regions = [r for p in pages for r in _regions_payload(db, doc_id, p["page_no"])]
        else:
            regions = _select_for_fields(_regions_payload(db, doc_id, page_no),
                                         [field_anchors(schema_strict, f) for f in fields])
        for i in range(0, len(fields), s.max_fields_per_task):
            chunk = fields[i:i + s.max_fields_per_task]
            tasks.append(Task(
                name=f"flat_p{page_no}_{i}",
                fields=chunk,
                regions=regions,
                page_no=page_no,
                schema_strict=schema_strict,
            ))

    # Liste: un task per elemento (richiede Fase 4 già eseguita)
    for lf in list_fields:
        inv_row = db.latest_extraction(doc_id, f"{lf}$inventory")
        if inv_row is None:
            log.warning("Inventario mancante per %s, salto", lf)
            continue
        items = json.loads(inv_row["value_json"] or "[]")
        for i, it in enumerate(items):
            page_no = it.get("page")
            page_regions = _regions_payload(db, doc_id, page_no) if page_no else []
            regions = _select_by_ids(page_regions, it.get("region_ids") or [])
            tasks.append(Task(
                name=f"{lf}[{i}]",
                fields=[lf],
                regions=regions,
                page_no=page_no,
                schema_strict=schema_strict,
                item_field=lf,
                item_index=i,
                anchor=it.get("anchor"),
            ))

    return tasks


def _is_list_of_model(ann: Any) -> bool:
    import typing as t

    origin = t.get_origin(ann)
    if origin in (list, list):
        args = t.get_args(ann)
        return bool(args) and isinstance(args[0], type)
    return False


def _task_missing_fields(task: Task, rows: list[dict], schema_strict: type) -> set[str]:
    """Campi che il task doveva produrre e che NON appaiono nella risposta.

    L'assenza è valida solo se DICHIARATA (foglio con value=null, che produce
    una riga con value_json=None): un campo semplicemente omesso è lavoro
    incompleto, non assenza — non deve diventare un Extracted nullo via
    _fill_missing in Fase 6."""
    if task.item_field is None:
        expected = set(task.fields)
        got = {r["field_path"] for r in rows}
    else:
        inner = _list_inner_type(schema_strict.model_fields[task.item_field].annotation)
        expected = set(inner.model_fields) if inner and issubclass(inner, BaseModel) else set()
        prefix = f"{task.item_field}[{task.item_index}]."
        got = {r["field_path"][len(prefix):] for r in rows
               if r["field_path"].startswith(prefix)}
    return expected - got


def select_regions_for_label(regions: list[dict], label: str, margin: int = 1) -> list[dict]:
    """Seleziona regioni pertinenti a un'etichetta + `margin` regioni sopra/sotto.
    Implementazione semplice: match sul testo normalizzato della regione."""
    norm_label = normalize(label)
    idx = next((i for i, r in enumerate(regions)
                if norm_label and norm_label in normalize(r["text"] or "")), None)
    if idx is None:
        return regions  # fallback: tutte
    lo = max(0, idx - margin)
    hi = min(len(regions), idx + margin + 1)
    return regions[lo:hi]


def build_prompt(task: Task) -> str:
    lines = ["REGIONI (id | tipo | testo):"]
    for r in task.regions:
        lines.append(f"[{r['region_id']}] {r['type']}: {r['text']}")
    lines.append("")
    if task.anchor:
        lines.append(f"ELEMENTO: {task.anchor}")
        lines.append("Estrai SOLO i campi di questo elemento della lista.")
        lines.append("")
    lines.append("CAMPI DA ESTRARRE:")
    for f in task.fields:
        lines.append(f"- {f}")
    lines.append("")
    lines.append("REGOLE:")
    lines.append("- per ogni campo restituisci un oggetto {value, quote, page, bbox, confidence}.")
    lines.append("- quote DEVE essere copiata carattere per carattere dal testo fornito sopra.")
    lines.append("- null è la risposta corretta quando il dato non è presente: "
                 "restituisci value=null E quote=null in quel caso.")
    lines.append("- confidence: 'high' se la quote è nitida, 'low' se ambigua.")
    return "\n".join(lines)


def run(
    doc_id: str,
    schema_strict: type,
    db: DB | None = None,
    settings: Settings | None = None,
    client: ExtractorClient | None = None,
) -> None:
    s = settings or get_settings()
    db = db or DB(s)
    st = db.get_status(doc_id)
    if st in {"extracted", "validated", "done"}:
        log.info("Estrazione già fatta: %s", doc_id)
        return
    if st not in {"enumerated", "reconciled", "needs_review"}:
        raise ValueError(f"Estrazione richiede enumerated, trovato {st}")

    client = client or ExtractorClient(s.extractor_url, s.extractor_model)
    tasks = build_tasks(doc_id, schema_strict, db, s)
    page_nos = {p["page_no"] for p in db.get_pages(doc_id)}

    failed = False
    for task in tasks:
        if db.task_status(doc_id, task.name) == "ok":
            continue  # task già completato (resume)
        prompt = build_prompt(task)
        # Schema dei soli campi del task, tutti obbligatori: il modello non
        # può omettere una chiave, solo dichiararne l'assenza (value=null).
        guided = guided_schema(schema_strict, task.fields,
                               single_item=task.item_field is not None)
        try:
            raw = client.extract(prompt, images_b64=task.images_b64,
                                 guided_json_schema=guided)
        except Exception as e:
            # esito del task persistito: un fallimento non è mai "campo assente"
            db.record_task(doc_id, task.name, "failed", str(e))
            log.error("Task %s fallito: %s", task.name, e)
            failed = True
            continue
        # Per i task di elemento lista il modello risponde con lo schema intero:
        # prendiamo il (primo) elemento del campo lista e lo appiattiamo con il
        # path indicizzato corretto (es. parties[2].name).
        payload, prefix = raw, ""
        if task.item_field is not None:
            item_list = raw.get(task.item_field) or []
            if len(item_list) > 1:
                log.warning("Task %s: attesi 1 elemento, ricevuti %d; uso il primo",
                            task.name, len(item_list))
            payload = item_list[0] if item_list else {}
            prefix = f"{task.item_field}[{task.item_index}]."
        rows = flatten_extracted(payload, prefix=prefix)
        # Copertura: il task deve rispondere a TUTTI i suoi campi. Una risposta
        # parziale è un fallimento, non "campo assente": l'assenza è valida
        # solo se dichiarata (riga con value=null).
        missing = _task_missing_fields(task, rows, schema_strict)
        if missing:
            db.record_task(doc_id, task.name, "failed",
                           f"campi omessi (assenza non dichiarata): {sorted(missing)}")
            log.error("Task %s: risposta incompleta, campi omessi: %s",
                      task.name, sorted(missing))
            failed = True
            continue
        for row in rows:
            # Le regioni nel prompt non riportano la pagina: se il modello non
            # la dà (o ne inventa una), vale quella del task (anchor/inventario).
            page = row["page"] if row["page"] in page_nos else task.page_no
            if page is None and len(page_nos) == 1:
                page = next(iter(page_nos))
            db.upsert_extraction(
                doc_id=doc_id,
                field_path=row["field_path"],
                value_json=row["value_json"],
                quote=row["quote"],
                page_no=page,
                bbox=_parse_bbox(row["bbox"]),
                attempt=1,
                status="pending",
                confidence=row.get("confidence"),
            )
        db.record_task(doc_id, task.name, "ok")
        log.info("Task %s: %d campi estratti", task.name, len(rows))

    if failed:
        # nessun avanzamento implicito: i task falliti restano visibili
        db.set_status(doc_id, "needs_review")
        return

    st = db.get_status(doc_id)
    if st in {"enumerated", "reconciled"}:
        db.transition(doc_id, st, "extracted")
    # da needs_review lo stato resta invariato: la chiusura spetta alla Fase 6
    # (percorso needs_review), che rivalida i pending e applica gli stessi
    # controlli di completezza della revisione umana. Spingere qui a
    # 'extracted' bypasserebbe la coda (es. campi rejected non risolti).


def _parse_bbox(b: Any) -> tuple[float, float, float, float] | None:
    if b is None:
        return None
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return tuple(float(x) for x in b)
    return None


__all__ = ["run", "build_tasks", "build_prompt", "Task", "select_regions_for_label"]
