"""Fase 5 — Estrazione campo per campo.

Avvia vLLM con Qwen3.8-27B in AWQ 4-bit.
- Raggruppa i campi in task: campi con stessa fonte vanno insieme, max 5-8 per chiamata.
  Per gli elementi di lista, un task per elemento.
- Contesto ristretto: solo le regioni pertinenti + una regione di margine sopra/sotto.
  Mai il documento intero. Se non sai quali regioni siano pertinenti, aggiungi un passo
  di routing (BM25 + embedding sulle regioni, top-k); per moduli strutturati la posizione
  è di solito stabile e puoi ancorarla a un'etichetta.
- Parametri: temperature=0, seed fisso, guided_json=SchemaLoose del task, backend xgrammar.
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

from .config import Settings, get_settings
from .db import DB
from .ocr_clients import ExtractorClient
from .schema import flatten_extracted, loose

log = logging.getLogger(__name__)


@dataclass
class Task:
    name: str
    fields: list[str]  # field paths
    regions: list[dict]  # [{region_id, text, type}]
    page_no: int | None
    schema_strict: type  # BaseModel strict per il task
    images_b64: list[str] | None = None


def build_tasks(
    doc_id: str,
    schema_strict: type,
    db: DB,
    settings: Settings,
) -> list[Task]:
    """Raggruppa i campi in task. Per moduli strutturati la posizione è stabile:
    ancoriamo ogni task a un'etichetta e selezioniamo le regioni della pagina che la
    contengono + margine sopra/sotto. Implementazione iniziale semplice: un task per
    pagina con tutti i campi piatti; liste -> un task per elemento (gestito a valle)."""
    tasks: list[Task] = []
    pages = db.get_pages(doc_id)

    # Campi piatti (non lista) -> un task per documento, regioni di tutte le pagine
    flat_fields = []
    list_fields = []
    for name, fi in schema_strict.model_fields.items():
        ann = fi.annotation
        if _is_list_of_model(ann):
            list_fields.append(name)
        else:
            flat_fields.append(name)

    if flat_fields:
        all_regions: list[dict] = []
        for p in pages:
            for r in db.get_regions(doc_id, p["page_no"], engine="a"):
                all_regions.append({
                    "region_id": r["region_id"],
                    "text": r["text"],
                    "type": r["region_type"],
                    "page": p["page_no"],
                })
        tasks.append(Task(
            name="flat",
            fields=flat_fields,
            regions=all_regions,
            page_no=None,
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
            regions = []
            if page_no:
                for r in db.get_regions(doc_id, page_no, engine="a"):
                    regions.append({
                        "region_id": r["region_id"],
                        "text": r["text"],
                        "type": r["region_type"],
                        "page": page_no,
                    })
            tasks.append(Task(
                name=f"{lf}[{i}]",
                fields=[lf],
                regions=regions,
                page_no=page_no,
                schema_strict=schema_strict,
            ))

    return tasks


def _is_list_of_model(ann: Any) -> bool:
    import typing as t

    origin = t.get_origin(ann)
    if origin in (list, list):
        args = t.get_args(ann)
        return bool(args) and isinstance(args[0], type)
    return False


def select_regions_for_label(regions: list[dict], label: str, margin: int = 1) -> list[dict]:
    """Seleziona regioni pertinenti a un'etichetta + `margin` regioni sopra/sotto.
    Implementazione semplice: match case-insensitive sul testo della regione."""
    norm_label = label.lower()
    idx = next((i for i, r in enumerate(regions) if norm_label in r["text"].lower()), None)
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
    if st == "extracted":
        log.info("Estrazione già fatta: %s", doc_id)
        return
    if st not in {"enumerated", "reconciled"}:
        raise ValueError(f"Estrazione richiede enumerated, trovato {st}")

    client = client or ExtractorClient(s.extractor_url, s.extractor_model)
    tasks = build_tasks(doc_id, schema_strict, db, s)
    loose_schema = loose(schema_strict)
    guided = loose_schema.model_json_schema()

    for task in tasks:
        prompt = build_prompt(task)
        raw = client.extract(prompt, images_b64=task.images_b64, guided_json_schema=guided)
        # raw è un dict con i campi del task; appiattisci e salva tentativi
        rows = flatten_extracted(raw, prefix="")
        for row in rows:
            db.upsert_extraction(
                doc_id=doc_id,
                field_path=row["field_path"],
                value_json=row["value_json"],
                quote=row["quote"],
                page_no=row["page"],
                bbox=_parse_bbox(row["bbox"]),
                attempt=1,
                status="pending",
                confidence=row.get("confidence"),
            )
        log.info("Task %s: %d campi estratti", task.name, len(rows))

    if db.get_status(doc_id) in {"enumerated", "reconciled"}:
        db.transition(doc_id, "enumerated", "extracted")


def _parse_bbox(b: Any) -> tuple[float, float, float, float] | None:
    if b is None:
        return None
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return tuple(float(x) for x in b)
    return None


__all__ = ["run", "build_tasks", "build_prompt", "Task", "select_regions_for_label"]
