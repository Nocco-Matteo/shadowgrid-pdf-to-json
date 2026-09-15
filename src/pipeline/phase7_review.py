"""Fase 7 — Revisione umana.

Coda ordinata per rischio:
  1. confidence='low';
  2. campi che hanno richiesto retry;
  3. campi obbligatori risultati nulli.
UI minima: crop dell'immagine attorno alla bbox a sinistra, campo editabile a destra,
accetta/correggi. Gradio basta (~100 righe).
Ogni correzione umana rientra nel gold set.

Chiusura del ciclo:
  - la coda è costruita sugli ULTIMI tentativi di ogni campo: una correzione
    (nuovo tentativo validated) fa uscire di coda il vecchio;
  - include TUTTI i motivi di revisione (needs_review E rejected) più i
    conflitti OCR non risolti;
  - una correzione OCR umana invalida le estrazioni dipendenti dalla regione
    corretta (bbox sovrapposta o citazione dal testo vecchio), che rientrano
    in coda: un valore già validato sul testo sbagliato non resta valido;
  - a coda vuota il documento viene finalizzato SOLO se: esistono estrazioni,
    nessun pending, nessun task di estrazione fallito, la copertura dei campi
    dello schema è completa; poi rivalidazione SchemaStrict e done. Mai
    sintetizzare assenze per lavoro non concluso (_fill_missing non è una
    scorciatoia);
  - i callback Gradio girano su thread worker: ogni callback usa una
    connessione SQLite dedicata, mai condivisa.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Settings, get_settings
from .db import DB

log = logging.getLogger(__name__)

# Stati dell'ultimo tentativo che richiedono revisione umana
_REVIEW_STATUSES = ("needs_review", "rejected")


def review_queue(doc_id: str, db: DB | None = None, settings: Settings | None = None) -> list[dict]:
    """Coda ordinata per rischio, costruita sugli ultimi tentativi.
    Include anche i conflitti OCR non risolti (divergent_low / fallback_a)."""
    s = settings or get_settings()
    db = db or DB(s)
    items = [
        dict(r) for r in db.latest_extractions(doc_id)
        if r["status"] in _REVIEW_STATUSES and "$" not in r["field_path"]
    ]

    # Conflitti OCR irrisolti -> pseudo-item low confidence
    for c in db.get_conflicts(doc_id, resolvers=("divergent_low", "fallback_a")):
        items.append({
            "doc_id": doc_id,
            "field_path": f"region_conflict:{c['region_id']}",
            "value_json": json.dumps(c["resolved_text"], ensure_ascii=False),
            "quote": c["text_a"],
            "page_no": c["page_no"],
            "bbox": c["bbox"],
            "attempt": 1,
            "status": "needs_review",
            "confidence": "low",
        })

    def risk_key(it: dict) -> tuple[int, int]:
        conf = it.get("confidence") or "high"
        attempt = it.get("attempt") or 1
        value = json.loads(it["value_json"]) if it["value_json"] else None
        # 0: confidence low; 1: retry (>1); 2: obbligatorio nullo
        if conf == "low":
            return (0, attempt)
        if attempt > 1:
            return (1, attempt)
        if value is None:
            return (2, attempt)
        return (3, attempt)

    items.sort(key=risk_key)
    return items


def crop_around(image_path: str, bbox, margin: int = 40) -> str | None:
    """Ritaglia un crop attorno alla bbox per la UI. Ritorna il path del crop."""
    try:
        import cv2

        img = cv2.imread(image_path)
        if img is None or bbox is None:
            return None
        h, w = img.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(w, x1 + margin)
        y1 = min(h, y1 + margin)
        crop = img[y0:y1, x0:x1]
        out = str(Path(image_path).with_suffix(".crop.png"))
        cv2.imwrite(out, crop)
        return out
    except Exception as e:
        log.warning("crop fallito: %s", e)
        return None


def apply_correction(
    doc_id: str,
    field_path: str,
    new_value: str,
    db: DB | None = None,
    settings: Settings | None = None,
) -> None:
    """Applica una correzione umana: aggiorna l'estrazione (o la risoluzione
    del conflitto OCR) e la salva nel gold set delle correzioni."""
    s = settings or get_settings()
    db = db or DB(s)

    if field_path.startswith("region_conflict:"):
        try:
            region_id = int(field_path.split(":", 1)[1])
        except ValueError:
            log.error("field_path conflitto malformato: %s", field_path)
            return
        # risoluzione umana -> testo canonico aggiornato (transazione unica)
        db.resolve_conflict(doc_id, region_id, str(new_value), "human")
        # le estrazioni già validate sul testo VECCHIO non restano valide:
        # tornano in coda umana (dipendenza per bbox o per citazione)
        invalidated = _invalidate_dependent_extractions(db, doc_id, region_id)
        if invalidated:
            log.info("Estrazioni invalidate dalla correzione OCR (%s): %s",
                     region_id, invalidated)
        _save_correction(doc_id, field_path, new_value)
        return

    row = db.latest_extraction(doc_id, field_path)
    attempt = (row["attempt"] + 1) if row else 1
    db.upsert_extraction(
        doc_id, field_path, json.dumps(new_value, ensure_ascii=False),
        row["quote"] if row else None, row["page_no"] if row else None,
        _bbox(row) if row else None, attempt, "validated", "high",
    )
    _save_correction(doc_id, field_path, new_value)


def _invalidate_dependent_extractions(db: DB, doc_id: str, region_id: int) -> list[str]:
    """Riporta in coda umana (status needs_review, confidence low) le
    estrazioni che dipendono dalla regione corretta: stessa pagina e bbox
    sovrapposta, oppure citazione simile al testo (vecchio) della regione."""
    from rapidfuzz.fuzz import partial_ratio

    from .geometry import iou
    from .text_norm import normalize

    row = db.conn.execute(
        "SELECT page_no, bbox, text FROM regions WHERE region_id=?", (region_id,)
    ).fetchone()
    if row is None:
        return []
    page_no, r_text = row["page_no"], row["text"] or ""
    r_bbox = json.loads(row["bbox"]) if row["bbox"] else None

    invalidated: list[str] = []
    for e in db.latest_extractions(doc_id):
        if "$" in e["field_path"] or e["status"] != "validated":
            continue
        if e["page_no"] != page_no:
            continue
        e_bbox = json.loads(e["bbox"]) if e["bbox"] else None
        dependent = (
            (r_bbox is not None and e_bbox is not None and iou(r_bbox, e_bbox) > 0)
            or (e["quote"] is not None
                and partial_ratio(normalize(e["quote"]), normalize(r_text)) >= 90)
        )
        if dependent:
            db.upsert_extraction(
                doc_id, e["field_path"], e["value_json"], e["quote"],
                e["page_no"], tuple(e_bbox) if e_bbox else None,
                e["attempt"], "needs_review", "low",
            )
            invalidated.append(e["field_path"])
    return invalidated


def finalize_review(
    doc_id: str,
    schema_strict: type | None = None,
    db: DB | None = None,
    settings: Settings | None = None,
) -> bool:
    """Chiude il ciclo di revisione: coda vuota + nessun task fallito +
    nessun pending + copertura completa -> rivalida il documento completo e
    lo porta a done. Mai sintetizzare assenze per lavoro non concluso."""
    from .phase6_validate import (
        _build_document,
        _check_coverage,
        _fill_missing,
        gate_schema,
    )

    if schema_strict is None:
        from .schema import ContractStrict

        schema_strict = ContractStrict
    s = settings or get_settings()
    db = db or DB(s)

    st = db.get_status(doc_id)
    if st not in ("needs_review", "validated"):
        log.warning("Finalizzazione non applicabile dallo stato %s (%s)", st, doc_id)
        return False

    latest = db.latest_extractions(doc_id)
    real = [r for r in latest if "$" not in r["field_path"]]
    if not real:
        log.error("Nessuna estrazione per %s: niente done", doc_id)
        return False

    pending = [r["field_path"] for r in latest if r["status"] == "pending"]
    if pending:
        log.error("Estrazioni ancora pending per %s: %s", doc_id, pending)
        return False

    remaining = review_queue(doc_id, db, s)
    if remaining:
        log.info("Revisione di %s incompleta: %d elementi in coda", doc_id, len(remaining))
        return False

    failed = db.failed_tasks(doc_id)
    if failed:
        log.error("Task di estrazione falliti per %s: %s -> niente done",
                  doc_id, [f["task_name"] for f in failed])
        return False

    missing = _check_coverage(schema_strict, db, doc_id)
    if missing:
        log.error("Campi mai prodotti per %s: %s -> niente done",
                  doc_id, sorted(missing)[:10])
        return False

    rows = [r for r in real if r["status"] == "validated"]
    doc_dict = _build_document(rows)
    _fill_missing(schema_strict, doc_dict)
    schema_ok, err = gate_schema(schema_strict, doc_dict)
    if not schema_ok:
        log.error("Schema strict fallito sul documento %s: %s", doc_id, err)
        return False

    if st == "needs_review":
        db.transition(doc_id, "needs_review", "done")
    else:
        db.transition(doc_id, "validated", "done")
    log.info("Documento %s finalizzato: done", doc_id)
    return True


def _save_correction(doc_id: str, field_path: str, new_value: str) -> None:
    corr_dir = Path("gold/corrections")
    corr_dir.mkdir(parents=True, exist_ok=True)
    out = corr_dir / f"{doc_id}__{field_path.replace('.', '_').replace('[', '_').replace(']', '')}.json"
    out.write_text(json.dumps({
        "doc_id": doc_id, "field_path": field_path, "value": new_value,
    }, ensure_ascii=False, indent=2))
    log.info("Correzione salvata: %s", out)


def _bbox(row):
    if row and row["bbox"]:
        try:
            return tuple(json.loads(row["bbox"]))
        except json.JSONDecodeError:
            return None
    return None


def launch_ui(
    doc_id: str,
    db: DB | None = None,
    settings: Settings | None = None,
    schema_strict: type | None = None,
) -> None:
    """UI Gradio minimale (~100 righe). Crop a sinistra, campo editabile a destra.

    I callback Gradio girano su thread worker: ogni callback apre una
    connessione SQLite dedicata (mai condivisa col thread principale)."""
    try:
        import gradio as gr
    except ImportError:
        log.error("Gradio non installato. pip install gradio")
        return

    s = settings or get_settings()
    if db is not None:
        queue = review_queue(doc_id, db, s)
    else:
        db_tmp = DB(s)
        try:
            queue = review_queue(doc_id, db_tmp, s)
        finally:
            db_tmp.close()
    if not queue:
        db_fresh = DB(s)
        try:
            ok = finalize_review(doc_id, schema_strict, db_fresh, s)
        finally:
            db_fresh.close()
        print("Documento finalizzato (done)." if ok
              else "Nessun campo da revisionare, ma il documento non è finalizzabile "
                   "(task falliti o schema non valido).")
        return

    state = {"idx": 0}

    def current():
        return queue[state["idx"]] if state["idx"] < len(queue) else None

    def render():
        db = DB(s)
        try:
            it = current()
            if it is None:
                return None, "", "Finito."
            page_no = it.get("page_no")
            page = db.get_page(doc_id, page_no) if page_no else None
            crop = crop_around(page["image_path"], _bbox(it)) if page else None
            info = f"{it['field_path']} (attempt {it['attempt']}, conf {it['confidence']})"
            return crop, json.loads(it["value_json"]) if it["value_json"] else "", info
        finally:
            db.close()

    def accept(new_value):
        it = current()
        if it is not None:
            db = DB(s)
            try:
                apply_correction(doc_id, it["field_path"], new_value, db, s)
            finally:
                db.close()
        state["idx"] += 1
        if current() is None:
            # fine coda: chiudi il ciclo (rivalidazione + done)
            db = DB(s)
            try:
                finalize_review(doc_id, schema_strict, db, s)
            finally:
                db.close()
        return render()

    def skip():
        state["idx"] += 1
        return render()

    with gr.Blocks() as ui:
        gr.Markdown(f"## Revisione — {doc_id}")
        with gr.Row():
            img = gr.Image(label="Crop")
            with gr.Column():
                info = gr.Textbox(label="Campo")
                val = gr.Textbox(label="Valore")
                with gr.Row():
                    btn_ok = gr.Button("Accetta")
                    btn_skip = gr.Button("Salta")
        btn_ok.click(accept, inputs=val, outputs=[img, val, info])
        btn_skip.click(skip, outputs=[img, val, info])
        ui.load(render, outputs=[img, val, info])

    ui.launch()


__all__ = [
    "review_queue",
    "crop_around",
    "apply_correction",
    "finalize_review",
    "launch_ui",
]
