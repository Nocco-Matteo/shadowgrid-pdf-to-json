"""Fase 7 — Revisione umana.

Coda ordinata per rischio:
  1. confidence='low';
  2. campi che hanno richiesto retry;
  3. campi obbligatori risultati nulli.
UI minima: crop dell'immagine attorno alla bbox a sinistra, campo editabile a destra,
accetta/correggi. Gradio basta (~100 righe).
Ogni correzione umana rientra nel gold set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Settings, get_settings
from .db import DB

log = logging.getLogger(__name__)


def review_queue(doc_id: str, db: DB | None = None, settings: Settings | None = None) -> list[dict]:
    """Coda ordinata per rischio. Ritorna liste di dict con i campi da revisionare."""
    s = settings or get_settings()
    db = db or DB(s)
    rows = db.get_extractions(doc_id, status="needs_review")
    items = [dict(r) for r in rows]

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
    """Applica una correzione umana: aggiorna l'estrazione e la salva nel gold set."""
    s = settings or get_settings()
    db = db or DB(s)
    row = db.latest_extraction(doc_id, field_path)
    attempt = (row["attempt"] + 1) if row else 1
    db.upsert_extraction(
        doc_id, field_path, json.dumps(new_value, ensure_ascii=False),
        row["quote"] if row else None, row["page_no"] if row else None,
        _bbox(row) if row else None, attempt, "validated", "high",
    )
    # Salva nel gold set delle correzioni
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


def launch_ui(doc_id: str, db: DB | None = None, settings: Settings | None = None) -> None:
    """UI Gradio minimale (~100 righe). Crop a sinistra, campo editabile a destra."""
    try:
        import gradio as gr
    except ImportError:
        log.error("Gradio non installato. pip install gradio")
        return

    s = settings or get_settings()
    db = db or DB(s)
    queue = review_queue(doc_id, db, s)
    if not queue:
        print("Nessun campo da revisionare.")
        return

    state = {"idx": 0}

    def current():
        return queue[state["idx"]] if state["idx"] < len(queue) else None

    def render():
        it = current()
        if it is None:
            return None, "", "Finito."
        page_no = it.get("page_no")
        page = db.get_page(doc_id, page_no) if page_no else None
        crop = crop_around(page["image_path"], _bbox(it)) if page else None
        info = f"{it['field_path']} (attempt {it['attempt']}, conf {it['confidence']})"
        return crop, json.loads(it["value_json"]) if it["value_json"] else "", info

    def accept(new_value):
        it = current()
        if it is not None:
            apply_correction(doc_id, it["field_path"], new_value, db, s)
        state["idx"] += 1
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


__all__ = ["review_queue", "crop_around", "apply_correction", "launch_ui"]
