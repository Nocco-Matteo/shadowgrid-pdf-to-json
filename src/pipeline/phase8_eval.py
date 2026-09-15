"""Fase 8 — Valutazione.

Suite pytest che gira la pipeline sul gold set e produce, per ogni campo dello schema:
precision, recall, exact match rate.

La metrica che conta più di tutte: tasso di errore silenzioso — campi con un valore
sbagliato ma confidence='high'. È l'unico errore che non intercetti, quindi l'unico
che danneggia davvero la fedeltà.

Ogni run registra git_sha, modello, versione prompt. Ogni modifica va confrontata
con la run precedente sullo stesso set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .db import DB

log = logging.getLogger(__name__)


@dataclass
class FieldMetrics:
    field_path: str
    precision: float
    recall: float
    exact_match: float
    n: int
    silent_errors: int = 0


@dataclass
class RunMetrics:
    run_id: int
    fields: list[FieldMetrics] = field(default_factory=list)
    silent_error_rate: float = 0.0

    def as_dict(self) -> dict:
        return {
            "fields": [f.__dict__ for f in self.fields],
            "silent_error_rate": self.silent_error_rate,
        }


def _norm_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return str(v).strip().lower()


def evaluate(
    gold_dir: Path | str = "gold",
    db: DB | None = None,
    settings: Settings | None = None,
    split: str = "sealed",
) -> RunMetrics:
    """Valuta la pipeline sul gold set. Confronta value estratto vs value annotato."""
    s = settings or get_settings()
    db = db or DB(s)
    gold_dir = Path(gold_dir)

    split_file = gold_dir / f"{split}.txt"
    doc_ids = (
        [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        if split_file.exists() else []
    )

    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    em: dict[str, int] = {}
    n: dict[str, int] = {}
    silent: dict[str, int] = {}
    extra_total = 0

    for doc_id in doc_ids:
        ann_path = gold_dir / "annotations" / f"{doc_id}.json"
        if not ann_path.exists():
            log.warning("Annotation mancante: %s", ann_path)
            continue
        ann = json.loads(ann_path.read_text())
        ann_flat = _flatten_annotation(ann)

        # ultimo tentativo per campo; un tentativo più recente non validato
        # (rejected/needs_review) non conta come valore prodotto
        extractions = {e["field_path"]: e for e in db.latest_extractions(doc_id)}
        for fp_path, expected in ann_flat.items():
            n[fp_path] = n.get(fp_path, 0) + 1
            got_row = extractions.get(fp_path)
            got = json.loads(got_row["value_json"]) if got_row and got_row["value_json"] else None
            if got_row is not None and got_row["status"] != "validated":
                got = None
            if _norm_value(got) == _norm_value(expected):
                tp[fp_path] = tp.get(fp_path, 0) + 1
                em[fp_path] = em.get(fp_path, 0) + 1
            else:
                if got is not None:
                    fp[fp_path] = fp.get(fp_path, 0) + 1
                    # errore silenzioso se confidence='high'
                    if got_row and got_row["confidence"] == "high":
                        silent[fp_path] = silent.get(fp_path, 0) + 1
                if expected is not None:
                    fn[fp_path] = fn.get(fp_path, 0) + 1

        # campi predetti NON attesi dal gold: elementi inventati. Vanno
        # contati come falsi positivi, altrimenti precision/recall restano
        # a 1.0 anche aggiungendo campi immaginari.
        for fp_path, e in extractions.items():
            if fp_path in ann_flat or "$" in fp_path:
                continue  # atteso, o metadato (inventari Fase 4)
            if e["status"] != "validated":
                continue
            got = json.loads(e["value_json"]) if e["value_json"] else None
            if got is None:
                continue  # assenza dichiarata, non un'invenzione
            fp[fp_path] = fp.get(fp_path, 0) + 1
            extra_total += 1
            if e["confidence"] == "high":
                silent[fp_path] = silent.get(fp_path, 0) + 1

    fields: list[FieldMetrics] = []
    total_silent = 0
    total_n = 0
    for fp_path in sorted(set(n) | set(fp)):
        t = tp.get(fp_path, 0)
        f_pos = fp.get(fp_path, 0)
        f_neg = fn.get(fp_path, 0)
        prec = t / (t + f_pos) if (t + f_pos) else 1.0
        rec = t / (t + f_neg) if (t + f_neg) else 1.0
        em_rate = em.get(fp_path, 0) / n[fp_path] if n.get(fp_path) else 0.0
        sil = silent.get(fp_path, 0)
        fields.append(FieldMetrics(fp_path, prec, rec, em_rate, n.get(fp_path, 0), sil))
        total_silent += sil
        total_n += n.get(fp_path, 0)

    # denominatore: tutte le predizioni valutate (attese + inventate)
    den = total_n + extra_total
    metrics = RunMetrics(run_id=0, fields=fields,
                         silent_error_rate=total_silent / den if den else 0.0)

    # Registra run
    run_id = db.start_run(s.git_sha, s.extractor_model, prompt_version="v1")
    db.finish_run(run_id, metrics.as_dict())
    metrics.run_id = run_id
    return metrics


def _flatten_annotation(ann: dict, prefix: str = "") -> dict[str, Any]:
    """Appiattisce l'annotazione (solo i valori attesi, senza quote/bbox)."""
    out: dict[str, Any] = {}
    for k, v in ann.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict) and "value" in v:
            out[path] = v["value"]
        elif isinstance(v, list):
            for i, item in enumerate(v):
                out.update(_flatten_annotation(item, f"{path}[{i}]."))
        elif isinstance(v, dict):
            out.update(_flatten_annotation(v, f"{path}."))
        else:
            out[path] = v
    return out


def compare_runs(db: DB, run_a: int, run_b: int) -> dict:
    """Confronta due run sullo stesso set. Ritorna delta per campo."""
    ra = db.conn.execute("SELECT metrics_json FROM runs WHERE run_id=?", (run_a,)).fetchone()
    rb = db.conn.execute("SELECT metrics_json FROM runs WHERE run_id=?", (run_b,)).fetchone()
    if not ra or not rb:
        raise ValueError("Run non trovate")
    ma = json.loads(ra["metrics_json"] or "{}")
    mb = json.loads(rb["metrics_json"] or "{}")
    fa = {f["field_path"]: f for f in ma.get("fields", [])}
    fb = {f["field_path"]: f for f in mb.get("fields", [])}
    deltas = {}
    for fp in sorted(set(fa) | set(fb)):
        a, b = fa.get(fp, {}), fb.get(fp, {})
        deltas[fp] = {
            "precision": (b.get("precision", 0) - a.get("precision", 0)),
            "recall": (b.get("recall", 0) - a.get("recall", 0)),
            "exact_match": (b.get("exact_match", 0) - a.get("exact_match", 0)),
            "silent_errors": (b.get("silent_errors", 0) - a.get("silent_errors", 0)),
        }
    return {
        "delta_fields": deltas,
        "silent_error_rate": (mb.get("silent_error_rate", 0) - ma.get("silent_error_rate", 0)),
    }


def print_metrics(m: RunMetrics) -> None:
    print(f"=== Run {m.run_id} ===")
    print(f"{'field':40} {'prec':>6} {'rec':>6} {'em':>6} {'n':>4} {'silent':>6}")
    for f in m.fields:
        print(f"{f.field_path:40} {f.precision:6.2f} {f.recall:6.2f} "
              f"{f.exact_match:6.2f} {f.n:4d} {f.silent_errors:6d}")
    print(f"Silent error rate: {m.silent_error_rate:.4f}")


__all__ = ["evaluate", "compare_runs", "print_metrics", "RunMetrics", "FieldMetrics"]
