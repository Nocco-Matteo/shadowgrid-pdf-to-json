"""Orchestratore CLI.

Subcomandi: ingest, rasterize, ocr-a, ocr-b, reconcile, enumerate, extract,
validate, review, eval, run (end-to-end). Ogni subcomando rispetta la macchina a
stati e salta lavoro già fatto. Gestione vita dei server vLLM tra le fasi.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import (
    phase1_ingest,
    phase2_ocr_a,
    phase3_ocr_b,
    phase4_enumerate,
    phase5_extract,
    phase6_validate,
    phase7_review,
    phase8_eval,
)
from .config import get_settings
from .db import DB
from .schema import ContractStrict
from .vllm_runner import VLLMRunner

log = logging.getLogger(__name__)

DEFAULT_SCHEMA = ContractStrict


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def cmd_ingest(args) -> None:
    db = DB()
    for p in args.paths:
        phase1_ingest.ingest(Path(p), db=db)


def cmd_rasterize(args) -> None:
    db = DB()
    for p in args.paths:
        doc_id = phase1_ingest.ingest(Path(p), db=db)
        phase1_ingest.rasterize(doc_id, db=db, degraded=args.degraded)


def cmd_ocr_a(args) -> None:
    s = get_settings()
    db = DB(s)
    runner = VLLMRunner(s)
    url = runner.start(s.model_a, port=8080, phase="ocr_a")
    s.ocr_a_url = url
    try:
        for doc_id in _resolve_doc_ids(args, db):
            phase2_ocr_a.run(doc_id, db=db, settings=s)
    finally:
        runner.stop()


def cmd_ocr_b(args) -> None:
    s = get_settings()
    db = DB(s)
    runner = VLLMRunner(s)
    url = runner.start(s.model_b, port=8080, phase="ocr_b")
    s.ocr_b_url = url
    try:
        for doc_id in _resolve_doc_ids(args, db):
            phase3_ocr_b.run(doc_id, db=db, settings=s)
    finally:
        runner.stop()


def cmd_reconcile(args) -> None:
    # alias per ocr-b (se si vuole solo riconciliare senza rilanciare OCR B)
    cmd_ocr_b(args)


def cmd_enumerate(args) -> None:
    s = get_settings()
    db = DB(s)
    runner = VLLMRunner(s)
    url = runner.start(s.extractor_model, port=8080, phase="enumerate")
    s.extractor_url = url
    try:
        for doc_id in _resolve_doc_ids(args, db):
            for lf in args.list_fields or []:
                phase4_enumerate.run(doc_id, lf, db=db, settings=s)
    finally:
        runner.stop()


def cmd_extract(args) -> None:
    s = get_settings()
    db = DB(s)
    runner = VLLMRunner(s)
    url = runner.start(s.extractor_model, port=8080, phase="extract")
    s.extractor_url = url
    try:
        for doc_id in _resolve_doc_ids(args, db):
            phase5_extract.run(doc_id, DEFAULT_SCHEMA, db=db, settings=s)
    finally:
        runner.stop()


def cmd_validate(args) -> None:
    s = get_settings()
    db = DB(s)
    # I retry di Fase 6 rilanciano l'estrattore via HTTP: il server deve essere
    # vivo durante la validazione (altrimenti ogni retry è destinato a fallire
    # e ricade sul valore precedente fino a esaurire i tentativi).
    runner = VLLMRunner(s)
    s.extractor_url = runner.start(s.extractor_model, port=8080, phase="extract")
    try:
        for doc_id in _resolve_doc_ids(args, db):
            phase6_validate.run(doc_id, DEFAULT_SCHEMA, db=db, settings=s)
    finally:
        runner.stop()


def cmd_review(args) -> None:
    db = DB()
    for doc_id in _resolve_doc_ids(args, db):
        phase7_review.launch_ui(doc_id, db=db, schema_strict=DEFAULT_SCHEMA)


def cmd_eval(args) -> None:
    db = DB()
    prev = db.last_run()  # prima di evaluate(), che registra la run corrente
    m = phase8_eval.evaluate(args.gold_dir, db=db, split=args.split)
    phase8_eval.print_metrics(m)
    if args.compare_last and prev and prev["run_id"] != m.run_id:
        delta = phase8_eval.compare_runs(db, prev["run_id"], m.run_id)
        print("\n=== Delta vs run precedente ===")
        print(json.dumps(delta, indent=2, ensure_ascii=False))


def cmd_run(args) -> None:
    """End-to-end: ingest -> rasterize -> ocr-a -> (ocr-b) -> enumerate -> extract -> validate."""
    s = get_settings()
    db = DB(s)
    schema = DEFAULT_SCHEMA
    for p in args.paths:
        doc_id = phase1_ingest.ingest(Path(p), db=db)
        phase1_ingest.rasterize(doc_id, db=db, degraded=args.degraded)

    # OCR A
    runner = VLLMRunner(s)
    s.ocr_a_url = runner.start(s.model_a, port=8080, phase="ocr_a")
    try:
        for p in args.paths:
            doc_id = phase1_ingest.ingest(Path(p), db=db)
            phase2_ocr_a.run(doc_id, db=db, settings=s)
    finally:
        runner.stop()

    # OCR B + reconcile (opzionale)
    if not args.skip_ocr_b:
        s.ocr_b_url = runner.start(s.model_b, port=8080, phase="ocr_b")
        try:
            for p in args.paths:
                doc_id = phase1_ingest.ingest(Path(p), db=db)
                phase3_ocr_b.run(doc_id, db=db, settings=s)
        finally:
            runner.stop()
    else:
        # senza secondo OCR: ocr_a -> reconciled
        for p in args.paths:
            doc_id = phase1_ingest.ingest(Path(p), db=db)
            if db.get_status(doc_id) == "ocr_a":
                db.transition(doc_id, "ocr_a", "reconciled")

    # Enumerate + Extract + Validate (stesso modello: un solo avvio del server).
    # La validazione avviene con il server ancora attivo: i retry di Fase 6
    # rilanciano l'estrattore via HTTP.
    s.extractor_url = runner.start(s.extractor_model, port=8080, phase="extract")
    try:
        for p in args.paths:
            doc_id = phase1_ingest.ingest(Path(p), db=db)
            for lf in args.list_fields or []:
                phase4_enumerate.run(doc_id, lf, db=db, settings=s)
            phase5_extract.run(doc_id, schema, db=db, settings=s)
        for p in args.paths:
            doc_id = phase1_ingest.ingest(Path(p), db=db)
            phase6_validate.run(doc_id, schema, db=db, settings=s)
    finally:
        runner.stop()


def _resolve_doc_ids(args, db: DB) -> list[str]:
    if args.doc_ids:
        return args.doc_ids
    if args.paths:
        return [phase1_ingest.ingest(Path(p), db=db) for p in args.paths]
    # tutti i pending nello stato atteso (best-effort)
    return [r["doc_id"] for r in db.conn.execute("SELECT doc_id FROM documents").fetchall()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline", description="Pipeline estrazione PDF->JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_targets(sp):
        sp.add_argument("paths", nargs="*", help="path PDF (mutuamente esclusivo con --doc-id)")
        sp.add_argument("--doc-id", dest="doc_ids", nargs="*", help="doc_id esistenti")

    sp = sub.add_parser("ingest")
    add_targets(sp)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("rasterize")
    add_targets(sp)
    sp.add_argument("--degraded", action="store_true")
    sp.set_defaults(func=cmd_rasterize)

    sp = sub.add_parser("ocr-a")
    add_targets(sp)
    sp.set_defaults(func=cmd_ocr_a)

    sp = sub.add_parser("ocr-b")
    add_targets(sp)
    sp.set_defaults(func=cmd_ocr_b)

    sp = sub.add_parser("reconcile")
    add_targets(sp)
    sp.set_defaults(func=cmd_reconcile)

    sp = sub.add_parser("enumerate")
    add_targets(sp)
    sp.add_argument("--list-field", dest="list_fields", nargs="*")
    sp.set_defaults(func=cmd_enumerate)

    sp = sub.add_parser("extract")
    add_targets(sp)
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("validate")
    add_targets(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("review")
    add_targets(sp)
    sp.set_defaults(func=cmd_review)

    sp = sub.add_parser("eval")
    sp.add_argument("--gold-dir", default="gold")
    sp.add_argument("--split", default="sealed")
    sp.add_argument("--compare-last", action="store_true")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("run")
    sp.add_argument("paths", nargs="*")
    sp.add_argument("--degraded", action="store_true")
    sp.add_argument("--skip-ocr-b", action="store_true")
    sp.add_argument("--list-field", dest="list_fields", nargs="*")
    sp.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
