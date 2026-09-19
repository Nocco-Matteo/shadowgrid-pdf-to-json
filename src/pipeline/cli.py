"""Orchestratore CLI.

Subcomandi: ingest, rasterize, ocr-a, ocr-b, reconcile, enumerate, extract,
validate, review, eval, export, run (end-to-end). Ogni subcomando rispetta la macchina a
stati e salta lavoro già fatto. Gestione vita dei server vLLM tra le fasi.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from . import (
    compendium,
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
from .schema import SCHEMAS, _list_inner_type
from .vllm_runner import VLLMRunner

log = logging.getLogger(__name__)


def _schema(args):
    """Schema di estrazione scelto con --schema (default da config: schema_name)."""
    return SCHEMAS[getattr(args, "schema", None) or get_settings().schema_name]


def _list_fields(schema, args) -> list[tuple[str, str | None]]:
    """Campi lista da enumerare in Fase 4 con la loro descrizione: quelli
    passati con --list-field, altrimenti tutte le liste dello schema."""
    from pydantic import BaseModel

    auto = [name for name, fi in schema.model_fields.items()
            if isinstance(_list_inner_type(fi.annotation), type)
            and issubclass(_list_inner_type(fi.annotation), BaseModel)]
    names = getattr(args, "list_fields", None) or auto
    return [(n, schema.model_fields[n].description if n in schema.model_fields else None)
            for n in names]


def _same_schema(doc_id: str, schema, db: DB) -> bool:
    """False (con errore esplicito) se il documento ha estrazioni di un altro
    schema: mescolarle darebbe un documento che nessuno dei due schemi accetta."""
    fields = {re.split(r"[.\[$]", r["field_path"], maxsplit=1)[0]
              for r in db.get_extractions(doc_id)}
    foreign = sorted(fields - set(schema.model_fields))
    if foreign:
        log.error("%s ha estrazioni di un altro schema (%s): documento saltato. "
                  "Per riestrarlo con --schema %s (OCR conservato): "
                  "python -m pipeline.cli reset-extraction --doc-id %s",
                  doc_id, ", ".join(foreign[:5]), _schema_name(schema), doc_id)
        return False
    return True


def _schema_name(schema) -> str:
    return next((k for k, v in SCHEMAS.items() if v is schema), schema.__name__)


def _enumerate(doc_id: str, schema, args, db: DB, s) -> None:
    for lf, desc in _list_fields(schema, args):
        phase4_enumerate.run(doc_id, lf, db=db, settings=s, description=desc)


def export_doc(doc_id: str, schema, db: DB, s, sources: list[str]) -> Path | None:
    """Scrive l'envelope del seed (solo elementi validati) validato contro il
    compendium. Ritorna il path, o None se lo schema non ha un seed."""
    seed_file = getattr(schema, "seed_file", None)
    if seed_file is None:
        return None
    if not sources:
        log.error("Export %s: manca --source (codice del libro, es. players_handbook)", doc_id)
        return None
    doc = phase6_validate._build_document(db.get_extractions(doc_id, status="validated"))
    seed_path = compendium.DEFAULT_SCHEMA_PATH.parents[1] / "seeds" / seed_file
    seed = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.exists() else None
    envelope, notes = compendium.export_race_traits(doc, sources, seed)
    errors = compendium.validate_envelope(envelope, seed_file)
    out_dir = Path(s.work_dir).parent / "export" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Un envelope invalido NON prende il nome buono: a valle lo caricherebbe
    # qualcuno convinto che sia un seed. Finisce accanto, marcato, e il file
    # canonico di una export precedente valida resta dov'è.
    out = out_dir / (seed_file if not errors else f"{seed_file}.invalid")
    out.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / f"{seed_file}.notes.txt").write_text("\n".join(notes + errors) + "\n",
                                                    encoding="utf-8")
    if errors:
        log.error("Export NON valido contro il compendium (%d errori): scritto in %s, "
                  "NON in %s. Errori: %s",
                  len(errors), out.name, seed_file, errors[:3])
    log.info("Export %s: %d definizioni, %d note (%s)", out, len(envelope["definitions"]),
             len(notes), out_dir / f"{seed_file}.notes.txt")
    return out


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
    url = runner.start_extractor(port=8080, phase="enumerate")
    s.extractor_url = url
    try:
        for doc_id in _resolve_doc_ids(args, db):
            if not _same_schema(doc_id, _schema(args), db):
                continue
            _enumerate(doc_id, _schema(args), args, db, s)
    finally:
        runner.stop()


def cmd_extract(args) -> None:
    s = get_settings()
    db = DB(s)
    runner = VLLMRunner(s)
    url = runner.start_extractor(port=8080, phase="extract")
    s.extractor_url = url
    try:
        for doc_id in _resolve_doc_ids(args, db):
            if not _same_schema(doc_id, _schema(args), db):
                continue
            phase5_extract.run(doc_id, _schema(args), db=db, settings=s)
    finally:
        runner.stop()


def cmd_validate(args) -> None:
    s = get_settings()
    db = DB(s)
    # I retry di Fase 6 rilanciano l'estrattore via HTTP: il server deve essere
    # vivo durante la validazione (altrimenti ogni retry è destinato a fallire
    # e ricade sul valore precedente fino a esaurire i tentativi).
    runner = VLLMRunner(s)
    s.extractor_url = runner.start_extractor(port=8080, phase="extract")
    try:
        for doc_id in _resolve_doc_ids(args, db):
            if not _same_schema(doc_id, _schema(args), db):
                continue
            phase6_validate.run(doc_id, _schema(args), db=db, settings=s)
    finally:
        runner.stop()


def cmd_review(args) -> None:
    db = DB()
    for doc_id in _resolve_doc_ids(args, db):
        phase7_review.launch_ui(doc_id, db=db, schema_strict=_schema(args))


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
    schema = _schema(args)
    doc_ids = [phase1_ingest.ingest(Path(p), db=db) for p in args.paths]
    for doc_id in doc_ids:
        phase1_ingest.rasterize(doc_id, db=db, degraded=args.degraded)

    runner = VLLMRunner(s)
    # Caricare un modello OCR da ~3-4GB per poi saltare ogni pagina costa
    # minuti di avvio per niente: si accende solo se c'è lavoro. È il caso
    # della riestrazione su un documento già OCR-ato (v. `reorder`).
    if any(db.get_status(d) in {"ingested", "rasterized"} for d in doc_ids):
        s.ocr_a_url = runner.start(s.model_a, port=8080, phase="ocr_a")
        try:
            for doc_id in doc_ids:
                phase2_ocr_a.run(doc_id, db=db, settings=s)
        finally:
            runner.stop()
    else:
        log.info("OCR A già fatto su tutti i documenti: server non avviato")

    # OCR B + reconcile (opzionale)
    if not args.skip_ocr_b:
        if any(db.get_status(d) == "ocr_a" for d in doc_ids):
            s.ocr_b_url = runner.start(s.model_b, port=8080, phase="ocr_b")
            try:
                for doc_id in doc_ids:
                    phase3_ocr_b.run(doc_id, db=db, settings=s)
            finally:
                runner.stop()
        else:
            log.info("OCR B già fatto su tutti i documenti: server non avviato")
    else:
        # senza secondo OCR: ocr_a -> reconciled
        for doc_id in doc_ids:
            if db.get_status(doc_id) == "ocr_a":
                db.transition(doc_id, "ocr_a", "reconciled")

    # Enumerate + Extract + Validate (stesso modello: un solo avvio del server).
    # La validazione avviene con il server ancora attivo: i retry di Fase 6
    # rilanciano l'estrattore via HTTP.
    doc_ids = [d for d in doc_ids if _same_schema(d, schema, db)]
    if not doc_ids:
        return
    s.extractor_url = runner.start_extractor(port=8080, phase="extract")
    try:
        for doc_id in doc_ids:
            _enumerate(doc_id, schema, args, db, s)
            phase5_extract.run(doc_id, schema, db=db, settings=s)
        for doc_id in doc_ids:
            phase6_validate.run(doc_id, schema, db=db, settings=s)
    finally:
        runner.stop()

    for doc_id in doc_ids:
        export_doc(doc_id, schema, db, s, args.sources or s.source_codes)


def cmd_reorder(args) -> None:
    """Riapplica l'ordine di lettura alle regioni già in DB (niente GPU)."""
    db = DB()
    for doc_id in _resolve_doc_ids(args, db):
        n = phase2_ocr_a.reorder(doc_id, db=db)
        if n and not args.keep_extraction:
            db.reset_extraction(doc_id)
            log.info("%s: %d pagine riordinate, estrazione azzerata "
                     "(stato reconciled): rifare da `enumerate`", doc_id, n)
        elif n:
            log.warning("%s: %d pagine riordinate ma estrazione CONSERVATA: "
                        "le citazioni e le sezioni si riferiscono al testo vecchio",
                        doc_id, n)
        else:
            log.info("%s: ordine già corretto, niente da fare", doc_id)


def cmd_reset_extraction(args) -> None:
    db = DB()
    for doc_id in _resolve_doc_ids(args, db):
        db.reset_extraction(doc_id)
        log.info("Estrazione azzerata per %s: stato reconciled (OCR conservato)", doc_id)


def cmd_export(args) -> None:
    s = get_settings()
    db = DB(s)
    for doc_id in _resolve_doc_ids(args, db):
        if _same_schema(doc_id, _schema(args), db):
            export_doc(doc_id, _schema(args), db, s, args.sources or s.source_codes)


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
    p.add_argument("--schema", choices=sorted(SCHEMAS),
                   help="schema di estrazione (default: schema_name in pipeline.yaml)")
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
    sp.add_argument("--source", dest="sources", action="append",
                    help="codice del libro per l'export (es. players_handbook), ripetibile")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser(
        "reorder",
        help="riapplica l'ordine di lettura alle regioni già in DB (niente OCR) "
             "e azzera l'estrazione")
    add_targets(sp)
    sp.add_argument("--keep-extraction", action="store_true",
                    help="non azzerare l'estrazione (sconsigliato: il testo cambia "
                         "sotto le citazioni già validate)")
    sp.set_defaults(func=cmd_reorder)

    sp = sub.add_parser("reset-extraction",
                        help="cancella estrazioni/task e torna a reconciled (OCR conservato)")
    add_targets(sp)
    sp.set_defaults(func=cmd_reset_extraction)

    sp = sub.add_parser("export", help="scrive il seed del compendium dagli elementi validati")
    add_targets(sp)
    sp.add_argument("--source", dest="sources", action="append",
                    help="codice del libro (es. players_handbook), ripetibile")
    sp.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
