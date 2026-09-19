# shadowgrid-pdf-to-json

Pipeline di estrazione strutturata da scansioni PDF → JSON nidificato con citazione
verificabile per ogni valore. Implementazione di `pipeline-estrazione-spec.md.pdf`.

Target: Windows + Ryzen 9 7950X + RTX 4090 (24GB), 32GB RAM — **tutto gira dentro WSL2** (vLLM non
supporta Windows nativo; setup via `setup_windows.ps1` → `setup_inner.sh`).
Fasi in serie (vLLM si spegne/riparte tra fasi); stato su SQLite WAL → kill & resume
sicuro.

## Configurazione

`pipeline.yaml` — nomi dei modelli (ID HuggingFace), endpoint vLLM, soglie, DPI.
Precedenza: argomento > env `PIPELINE_*` > `.env` > `pipeline.yaml` > default.
File alternativo: `PIPELINE_CONFIG=/path/altro.yaml`.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # sviluppo (pesanti CUDA commentati)
# sulla target machine: scommenta vllm/paddlepaddle-gpu in requirements.txt
# poi: pip install -r requirements.txt

pytest -q                            # test (con stub per dipendenze pesanti)
ruff check src tests

python -m pipeline.cli ingest path/to/doc.pdf
python -m pipeline.cli run path/to/doc.pdf --source players_handbook   # end-to-end + export
python -m pipeline.cli export --doc-id <doc_id> --source players_handbook
python -m pipeline.cli eval                    # valutazione sul gold set

# manutenzione: riapplica l'ordine di lettura alle regioni già in DB (niente
# GPU) e azzera l'estrazione; poi `run` rifà solo Fase 4-6, senza ricaricare
# i modelli OCR. Serve dopo un cambio di geometry.reading_order.
python -m pipeline.cli reorder --doc-id <doc_id>
```

Lo schema di default è `race_traits`: estrae i tratti razziali che cambiano un
numero e scrive `runs/export/<doc_id>/raceTraits.json` nel formato di
`seeds/raceTraits.json`, validato contro `schema/compendium.schema.json` (solo
gli elementi validati; esclusi e motivi in `raceTraits.json.notes.txt`).
`--schema contract` usa il contratto di esempio dello smoke test.

## Documenti

- `IMPLEMENTATION_STEPS.md` — step dettagliati per fase.
- `pipeline-estrazione-spec.md.pdf` — specifica sorgente.
- `AGENTS.md` — comandi di verifica e note operative.

## Layout

```
src/pipeline/   codice (schema, db, fasi 1-8, cli)
tests/          pytest (schema, db, geometry, text_norm, validate, eval)
gold/           gold set + protocollo di annotazione
scripts/        entrypoint alternativi
runs/           log run e vLLM (generato)
```

## Stato

Macchina a stati: `ingested → rasterized → ocr_a → ocr_b → reconciled →
enumerated → extracted → validated → done` con rami `needs_review`, `failed`.
Ogni fase è idempotente.
