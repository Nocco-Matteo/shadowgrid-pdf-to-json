# shadowgrid-pdf-to-json

Pipeline di estrazione strutturata da scansioni PDF → JSON nidificato con citazione
verificabile per ogni valore. Implementazione di `pipeline-estrazione-spec.md.pdf`.

Target: Ryzen 9 7950X, RTX 4090 (24GB), 32GB RAM. Fasi in serie (vLLM si spegne/riparte
tra fasi); stato su SQLite WAL → kill & resume sicuro.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[ocr,dev]"          # ambiente leggero (sviluppo/test)
# sulla target machine con CUDA: pip install -e ".[ocr,vllm,ui,dev]"

pytest -q                            # test (con stub per dipendenze pesanti)
ruff check src tests

python -m pipeline.cli ingest path/to/doc.pdf
python -m pipeline.cli run path/to/doc.pdf   # end-to-end
python -m pipeline.cli eval                    # valutazione sul gold set
```

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
