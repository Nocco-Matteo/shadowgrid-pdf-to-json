# Comandi di verifica (per Devin/agenti)

## Ambiente

Sviluppo (senza GPU): `pip install -r requirements.txt`
Target (CUDA): scommenta vllm/paddlepaddle-gpu in `requirements.txt`, poi `pip install -r requirements.txt`

## Verifica

- Lint: `ruff check src tests`
- Test: `pytest -q`
- Singolo modulo: `pytest tests/test_schema.py -q`

## Run

- End-to-end: `python -m pipeline.cli run <pdf>`
- Solo fase: `python -m pipeline.cli <subcommand> <pdf>` (v. `cli.py`)
- Valutazione gold: `python -m pipeline.cli eval`

## Note operative

- VRAM: PaddleOCR-VL ~3GB; Qwen3.8-27B AWQ 4-bit ~18GB + KV. 32GB RAM sistema →
  niente offload → fasi in serie; vLLM si spegne/riparte tra fasi. Stato su SQLite WAL.
- Le dipendenze pesanti (vllm, paddleocr, pymupdf, opencv) sono opzionali: i test
  girano con stub quando non installate.
- Gold set: annotare PRIMA di guardare l'output del modello. Split 15 dev / resto
  sigillato. Vedi `gold/README.md`.
- Metrica chiave: tasso di errore silenzioso
  (`confidence='high'` ma valore sbagliato).
