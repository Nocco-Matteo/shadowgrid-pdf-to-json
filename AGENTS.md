# Comandi di verifica (per Devin/agenti)

## Ambiente

Sviluppo (senza GPU): `pip install -r requirements.txt`
Target (CUDA): scommenta vllm/paddlepaddle-gpu in `requirements.txt`, poi `pip install -r requirements.txt`

## Verifica

- Lint: `ruff check src tests`
- Test: `pytest -q`
- Singolo modulo: `pytest tests/test_schema.py -q`
- Smoke test e2e (senza GPU, client stub): `pytest tests/test_smoke_e2e.py -q`
- Smoke test REALE (target WSL2+GPU, modelli veri): `python scripts/smoke_real.py [--skip-ocr-b] [--pdf file.pdf]`

## Run

- End-to-end: `python -m pipeline.cli run <pdf>`
- Solo fase: `python -m pipeline.cli <subcommand> <pdf>` (v. `cli.py`)
- Valutazione gold: `python -m pipeline.cli eval`

## Note operative

- VRAM: PaddleOCR-VL-1.6 ~3GB; DeepSeek-OCR-2 ~4GB; Qwen3.8-27B AWQ 4-bit ~18GB + KV.
  32GB RAM sistema → niente offload → fasi in serie; vLLM si spegne/riparte tra fasi.
  Stato su SQLite WAL.
- Target: Windows + RTX 4090, ma **tutto gira dentro WSL2** (vLLM non supporta
  Windows nativo). Setup: `setup_windows.ps1` → `setup_inner.sh` in Ubuntu-22.04.
- Configurazione: `pipeline.yaml` (nomi modelli, URL, soglie). Precedenza:
  argomento > env `PIPELINE_*` > `.env` > `pipeline.yaml` > default.
  File alternativo: `PIPELINE_CONFIG=/path/altro.yaml`.
  Modelli (ID HuggingFace verificati):
    PaddlePaddle/PaddleOCR-VL-1.6 (OCR primario)
    deepseek-ai/DeepSeek-OCR-2 (OCR secondario)
    nicosuter/Qwen3.8-27B-AWQ (estrattore)
  Pre-scarica con: huggingface-cli download <model_id>
- Le dipendenze pesanti (vllm, paddleocr, pymupdf, opencv) sono opzionali: i test
  girano con stub quando non installate.
- Gold set: annotare PRIMA di guardare l'output del modello. Split 15 dev / resto
  sigillato. Vedi `gold/README.md`.
- Metrica chiave: tasso di errore silenzioso
  (`confidence='high'` ma valore sbagliato).
