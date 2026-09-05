# Pipeline di estrazione strutturata da scansioni — Step di implementazione

Specifica sorgente: `pipeline-estrazione-spec.md.pdf`.
Target hardware: Ryzen 9 7950X, RTX 4090 (24GB), 32GB RAM.
Output: JSON nidificato conforme a schema, con citazione verificabile per ogni valore.

Questo documento traduce la specifica in step operativi ordinati per implementazione
**non numerica** (come da paragrafo "Ordine di implementazione consigliato"): prima lo
scheletro end-to-end più corto che produce un JSON validato, poi ispessimento.

Convenzione: ogni fase legge lo stato da SQLite e **salta** ciò che è già fatto. Il
processo deve poter essere ucciso in qualsiasi momento e ripartire senza perdere lavoro.

---

## Step 0 — Scaffold del progetto

- [0.1] Creare `pyproject.toml` (PEP 621) con dipendenze opzionali raggruppate:
  `core` (pydantic v2, pydantic-settings, rapidfuzz, sqlite-utils), `ocr`
  (pymupdf, opencv-python-headless, numpy, pillow), `vllm` (vllm, paddleocr),
  `ui` (gradio), `dev` (pytest, pytest-cov, ruff).
- [0.2] Creare `requirements.txt` pin-only per riproducibilità su target machine.
- [0.3] Package `src/pipeline/` con `__init__.py`, `config.py` (pydantic-settings:
  `PIPELINE_DB_PATH`, `PIPELINE_WORK_DIR`, `PIPELINE_VLLM_URL`, `PIPELINE_OCR_A_URL`,
  `PIPELINE_OCR_B_URL`, `PIPELINE_MODEL_A`, `PIPELINE_MODEL_B`, `PIPELINE_EXTRACTOR_MODEL`,
  seed, dpi, soglie fuzzy, retry max).
- [0.4] `AGENTS.md` con comandi di verifica (lint, test, run).
- [0.5] Struttura cartelle: `src/pipeline/`, `tests/`, `gold/`, `scripts/`, `runs/`.

**Verifica**: `python -c "import pipeline"` funziona; `pytest --collect-only` raccoglie i test.

---

## Step 1 — Schema come fonte unica di verità (`schema.py`)

- [1.1] `Extracted[T]` generico con `value: T | None`, `quote: str | None`,
  `page: int | None`, `bbox: tuple[float,float,float,float] | None`,
  `confidence: Literal["high","low"]`.
- [1.2] `model_validator(mode="after")` `quote_required_with_value`: `value` e `quote`
  devono essere entrambi `None` o entrambi presenti.
- [1.3] Helper `loose()` che da uno schema strict produce il `SchemaLoose`:
  - tutti i campi `Optional`;
  - rimuove `pattern`, `ge`, `le`, `min_length`, `max_length`, validator custom;
  - tipi semplici (per compilazione xgrammar veloce, nessun caso patologico).
- [1.4] `SchemaStrict` di esempio (contratto) con campi piatti + una lista nidificata
  (`parties: list[Party]`), per esercitare la Fase 4. Sostituibile dall'utente.
- [1.5] Funzione `dump_field_path(value, path)` per appiattire in `(field_path, value_json,
  quote, page, bbox)` da salvare in `extractions`.

**Verifica**: `tests/test_schema.py` — `Extracted(value=None, quote=None)` ok;
`Extracted(value=1, quote=None)` raise; `loose()` non contiene vincoli.

---

## Step 2 — Storage e macchina a stati (`db.py`)

- [2.1] SQLite in WAL (`PRAGMA journal_mode=WAL`, `synchronous=NORMAL`).
- [2.2] Tabelle:
  - `documents(doc_id PK, path, sha256, n_pages, status, created_at)`;
  - `pages(doc_id, page_no, image_path, dpi, deskew_angle, full_text)`;
  - `regions(region_id PK, doc_id, page_no, bbox, region_type, text, engine, order_idx)`;
  - `region_conflicts(region_id, text_a, text_b, resolved_text, resolver)`;
  - `extractions(doc_id, field_path, value_json, quote, page_no, bbox, attempt, status)`;
  - `runs(run_id PK, git_sha, model, prompt_version, started_at, finished_at, metrics_json)`.
- [2.3] Macchina a stati `ingested → rasterized → ocr_a → ocr_b → reconciled →
  enumerated → extracted → validated → done` con rami `needs_review`, `failed`.
  Funzione `transition(doc_id, src, dst)` con assert sugli archi ammessi.
- [2.4] API di alto livello: `upsert_document`, `set_page`, `add_regions`,
  `get_regions(page)`, `upsert_extraction`, `get_pending(status)`, `start_run`, `finish_run`.
- [2.5] Helper `skip_if_done(status_required, next_status)` decoratore per le fasi.

**Verifica**: `tests/test_db_state.py` — inserimento, transizioni lecite/illecite,
idempotenza (ri-esecuzione fase non duplica righe).

---

## Step 3 — Utility geometriche e normalizzazione testo

- [3.1] `geometry.py`: `rotate_bbox(bbox, angle, img_w, img_h)`,
  `invert_deskew_bbox(bbox, deskew_angle, ...)` per riproiettare le bbox OCR sulle
  coordinate dell'immagine originale.
- [3.2] `text_norm.py`: `normalize(s)` → lowercase, collassa spazi, unifica punteggiatura,
  rimuove diacritici. Usato dal fuzzy match di Fase 6 e dalla riconciliazione Fase 3.

**Verifica**: `tests/test_geometry.py` (rotazione idempotente con angolo opposto),
`tests/test_text_norm.py` (casi notevoli).

---

## Step 4 — Fase 1: Ingestione e rasterizzazione (`phase1_ingest.py`)

- [4.1] `ingest(path)`: calcola `sha256` → `doc_id`; se esiste in DB, skip (deduplica).
- [4.2] `rasterize(doc_id, dpi=300)`: PyMuPDF `page.get_pixmap(dpi=300)`, salva PNG in
  `work_dir/<doc_id>/page_<n>.png`. Registra `pages.image_path`, `dpi`.
- [4.3] `deskew(img)`: stima angolo con `cv2.minAreaRect` sul contorno del testo
  binarizzato (fallback Hough). Ruota solo se `|angle| > 0.3°`. Salva `deskew_angle`.
- [4.4] `normalize_image(img)`: grayscale + `cv2.createCLAHE`. **Non** binarizzare
  aggressivamente (i VLM leggono meglio il grayscale).
- [4.5] Punto di controllo umano: funzione `sample_pages(doc_id, k=20)` che apre un
  montage a schermo per ispezione visiva (utility, non bloccante).
- [4.6] Stato: `ingested → rasterized`.

**Verifica**: su un PDF di test (se PyMuPDF disponibile) produce PNG e riga `pages`.
Altrimenti test con mock che verifica la logica di skip per sha256 già presente.

---

## Step 5 — Fase 2: OCR primario PaddleOCR-VL (`phase2_ocr_a.py`, `ocr_clients.py`, `vllm_runner.py`)

- [5.1] `vllm_runner.py`: `start_vllm(model, extra_args)` / `stop_vllm()` via subprocess,
  health-check su `/v1/models` con timeout. Log in `runs/vllm_<phase>.log`.
- [5.2] `ocr_clients.py`: client HTTP sincrono per PaddleOCR-VL via vLLM server
  (`PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url=...)`). Wrapper
  `ocr_page_a(image_path) -> list[Region]` con `region_type` (text/table/formula/stamp),
  `bbox`, `text`, `order_idx`.
- [5.3] `phase2_ocr_a.run(doc_id)`: per ogni pagina chiama il client, riproietta le bbox
  invertendo il deskew, scrive in `regions` con `engine='a'`, concatena in `pages.full_text`
  in ordine di lettura. Tabelle conservate come struttura (JSON in `regions.text`).
- [5.4] Stato: `rasterized → ocr_a`.

**Verifica**: test con client mock che restituisce regioni fisse; si verifica scrittura
su `regions` e `full_text` ordinato.

---

## Step 6 — Fase 5: Estrazione campo per campo (`phase5_extract.py`)  *(prima della 3 e 4 nello scheletro)*

- [6.1] `build_tasks(schema)`: raggruppa campi per fonte condivisa, max 5–8 per task;
  un task per elemento di lista.
- [6.2] `select_regions(doc_id, page, anchor_label)`: contesto ristretto — regioni
  pertinenti + una regione di margine sopra/sotto. Se non noto, routing BM25+embedding
  (stub iniziale: selezione per etichetta/anchor).
- [6.3] `build_prompt(regions, fields)`: ordine — regioni con `region_id`, definizioni
  campi, regola "quote copiata carattere per carattere", istruzione esplicita che
  `null` è la risposta corretta quando il dato non è presente.
- [6.4] `extract_task(task)`: vLLM con `temperature=0`, seed fisso,
  `guided_json=SchemaLoose(task)`, backend xgrammar. Salva tentativi in `extractions`
  con `attempt` crescente.
- [6.5] Stato: `enumerated → extracted` (la Fase 4 va fatta prima in produzione, ma
  nello scheletro iniziale si estrae su campi piatti senza liste).

**Verifica**: test con LLM stub che ritorna JSON guidato; si verifica scrittura
`extractions` con `attempt=1`.

---

## Step 7 — Fase 6: Verifica e validazione (`phase6_validate.py`)

- [7.1] **Cancello grounding**: `rapidfuzz.fuzz.partial_ratio(norm(quote),
  norm(page_full_text)) >= 90`. Sotto soglia → allucinato, scarta.
- [7.2] **Cancello coerenza value↔quote** (`check_value_quote`):
  - numeri/date → token numerici dalla quote, `value` è normalizzazione;
  - enum → quote contiene il termine mappato;
  - stringhe libere → `value` è sottostringa normalizzata della quote.
- [7.3] **Cancello schema**: `SchemaStrict.model_validate()`.
- [7.4] **Retry**: se 6.1 o 6.3 falliscono, rilancia con errore accodato al prompt,
  max 2 tentativi. Al terzo → `status='needs_review'`.
- [7.5] **Localizzazione**: cerca `quote` nelle regioni della pagina, prendi la bbox
  della regione contenitore, salvala in `extractions.bbox`.
- [7.6] Stato: `extracted → validated` (o `needs_review`).

**Verifica**: `tests/test_validate.py` con casi: grounding alto/basso, coerenza
ok/fail, schema ok/fail, retry count, localizzazione bbox.

---

## Step 8 — Fase 8: Valutazione (`phase8_eval.py`, `tests/test_eval_gold.py`)

- [8.1] Suite pytest che gira la pipeline sul gold set e produce, per campo:
  precision, recall, exact match rate.
- [8.2] Metrica chiave: **tasso di errore silenzioso** — campi con valore sbagliato
  ma `confidence='high'`.
- [8.3] Ogni run registra `git_sha`, modello, versione prompt; confronto con run
  precedente sullo stesso set (funzione `compare_runs(run_a, run_b)`).
- [8.4] `gold/README.md` con protocollo di annotazione (annotare **prima** di guardare
  l'output del modello; split 15 dev / resto sigillato).

**Verifica**: `pytest tests/test_eval_gold.py` gira (con gold stub) e produce report.

---

## Step 9 — Fase 4: Enumerazione liste (`phase4_enumerate.py`)

- [9.1] Per ogni campo lista, chiamata dedicata che restituisce
  `{"items":[{"anchor","page","region_ids"}]}`.
- [9.2] Verifica ogni `anchor` contro il testo OCR (fuzzy). Anchor non trovata →
  elemento scartato (era inventato).
- [9.3] Controllo di copertura indipendente: conta righe tabella / occorrenze pattern
  numerazione / intestazioni sezione. Se il conteggio non torna → documento in
  revisione (non procedere in silenzio).
- [9.4] Deduplica per anchor normalizzata.
- [9.5] Stato: `reconciled → enumerated`.

**Verifica**: test con anchor presenti/assenti e conteggio discordante.

---

## Step 10 — Fase 6.2 già in Step 7; qui aggiungiamo solo il test di regressione

- [10.1] Caso "citazione autentica, interpretazione sbagliata" nel gold set.

---

## Step 11 — Fase 3: Secondo OCR e riconciliazione (`phase3_ocr_b.py`)

- [11.1] Spegni server A, avvia DeepSeek-OCR 2 (o dots.ocr) — `engine='b'`.
- [11.2] Allineamento segmentazioni: primo passo per IoU bbox (soglia 0.5); non
  appaiate → allineamento per contenuto con `rapidfuzz`.
- [11.3] Confronto similarità normalizzata: `>= 0.95` accetta testo A; `< 0.95`
  conflitto.
- [11.4] Risoluzione: ritaglia bbox unione + ~10px margine, upscale 2×, rileggi con
  terzo passaggio a piena risoluzione. Maggioranza 2-su-3. Se tutti divergono →
  `confidence='low'` e coda umana.
- [11.5] Scrivi `region_conflicts`. Stato: `ocr_b → reconciled`.
- [11.6] Attesa: conflitti > ~5% del testo → problema a monte (Fase 1), non in OCR.
  Funzione `conflict_rate(doc_id)` con warning.

**Verifica**: test con due OCR mock che divergono su una regione; si verifica
risoluzione 2-su-3 e scrittura conflitto.

---

## Step 12 — Fase 7: Revisione umana (`phase7_review.py`)

- [12.1] Coda ordinata per rischio: `confidence='low'` → campi con retry → campi
  obbligatori nulli.
- [12.2] UI Gradio (~100 righe): crop immagine attorno alla bbox a sinistra, campo
  editabile a destra, accetta/correggi.
- [12.3] Ogni correzione rientra nel gold set (scritta in `gold/corrections/`).
- [12.4] Stato: `needs_review → done` (o `failed`).

**Verifica**: test che la coda è ordinata correttamente; UI si avvia (smoke test).

---

## Step 13 — Orchestratore CLI (`cli.py`, `scripts/run_pipeline.py`)

- [13.1] `cli.py` con subcomandi: `ingest`, `rasterize`, `ocr-a`, `ocr-b`, `reconcile`,
  `enumerate`, `extract`, `validate`, `review`, `eval`, `run` (end-to-end).
- [13.2] Ogni subcomando rispetta la macchina a stati e salta lavoro già fatto.
- [13.3] Gestione vita dei server vLLM: start/stop tra fasi (VRAM: A ~3GB,
  Qwen3.8-27B AWQ 4-bit ~18GB+KV; 32GB RAM sistema → niente offload → fasi in serie).
- [13.4] `--resume` default true (è il comportamento normale); `--force` per riprocessare.
- [13.5] Logging strutturato su stderr + `runs/` file.

**Verifica**: `python -m pipeline.cli --help` elenca i subcomandi; dry-run su PDF
mock (con dipendenze pesanti sostituite da stub se assenti).

---

## Step 14 — Verifica finale

- [14.1] `ruff check src tests` pulito.
- [14.2] `pytest -q` verde (test con stub per le dipendenze pesanti non installabili
  nell'ambiente di sviluppo).
- [14.3] `AGENTS.md` aggiornato con comandi di build/test/run e note VRAM.
- [14.4] Commit iniziale.

---

## Note trasversali

- **Idempotenza**: ogni fase legge lo stato e salta. Non duplicare righe.
- **VRAM**: fasi in serie, vLLM si spegne/riparte tra fasi. Stato su SQLite.
- **Quote obbligatoria**: senza `quote` niente verifica; senza `bbox` niente revisione
  umana efficiente. Lo `Extracted[T]` lo impone a livello di tipo.
- **Null esplicito**: il prompt deve autorizzare il modello a non rispondere.
- **Gold set**: annotare prima di guardare l'output del modello; split 15 dev / resto
  sigillato fino alla valutazione finale.
- **Metrica che conta**: tasso di errore silenzioso
  (`confidence='high'` ma valore sbagliato).
