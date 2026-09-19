"""Fase 6 — Verifica e validazione.

Tre cancelli in sequenza. Un campo che ne fallisce uno non entra nell'output.

6.1 Cancello grounding:
    score = rapidfuzz.fuzz.partial_ratio(norm(quote), norm(page_full_text))
    assert score >= 90   (sotto soglia -> allucinato, scarta)

6.2 Cancello coerenza value <-> quote (conservativo):
    - numeri/date -> token numerici CON SEGNO della quote, value è una normalizzazione;
    - enum -> la quote contiene il termine che mappa a quel valore;
    - stringhe libere -> value è una sottostringa normalizzata della quote.
    Nessun overlap permissivo: meglio la coda umana dell'errore silenzioso.
    Assenza dichiarata (value=null E quote=null): validata senza cancelli.

6.3 Cancello schema: SchemaStrict.model_validate().

6.4 Retry: al fallimento di 6.1 o 6.3, rilancia la stessa chiamata con l'errore di
    validazione accodato al prompt. Massimo 2 tentativi. Al terzo, needs_review.

6.5 Localizzazione: cerca la quote nelle regioni della pagina, prendi la bbox della
    regione che la contiene, salvala. Serve alla Fase 7.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel
from rapidfuzz.fuzz import partial_ratio

from .config import Settings, get_settings
from .db import DB
from .ocr_clients import ExtractorClient
from .schema import BASE_WALKING_SPEED, _is_extracted_model, _list_inner_type, guided_schema
from .text_norm import normalize

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 6.1 Grounding
# ---------------------------------------------------------------------------

# Varianti Unicode del segno meno (− U+2212, – U+2013, — U+2014): prima di
# estrarre i numeri vanno ridotte al meno ASCII, altrimenti "42" verrebbe
# accettato contro "−42".
_UNI_MINUS = re.compile(r"[\u2212\u2013\u2014]")
# Numero con segno (anche staccato: "- 42") — il segno fa parte del valore.
_NUM_RE = re.compile(r"[-+]?\s*\d[\d.,]*")
# Token contenente cifre (identificativi, date, importi): "44/B", "2024-03-15",
# "1.234,50", "ABC123". Vanno confrontati ESATTAMENTE, non fuzzy.
_IDENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,/\\-]*[A-Za-z0-9]|[A-Za-z0-9]")


def _signed_numbers(text: str) -> list[float]:
    """Token numerici con segno, normalizzati. '−42' e '- 42' -> -42.0."""
    t = _UNI_MINUS.sub("-", text)
    out: list[float] = []
    for tok in _NUM_RE.findall(t):
        nv = _normalize_number(tok.replace(" ", ""))
        if nv is not None:
            out.append(nv)
    return out


def _identifier_tokens(text: str) -> list[str]:
    """Token con almeno una cifra: numeri, date, identificativi."""
    t = _UNI_MINUS.sub("-", text)
    return [tok for tok in _IDENT_RE.findall(t) if any(c.isdigit() for c in tok)]


def gate_grounding(quote: str, page_full_text: str, threshold: int = 90) -> bool:
    """La citazione deve essere (quasi) verbatim nel testo della pagina.

    Tre controlli:
    1. lunghezza: una quote più lunga dell'intera pagina non può essere
       verbatim (chiude l'inversione di partial_ratio);
    2. fuzzy sul contesto (partial_ratio >= threshold): tollera rumore OCR
       sulle parole;
    3. corrispondenza ESATTA dei token numerici/identificativi: il fuzzy sul
       contesto non basta — "CONTRATTO N. 44/B" e "CONTRATTO N. 45/B" hanno
       score alto. Il segno fa parte del numero: una quote con "-42" non è
       verificata da una pagina che dice "42" (e viceversa)."""
    if not quote or not page_full_text:
        return False
    nq = normalize(quote)
    nft = normalize(page_full_text)
    if len(nq) > len(nft):
        return False  # non può essere verbatim se è più lunga della pagina
    if partial_ratio(nq, nft) < threshold:
        return False
    # token con cifre: devono apparire tali e quali nel testo della pagina
    for ident in _identifier_tokens(quote):
        if normalize(ident) not in nft:
            return False
    # numeri con segno: ogni numero della quote deve esistere nella pagina
    q_nums = _signed_numbers(quote)
    if q_nums:
        p_nums = set(_signed_numbers(page_full_text))
        if not all(nv in p_nums for nv in q_nums):
            return False
    return True


# ---------------------------------------------------------------------------
# 6.2 Coerenza value <-> quote
# ---------------------------------------------------------------------------


def _normalize_number(s: str) -> float | None:
    """Normalizza un token numerico europeo/anglosassone in float.
    '1.234,50' -> 1234.5 ; '1,234.50' -> 1234.5 ; '42' -> 42.0."""
    s = s.strip()
    if not s:
        return None
    has_comma = "," in s
    has_dot = "." in s
    if has_comma and has_dot:
        # l'ultimo separatore è il decimale
        if s.rfind(",") > s.rfind("."):
            # europeo: . migliaia, , decimali
            s = s.replace(".", "").replace(",", ".")
        else:
            # anglosassone: , migliaia, . decimali
            s = s.replace(",", "")
    elif has_comma:
        # solo virgola -> decimale
        s = s.replace(",", ".")
    elif has_dot:
        # solo punto: potrebbe essere migliaia (1.234) o decimale (1.5)
        # euristicamente: se 3 cifre dopo l'ultimo punto e ci sono più gruppi -> migliaia
        parts = s.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3):
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _numbers_in(value: Any) -> list[float]:
    """Valori numerici (non booleani) di una struttura annidata; le chiavi no."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers_in(v)]
    if isinstance(value, list):
        return [n for v in value for n in _numbers_in(v)]
    return []


def _expected_numbers(value: Any) -> list[set[float]]:
    """Per ogni numero della struttura, i valori che possono testimoniarlo nella
    citazione. Di norma solo il numero stesso.

    Eccezione: le derivazioni che lo schema CHIEDE esplicitamente al modello.
    `speed_bonus.amount` è per contratto la differenza da BASE_WALKING_SPEED
    ("35 feet" -> 5), quindi pretendere di trovare "5" nella citazione
    boccerebbe per forza ogni tratto di velocità: il cancello rifiuterebbe il
    modello per averlo obbedito. Qui si accetta la differenza O il valore
    assoluto, che restano entrambi ancorati alla citazione — non è un
    allentamento, è la stessa verifica applicata alla forma giusta.
    """
    if isinstance(value, dict):
        out: list[set[float]] = []
        derived = value.get("kind") == "speed_bonus"
        for k, v in value.items():
            if (derived and k == "amount"
                    and isinstance(v, (int, float)) and not isinstance(v, bool)):
                out.append({float(v), float(v) + BASE_WALKING_SPEED})
            else:
                out.extend(_expected_numbers(v))
        return out
    if isinstance(value, list):
        return [c for v in value for c in _expected_numbers(v)]
    return [{n} for n in _numbers_in(value)]


def gate_value_quote(value: Any, quote: str) -> bool:
    """Il valore deve essere derivabile dalla citazione. Controlli CONSERVATIVI:

    - numeri: il valore (segno compreso) deve comparire come token numerico
      della citazione, con le varianti del segno gestite ("−42", "- 42").
      `42` NON è accettabile se la citazione dice `-42`.
    - stringhe/enum: il valore normalizzato deve essere contenuto nella
      citazione normalizzata. Nessun overlap permissivo di token: un valore
      più lungo della citazione ("Mario Rossi Verdi" vs "Mario Rossi") è
      un'invenzione, non un'interpretazione.

    Meglio un falso positivo in coda umana che un errore silenzioso."""
    if value is None and quote is None:
        return True
    if value is None or quote is None:
        return False
    # valori strutturati (es. effects del compendium): sono un'interpretazione
    # della regola, non una sua copia. Controllo conservativo sui soli numeri:
    # ognuno deve comparire nella citazione (un "35 feet" -> speed_bonus 5 è
    # derivato e finisce in revisione umana, non passa in silenzio).
    if isinstance(value, (list, dict)):
        q_nums = _signed_numbers(quote)
        return all(any(abs(nv - n) < 1e-6 for nv in q_nums for n in cand)
                   for cand in _expected_numbers(value))
    # numeri / date
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        return any(abs(nv - v) < 1e-6 for nv in _signed_numbers(quote))
    # enum / stringhe libere
    nv = normalize(str(value))
    if not nv:
        return False
    return nv in normalize(quote)


# ---------------------------------------------------------------------------
# 6.3 Schema
# ---------------------------------------------------------------------------


def gate_schema(schema_strict: type, extracted_dict: dict) -> tuple[bool, str | None]:
    try:
        schema_strict.model_validate(extracted_dict)
        return True, None
    except Exception as e:  # ValidationError
        return False, str(e)


# ---------------------------------------------------------------------------
# 6.5 Localizzazione
# ---------------------------------------------------------------------------


def localize_bbox(quote: str, regions: list[dict]) -> tuple[float, float, float, float] | None:
    """Cerca la quote nelle regioni della pagina; ritorna la bbox della regione
    che la contiene (best partial_ratio)."""
    if not quote:
        return None
    best_score, best_bbox = 0, None
    nq = normalize(quote)
    for r in regions:
        score = partial_ratio(nq, normalize(r["text"] or ""))
        if score > best_score:
            best_score = score
            best_bbox = json.loads(r["bbox"]) if r["bbox"] else None
    return tuple(best_bbox) if best_bbox and best_score >= 80 else None


# ---------------------------------------------------------------------------
# Fase
# ---------------------------------------------------------------------------


def _schema_leaf_paths(schema: type, db: DB, doc_id: str, prefix: str = "") -> set[str]:
    """Field path delle foglie Extracted attese per il documento, con le
    lunghezze delle liste prese dagli inventari di Fase 4."""
    out: set[str] = set()
    for name, fi in schema.model_fields.items():
        ann = fi.annotation
        inner = _list_inner_type(ann)
        if inner is not None and isinstance(inner, type) and issubclass(inner, BaseModel):
            inv = db.latest_extraction(doc_id, f"{prefix}{name}$inventory")
            n = len(json.loads(inv["value_json"] or "[]")) if inv else 0
            for i in range(n):
                out |= _schema_leaf_paths(inner, db, doc_id, f"{prefix}{name}[{i}].")
        elif _is_extracted_model(ann):
            out.add(f"{prefix}{name}")
        elif isinstance(ann, type) and issubclass(ann, BaseModel):
            out |= _schema_leaf_paths(ann, db, doc_id, f"{prefix}{name}.")
    return out


def _check_coverage(schema_strict: type, db: DB, doc_id: str) -> set[str]:
    """Campi attesi mai prodotti da nessun tentativo: lavoro incompleto,
    NON assenza (l'assenza dichiarata è una riga con value=null)."""
    expected = _schema_leaf_paths(schema_strict, db, doc_id)
    present = {r["field_path"] for r in db.latest_extractions(doc_id)
               if "$" not in r["field_path"]}
    return expected - present


def run(
    doc_id: str,
    schema_strict: type,
    db: DB | None = None,
    settings: Settings | None = None,
    client: ExtractorClient | None = None,
) -> None:
    s = settings or get_settings()
    db = db or DB(s)
    st = db.get_status(doc_id)
    if st in ("validated", "done"):
        log.info("Validazione già fatta: %s", doc_id)
        return
    if st == "needs_review":
        # ripresa: rivalida i pending, poi chiudi con la logica della revisione
        # (coda vuota + task ok + schema) o resta in needs_review
        _run_from_needs_review(doc_id, schema_strict, db, s, client)
        return
    if st != "extracted":
        raise ValueError(f"Stato atteso extracted, trovato {st}")

    client = client or ExtractorClient(s.extractor_url, s.extractor_model)

    extractions = db.get_extractions(doc_id, status="pending")
    # Guardia: zero estrazioni reali (es. risposta {} dell'estrattore) non è
    # "documento senza dati", è un'estrazione fallita. I null espliciti
    # producono righe, quindi qui zero righe = qualcosa si è rotto a monte.
    real_rows = [r for r in db.get_extractions(doc_id) if "$" not in r["field_path"]]
    if not real_rows:
        log.error("Nessuna estrazione per %s: estrazione fallita o risposta vuota",
                  doc_id)
        db.set_status(doc_id, "needs_review")
        return
    # raggruppa per field_path per gestire i retry
    by_field: dict[str, list] = {}
    for e in extractions:
        by_field.setdefault(e["field_path"], []).append(e)

    needs_review = False
    for field_path, rows in by_field.items():
        if "$" in field_path:
            continue  # metadati (es. inventari di Fase 4)
        latest = sorted(rows, key=lambda r: r["attempt"])[-1]
        ok, attempt = _validate_one(doc_id, field_path, latest, schema_strict, db, s, client)
        if not ok:
            needs_review = True

    if needs_review:
        db.set_status(doc_id, "needs_review")
        return

    # Copertura: campi mai prodotti NON diventano assenze via _fill_missing.
    missing = _check_coverage(schema_strict, db, doc_id)
    if missing:
        log.error("Campi mai prodotti per %s (task incompleti?): %s",
                  doc_id, sorted(missing)[:10])
        db.set_status(doc_id, "needs_review")
        return

    # 6.3 sul documento completo: ricostruisci il dict e valida SchemaStrict.
    # _fill_missing ora non sintetizza nulla di significativo: la copertura
    # è stata verificata sopra (serve solo a completare la struttura).
    doc_dict = _build_document(db.get_extractions(doc_id, status="validated"))
    _fill_missing(schema_strict, doc_dict)
    schema_ok, err = gate_schema(schema_strict, doc_dict)
    if not schema_ok:
        log.error("Schema strict fallito sul documento %s: %s", doc_id, err)
        db.set_status(doc_id, "needs_review")
        return
    db.transition(doc_id, "extracted", "validated")


def _run_from_needs_review(
    doc_id: str,
    schema_strict: type,
    db: DB,
    s: Settings,
    client: ExtractorClient | None,
) -> None:
    """Rivalidazione di un documento in needs_review (es. dopo retry di Fase 5
    o correzioni umane): processa i pending e chiude il ciclo con gli stessi
    criteri di completezza della revisione umana."""
    client = client or ExtractorClient(s.extractor_url, s.extractor_model)
    extractions = db.get_extractions(doc_id, status="pending")
    by_field: dict[str, list] = {}
    for e in extractions:
        by_field.setdefault(e["field_path"], []).append(e)
    for field_path, rows in by_field.items():
        if "$" in field_path:
            continue
        latest = sorted(rows, key=lambda r: r["attempt"])[-1]
        _validate_one(doc_id, field_path, latest, schema_strict, db, s, client)

    from .phase7_review import finalize_review

    if not finalize_review(doc_id, schema_strict, db=db, settings=s):
        log.info("Documento %s resta in needs_review", doc_id)


def _compendium_def(schema_strict: type, field_path: str) -> str | None:
    """Nome del `$defs` del compendium a cui il campo è vincolato, se c'è.

    I campi marcati con `schema.compendium(<def>, ...)` portano una struttura
    della tassonomia chiusa, non un valore libero: `traits[3].effects` ->
    "featureEffect".
    """
    from pydantic import BaseModel

    model: Any = schema_strict
    field = None
    for name in (n for n in re.split(r"[.\[]", field_path) if n and not n[0].isdigit()):
        name = name.rstrip("]")
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            return None
        field = model.model_fields.get(name)
        if field is None:
            return None
        model = _list_inner_type(field.annotation)
        if _is_extracted_model(model):
            model = None
    extra = getattr(field, "json_schema_extra", None)
    return extra.get("compendium") if isinstance(extra, dict) else None


def gate_compendium(value: Any, def_name: str) -> list[str]:
    """Errori del valore contro la tassonomia chiusa del compendium.

    Esiste come cancello PER CAMPO perché il controllo a livello di documento
    (6.3) arriva troppo tardi: boccia il documento intero ma le singole
    estrazioni restano `validated`, e l'export prende proprio quelle. Nella
    run 2 sono usciti 3 effetti invalidi in un seed dichiarato non valido dal
    log della riga sopra — cioè esattamente il dato che non deve passare.
    """
    from .compendium import validate_def

    items = value if isinstance(value, list) else [value]
    out: list[str] = []
    for i, item in enumerate(items):
        for e in validate_def(item, def_name):
            out.append(f"[{i}] {e}" if isinstance(value, list) else e)
    return out


def _validate_one(
    doc_id: str,
    field_path: str,
    row,
    schema_strict: type,
    db: DB,
    s: Settings,
    client: ExtractorClient,
) -> tuple[bool, int]:
    """Esegue i 3 cancelli su un'estrazione. Ritorna (ok, attempt_finale)."""
    attempt = row["attempt"]
    quote = row["quote"]
    page_no = row["page_no"]
    value = json.loads(row["value_json"]) if row["value_json"] else None

    # Assenza dichiarata dal modello (value=null E quote=null): verificata,
    # non fallita. I cancelli non hanno senso senza citazione.
    if value is None and quote is None:
        db.upsert_extraction(doc_id, field_path, None, None, page_no,
                             None, attempt, "validated", row["confidence"])
        return True, attempt

    # 6.1 grounding (sul testo canonico riconciliato della pagina)
    page_full_text = ""
    if page_no:
        page_full_text = db.get_page_text(doc_id, page_no)
    if not gate_grounding(quote, page_full_text, s.grounding_threshold):
        score = partial_ratio(normalize(quote or ""), normalize(page_full_text)) if page_full_text else 0
        log.warning("Grounding fallito per %s (attempt %d): quote=%r page=%s "
                    "testo pagina=%d caratteri score=%.0f", field_path, attempt,
                    quote, page_no, len(page_full_text), score)
        return _maybe_retry(doc_id, field_path, row, schema_strict, db, s, client,
                            err="grounding_failed")

    # 6.2 coerenza value/quote
    if not gate_value_quote(value, quote):
        log.warning("Coerenza value/quote fallita per %s", field_path)
        # non si ritenta per 6.2: citazione autentica ma interpretazione sbagliata
        db.upsert_extraction(doc_id, field_path, row["value_json"], quote, page_no,
                             _bbox(row), attempt, "rejected", row["confidence"])
        db.set_status(doc_id, "needs_review")
        return False, attempt

    # 6.3 tassonomia del compendium, per campo. Il check sul documento intero
    # resta (run -> _build_document + gate_schema) ma arriva dopo l'export.
    def_name = _compendium_def(schema_strict, field_path)
    if def_name and value is not None:
        errors = gate_compendium(value, def_name)
        if errors:
            log.warning("Compendium fallito per %s (attempt %d): %s",
                        field_path, attempt, errors[:2])
            return _maybe_retry(doc_id, field_path, row, schema_strict, db, s, client,
                                err=f"compendium: {'; '.join(errors[:2])}")

    # 6.5 localizzazione bbox (sulle regioni canoniche)
    regions = []
    if page_no:
        regions = [
            {"text": r["text"], "bbox": r["bbox"]}
            for r in db.get_canonical_regions(doc_id, page_no)
        ]
    bbox = localize_bbox(quote, regions) or _bbox(row)

    db.upsert_extraction(doc_id, field_path, row["value_json"], quote, page_no,
                         bbox, attempt, "validated", row["confidence"])
    return True, attempt


_SEG = re.compile(r"([^\.\[\]]+)(?:\[(\d+)\])?")


def _set_path(doc: dict, path: str, leaf: dict) -> None:
    """Scrive `leaf` nel dict annidato seguendo un field_path tipo 'a.b[0].c'."""
    segs = [(m.group(1), int(m.group(2)) if m.group(2) is not None else None)
            for m in _SEG.finditer(path)]
    cur = doc
    for i, (name, idx) in enumerate(segs):
        last = i == len(segs) - 1
        if idx is None:
            if last:
                cur[name] = leaf
            else:
                cur = cur.setdefault(name, {})
        else:
            lst = cur.setdefault(name, [])
            while len(lst) <= idx:
                lst.append({})
            if last:
                lst[idx] = leaf
            else:
                cur = lst[idx]


def _build_document(extractions) -> dict:
    """Ricostruisce il documento (dict di dict stile Extracted) dalle estrazioni."""
    doc: dict = {}
    for row in extractions:
        fp = row["field_path"]
        if "$" in fp:
            continue
        leaf = {
            "value": json.loads(row["value_json"]) if row["value_json"] else None,
            "quote": row["quote"],
            "page": row["page_no"],
            "bbox": json.loads(row["bbox"]) if row["bbox"] else None,
            "confidence": row["confidence"] or "high",
        }
        _set_path(doc, fp, leaf)
    return doc


def _fill_missing(schema: type, node: dict) -> None:
    """Campi assenti dalle estrazioni -> Extracted nulli (o None), così
    SchemaStrict può validare il documento completo."""
    from pydantic import BaseModel

    for name, fi in schema.model_fields.items():
        ann = fi.annotation
        inner = _list_inner_type(ann)
        if inner is not None and isinstance(inner, type) and issubclass(inner, BaseModel):
            for item in node.setdefault(name, []):
                _fill_missing(inner, item)
        elif _is_extracted_model(ann):
            node.setdefault(name, {
                "value": None, "quote": None, "page": None,
                "bbox": None, "confidence": "low",
            })
        elif isinstance(ann, type) and issubclass(ann, BaseModel):
            _fill_missing(ann, node.setdefault(name, {}))
        else:
            node.setdefault(name, None)


def _get_path(doc: Any, path: str) -> Any:
    """Naviga un dict seguendo un field_path tipo 'a.b[0].c'. None se assente."""
    cur = doc
    for m in _SEG.finditer(path):
        name, idx = m.group(1), m.group(2)
        if not isinstance(cur, dict) or name not in cur:
            return None
        cur = cur[name]
        if idx is not None:
            if not isinstance(cur, list) or int(idx) >= len(cur):
                return None
            cur = cur[int(idx)]
    return cur


def _retry_pages(doc_id: str, field_path: str, row, db: DB, max_pages: int = 3) -> list[int]:
    """Pagine da rileggere nel retry: quella della riga; per un elemento di
    lista quelle dell'inventario (elemento + sezione); per documenti corti
    tutte. Vuota se non c'è modo di restringere il contesto."""
    pages: list[int] = [row["page_no"]] if row["page_no"] else []
    top = _SEG.match(field_path)
    if top and top.group(2) is not None:
        inv = db.latest_extraction(doc_id, f"{top.group(1)}$inventory")
        items = json.loads(inv["value_json"] or "[]") if inv else []
        idx = int(top.group(2))
        if idx < len(items):
            for pn in (items[idx].get("section_page"), items[idx].get("page")):
                if pn and pn not in pages:
                    pages.append(pn)
    if pages:
        return sorted(pages)
    all_pages = [p["page_no"] for p in db.get_pages(doc_id)]
    return all_pages if len(all_pages) <= max_pages else []


def _retry_extract(
    doc_id: str,
    field_path: str,
    row,
    err: str,
    schema_strict: type,
    db: DB,
    s: Settings,
    client: ExtractorClient,
) -> dict:
    """Rilancia l'estrattore sul singolo campo con l'errore accodato al prompt.
    Ritorna il nuovo leaf {value_json, quote, page_no, bbox, confidence};
    in caso di risposta inutilizzabile, ricade sui valori precedenti."""
    page_nos = _retry_pages(doc_id, field_path, row, db)
    if not page_nos:
        # senza pagina il contesto sarebbe l'intero documento: su un manuale
        # sfora il contesto dell'estrattore. Meglio la revisione umana.
        log.warning("Retry %s: nessuna pagina nota, niente contesto da rileggere", field_path)
        return {"value_json": row["value_json"], "quote": row["quote"],
                "page_no": row["page_no"], "bbox": _bbox(row),
                "confidence": row["confidence"]}
    regions = [
        {"region_id": r["region_id"], "type": r["region_type"], "text": r["text"], "page": pn}
        for pn in page_nos for r in db.get_canonical_regions(doc_id, pn)
    ]
    # Schema del solo campo in retry. Per un elemento di lista (parties[1].name)
    # si parte dal modello interno ristretto al campo che ha fallito: chiedere
    # l'elemento intero fa riemettere tutti i suoi campi per correggerne uno, e
    # su un retry che non converge il modello esaurisce max_tokens prima ancora
    # di arrivarci — nella run 3 due tentativi su traits[12].effects sono morti
    # dentro `raceName`, che era già validato.
    top = _SEG.match(field_path)
    top_field, item_idx = top.group(1), top.group(2)
    leaf_name = field_path[top.end():].lstrip(".")
    inner = (_list_inner_type(schema_strict.model_fields[top_field].annotation)
             if item_idx is not None and top_field in schema_strict.model_fields else None)
    narrow = bool(inner is not None and leaf_name
                  and not set(leaf_name) & set(".[")
                  and leaf_name in getattr(inner, "model_fields", {}))
    lines = ["REGIONI (id | pagina | tipo | testo):"]
    lines += [f"[{r['region_id']}] p.{r['page']} {r['type']}: {r['text']}" for r in regions]
    lines += [
        "",
        f"CAMPO DA ESTRARRE: {field_path}",
    ]
    if narrow:
        lines.append(f"Rispondi SOLO con il campo '{leaf_name}' di quell'elemento: "
                     f"gli altri campi sono già stati validati, non riemetterli.")
    elif item_idx is not None:
        lines.append(f"Rispondi con la lista '{top_field}' contenente SOLO l'elemento a cui "
                     f"appartiene il valore precedente (compila tutti i suoi campi).")
    lines += [
        "",
        "REGOLE:",
        "- restituisci un oggetto {value, quote, page, bbox, confidence}.",
        "- quote DEVE essere copiata carattere per carattere dal testo fornito sopra.",
        "- null è la risposta corretta quando il dato non è presente.",
        "",
        f"IL TENTATIVO PRECEDENTE È FALLITO: {err}",
        f"Valore precedente: {row['value_json']}  quote: {row['quote']}",
        "Correggi: la nuova quote deve apparire verbatim nelle regioni fornite.",
        # Senza questa riga un vincolo del tipo "serve amount" si può
        # soddisfare solo AGGIUNGENDO, e il modello aggiunge: alla run 3 un
        # save_bonus senza amount è tornato come fromAbilityModifier='con',
        # cioè una meccanica inventata, su un testo che dice "advantage".
        "- se un elemento della lista non corrisponde a un cambiamento numerico "
        "scritto nel testo, la correzione è RIMUOVERLO dalla lista (o rispondere "
        "value=null se resta vuota), non completarlo con campi che il testo non "
        "dice. Nessun effetto è meglio di un effetto inventato per far quadrare "
        "lo schema.",
    ]
    fallback = {"value_json": row["value_json"], "quote": row["quote"],
                "page_no": row["page_no"], "bbox": _bbox(row),
                "confidence": row["confidence"]}
    try:
        raw = client.extract(
            "\n".join(lines),
            guided_json_schema=(guided_schema(inner, [leaf_name]) if narrow else
                                guided_schema(schema_strict, [top_field],
                                              single_item=item_idx is not None)))
    except Exception as e:
        log.warning("Retry extract fallito (%s): %s", field_path, e)
        return fallback
    path_in_raw = (leaf_name if narrow else field_path if item_idx is None else
                   f"{top_field}[0]" + field_path[top.end():])
    leaf = _get_path(raw, path_in_raw)
    if narrow and not isinstance(leaf, dict):
        # il modello ha incartato la risposta nell'elemento di lista comunque
        leaf = _get_path(raw, f"{top_field}[0].{leaf_name}")
    if not isinstance(leaf, dict) or ("value" not in leaf and "quote" not in leaf):
        log.warning("Retry: campo %s assente nella risposta", field_path)
        return fallback
    return {
        "value_json": json.dumps(leaf.get("value"), ensure_ascii=False)
                      if leaf.get("value") is not None else None,
        "quote": leaf.get("quote"),
        "page_no": leaf.get("page", row["page_no"]),
        "bbox": tuple(leaf["bbox"]) if isinstance(leaf.get("bbox"), (list, tuple))
                and len(leaf["bbox"]) == 4 else _bbox(row),
        "confidence": leaf.get("confidence", row["confidence"]),
    }


def _maybe_retry(
    doc_id: str,
    field_path: str,
    row,
    schema_strict: type,
    db: DB,
    s: Settings,
    client: ExtractorClient,
    err: str,
) -> tuple[bool, int]:
    attempt = row["attempt"]
    if attempt >= s.max_retries + 1:
        # al 3° tentativo (max_retries=2 -> attempt 3) -> needs_review
        db.upsert_extraction(doc_id, field_path, row["value_json"], row["quote"],
                             row["page_no"], _bbox(row), attempt, "needs_review",
                             row["confidence"])
        db.set_status(doc_id, "needs_review")
        return False, attempt
    # rilancia l'estrattore con l'errore accodato al prompt, poi rivalida
    new = _retry_extract(doc_id, field_path, row, err, schema_strict, db, s, client)
    next_attempt = attempt + 1
    db.upsert_extraction(doc_id, field_path, new["value_json"], new["quote"],
                         new["page_no"], new["bbox"], next_attempt, "pending",
                         new["confidence"])
    log.info("Retry %d per %s (err=%s)", next_attempt, field_path, err)
    new_row = db.latest_extraction(doc_id, field_path)
    return _validate_one(doc_id, field_path, new_row, schema_strict, db, s, client)


def _bbox(row):
    if row["bbox"]:
        try:
            return tuple(json.loads(row["bbox"]))
        except json.JSONDecodeError:
            return None
    return None


__all__ = [
    "run",
    "gate_grounding",
    "gate_value_quote",
    "gate_schema",
    "localize_bbox",
]
