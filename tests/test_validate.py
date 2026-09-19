"""Test di phase6_validate.py: cancelli grounding, coerenza, localizzazione."""
from __future__ import annotations

from pipeline.phase6_validate import (
    gate_grounding,
    gate_value_quote,
    localize_bbox,
)


def test_grounding_ok():
    assert gate_grounding("Contratto n. 44/B", "Il Contratto n. 44/B del 2024", 90)


def test_grounding_fail_hallucinated():
    assert not gate_grounding("pippo pluto", "Il Contratto n. 44/B del 2024", 90)


def test_grounding_empty_quote():
    assert not gate_grounding("", "testo", 90)


def test_grounding_quote_longer_than_page():
    # partial_ratio allineerebbe la pagina DENTRO la quote (score alto):
    # una citazione più lunga dell'intera pagina non può essere verbatim
    assert not gate_grounding(
        "Mario Rossi Verdi e tanto altro testo inventato per superare la pagina",
        "Mario Rossi",
        90,
    )


def test_grounding_identifier_substitution():
    """Regressione: '44/B' vs '45/B' — il fuzzy sul contesto tiene (score
    alto), ma il token identificativo è diverso: non può essere verbatim."""
    assert not gate_grounding("CONTRATTO N. 44/B", "CONTRATTO N. 45/B", 90)
    assert gate_grounding("CONTRATTO N. 44/B", "il CONTRATTO N. 44/B firmato", 90)


def test_grounding_sign_mismatch():
    """Il segno fa parte del numero anche nel grounding (normalize lo cancella:
    serve il controllo sui numeri con segno)."""
    assert not gate_grounding("Saldo: -42 euro", "Saldo: 42 euro", 90)
    assert not gate_grounding("Saldo: 42 euro", "Saldo: -42 euro", 90)
    assert gate_grounding("Saldo: -42 euro", "Saldo: -42 euro", 90)


def test_value_quote_sign_variants():
    """Varianti del segno: − U+2212 e '- 42' sono comunque -42."""
    assert not gate_value_quote(42, "Saldo: \u221242 euro")  # meno Unicode
    assert gate_value_quote(-42, "Saldo: \u221242 euro")
    assert not gate_value_quote(42, "Saldo: - 42 euro")  # meno staccato
    assert gate_value_quote(-42, "Saldo: - 42 euro")


def test_value_quote_number_ok():
    assert gate_value_quote(44, "Contratto n. 44/B")


def test_value_quote_number_wrong():
    assert not gate_value_quote(45, "Contratto n. 44/B")


def test_value_quote_number_sign_mismatch():
    # il segno fa parte del valore: 42 non è accettabile se la quote dice -42
    assert not gate_value_quote(42, "Saldo: -42 euro")
    assert gate_value_quote(-42, "Saldo: -42 euro")


def test_value_quote_float_comma():
    assert gate_value_quote(1234.5, "Importo: 1.234,50 euro")


def test_value_quote_string_substring():
    assert gate_value_quote("ACME", "Società ACME S.r.l.")


def test_value_quote_string_wrong():
    assert not gate_value_quote("XYZ", "Società ACME S.r.l.")


def test_value_quote_invented_longer_value():
    # regressione: value più lungo della quote (token overlap 60% lo accettava)
    assert not gate_value_quote("Mario Rossi Verdi", "Mario Rossi")


def test_value_quote_reordered_value_rejected():
    # conservativo: il valore riordinato non è contenuto nella citazione
    assert not gate_value_quote("Rossi Mario", "Mario Rossi")


def test_value_quote_both_none():
    assert gate_value_quote(None, None)


def test_value_quote_enum_via_substring():
    # enum mappata: la quote contiene il termine che mappa al valore
    assert gate_value_quote("EUR", "Importo in euro")


def test_localize_bbox_found():
    regions = [
        {"text": "preambolo", "bbox": "[0,0,10,10]"},
        {"text": "Contratto n. 44/B", "bbox": "[10,10,100,30]"},
    ]
    bbox = localize_bbox("Contratto n. 44/B", regions)
    assert bbox == (10, 10, 100, 30)


def test_localize_bbox_not_found():
    regions = [{"text": "altro", "bbox": "[0,0,10,10]"}]
    assert localize_bbox("pippo", regions) is None


def test_retry_reextracts_with_error(tmp_path):
    """Grounding fallito -> retry rilancia l'estrattore con l'errore nel prompt."""
    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase6_validate import run
    from pipeline.schema import ContractStrict

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.set_status("d", "extracted")
    db.set_page("d", 1, full_text=(
        "Contratto n. 44/B del 2024\n"
        "Data emissione: 2024-03-15\n"
        "Importo: 1.234,50 EUR\n"
        "Valuta: EUR\n"
        "Parti: ACME - buyer"
    ))
    # tentativo 1: quote allucinata -> grounding fallisce
    db.upsert_extraction("d", "contract_number", '"44/B"', "pippo", 1, None, 1, "pending")
    # gli altri campi del task sono a posto
    db.upsert_extraction("d", "issue_date", '"2024-03-15"',
                         "Data emissione: 2024-03-15", 1, None, 1, "pending")
    db.upsert_extraction("d", "amount_eur", "1234.5",
                         "Importo: 1.234,50 EUR", 1, None, 1, "pending")
    db.upsert_extraction("d", "currency", '"EUR"', "Valuta: EUR", 1, None, 1, "pending")
    # inventario + parte (amount > 0 richiede almeno una parte)
    db.upsert_extraction("d", "parties$inventory",
                         '[{"anchor": "ACME", "page": 1, "region_ids": []}]',
                         None, None, None, 1, "validated")
    db.upsert_extraction("d", "parties[0].name", '"ACME"', "ACME", 1, None, 1, "pending")
    db.upsert_extraction("d", "parties[0].role", '"buyer"', "ACME - buyer", 1, None, 1, "pending")
    db.upsert_extraction("d", "parties[0].vat_id", None, None, None, None, 1, "pending")

    calls = []

    def leaf(value, quote):
        return {"value": value, "quote": quote, "page": 1,
                "bbox": [0, 0, 10, 10], "confidence": "high"}

    class Fixer:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            calls.append(prompt)
            return {"contract_number": leaf("44/B", "Contratto n. 44/B")}

    run("d", ContractStrict, db=db, settings=s, client=Fixer())

    row = db.latest_extraction("d", "contract_number")
    assert row["attempt"] == 2
    assert row["status"] == "validated"
    assert db.get_status("d") == "validated"
    # l'errore è stato accodato al prompt del retry
    assert calls and "FALLITO" in calls[0]


def test_retry_exhausted_goes_needs_review(tmp_path):
    """Tre tentativi falliti -> needs_review."""
    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase6_validate import run
    from pipeline.schema import ContractStrict

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.set_status("d", "extracted")
    db.set_page("d", 1, full_text="Contratto n. 44/B del 2024")
    db.upsert_extraction("d", "contract_number", '"44/B"', "pippo", 1, None, 1, "pending")

    class AlwaysBad:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            return {"contract_number": {"value": "44/B", "quote": "ancora pippo",
                                        "page": 1, "bbox": None, "confidence": "high"}}

    run("d", ContractStrict, db=db, settings=s, client=AlwaysBad())

    row = db.latest_extraction("d", "contract_number")
    assert row["status"] == "needs_review"
    assert db.get_status("d") == "needs_review"


def test_zero_extractions_needs_review(tmp_path):
    """Risposta vuota dell'estrattore (zero righe) NON diventa documento validato."""
    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase6_validate import run
    from pipeline.schema import ContractStrict

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.set_status("d", "extracted")
    db.set_page("d", 1, full_text="Contratto n. 44/B del 2024")

    class Unused:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            raise AssertionError("non deve essere chiamato: nessun pending")

    run("d", ContractStrict, db=db, settings=s, client=Unused())
    assert db.get_status("d") == "needs_review"


def test_explicit_null_validated_as_absence(tmp_path):
    """value=null E quote=null = assenza verificata: validata senza cancelli."""
    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase6_validate import run
    from pipeline.schema import ContractStrict

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.set_status("d", "extracted")
    db.set_page("d", 1, full_text=(
        "Contratto n. 44/B del 2024\n"
        "Data emissione: 2024-03-15\n"
        "Importo: 1.234,50 EUR\n"
        "Valuta: EUR\n"
        "Parti: ACME - buyer"
    ))
    db.upsert_extraction("d", "contract_number", '"44/B"', "Contratto n. 44/B", 1,
                         None, 1, "pending")
    db.upsert_extraction("d", "issue_date", None, None, None, None, 1, "pending")
    db.upsert_extraction("d", "amount_eur", "1234.5",
                         "Importo: 1.234,50 EUR", 1, None, 1, "pending")
    db.upsert_extraction("d", "currency", '"EUR"', "Valuta: EUR", 1, None, 1, "pending")
    db.upsert_extraction("d", "parties$inventory",
                         '[{"anchor": "ACME", "page": 1, "region_ids": []}]',
                         None, None, None, 1, "validated")
    db.upsert_extraction("d", "parties[0].name", '"ACME"', "ACME", 1, None, 1, "pending")
    db.upsert_extraction("d", "parties[0].role", '"buyer"', "ACME - buyer", 1, None, 1, "pending")
    db.upsert_extraction("d", "parties[0].vat_id", None, None, None, None, 1, "pending")

    class NoCalls:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            raise AssertionError("niente da ritentare")

    run("d", ContractStrict, db=db, settings=s, client=NoCalls())
    row = db.latest_extraction("d", "issue_date")
    assert row["status"] == "validated"
    assert row["value_json"] is None
    assert db.get_status("d") == "validated"


def test_grounding_uses_canonical_text(tmp_path):
    """Il grounding usa il testo canonico (riconciliato), non l'originale A."""
    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase6_validate import run
    from pipeline.schema import ContractStrict

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.set_status("d", "extracted")
    db.set_page("d", 1, image_path="/p1.png", dpi=300, deskew_angle=0.0)
    from pipeline.ocr_clients import Region

    # OCR A con typo: full_text (originale) resta con l'errore...
    db.save_page_ocr_a("d", 1, [
        Region(page_no=1, bbox=(0, 0, 10, 10), region_type="text",
               text="Impofto: 1.234,50 EUR", order_idx=0)
    ], "Impofto: 1.234,50 EUR")
    rid = db.get_regions("d", 1, engine="a")[0]["region_id"]
    # ...la riconciliazione corregge solo il testo canonico
    db.update_region_canonical_text(rid, "Importo: 1.234,50 EUR")
    db.rebuild_page_canonical_text("d", 1)
    assert "Impofto" in db.get_page("d", 1)["full_text"]

    db.upsert_extraction("d", "amount_eur", "1234.50", "Importo: 1.234,50 EUR", 1,
                         None, 1, "pending")

    class NoRetry:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            raise AssertionError("grounding deve passare sul canonico")

    run("d", ContractStrict, db=db, settings=s, client=NoRetry())
    assert db.latest_extraction("d", "amount_eur")["status"] == "validated"


def test_partial_document_needs_review(tmp_path):
    """Copertura schema: un documento con un solo campo estratto (gli altri
    mai prodotti dal modello) non può diventare `validated` riempiendo i
    mancanti con null."""
    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase6_validate import run
    from pipeline.schema import ContractStrict

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.set_status("d", "extracted")
    db.set_page("d", 1, full_text="Contratto n. 44/B del 2024")
    # Solo contract_number prodotto: tutti gli altri campi mai visti
    db.upsert_extraction("d", "contract_number", '"44/B"',
                         "Contratto n. 44/B", 1, None, 1, "pending")

    class NoCalls:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            raise AssertionError("niente da ritentare")

    run("d", ContractStrict, db=db, settings=s, client=NoCalls())
    # il campo buono si valida, ma il documento resta in revisione
    assert db.latest_extraction("d", "contract_number")["status"] == "validated"
    assert db.get_status("d") == "needs_review"


def test_validate_from_needs_review_finalizes(tmp_path):
    """Resume e2e: `validate` su un doc in needs_review con tutti gli ultimi
    tentativi già validati lo porta a done (finalizzazione)."""
    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase6_validate import run
    from pipeline.schema import ContractStrict

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.set_status("d", "needs_review")
    db.set_page("d", 1, full_text="x")
    db.upsert_extraction("d", "contract_number", '"44/B"', "q", 1, None, 1, "validated")
    db.upsert_extraction("d", "issue_date", None, None, None, None, 1, "validated")
    db.upsert_extraction("d", "amount_eur", "1234.5", "q", 1, None, 1, "validated")
    db.upsert_extraction("d", "currency", '"EUR"', "q", 1, None, 1, "validated")
    db.upsert_extraction("d", "parties$inventory",
                         '[{"anchor": "ACME", "page": 1, "region_ids": []}]',
                         None, None, None, 1, "validated")
    db.upsert_extraction("d", "parties[0].name", '"ACME"', "q", 1, None, 1, "validated")
    db.upsert_extraction("d", "parties[0].role", '"buyer"', "q", 1, None, 1, "validated")
    db.upsert_extraction("d", "parties[0].vat_id", None, None, None, None, 1, "validated")

    run("d", ContractStrict, db=db, settings=s)
    assert db.get_status("d") == "done"


def test_retry_list_item_reads_single_element_response(tmp_path):
    """Il retry di parties[1].name chiede SOLO il campo name, non l'elemento
    intero: riemettere i campi già validati costa token e nella run 3 ha fatto
    esaurire max_tokens prima ancora di arrivare al campo sbagliato.

    Se il modello incarta comunque la risposta nell'elemento di lista (come fa
    qui lo stub), va letta lo stesso, leggendo l'elemento 0 e non l'indice 1."""
    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase6_validate import run
    from pipeline.schema import ContractStrict

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.set_status("d", "extracted")
    db.set_page("d", 1, full_text=(
        "CONTRATTO N. 44/B\nData emissione: 2024-03-15\nImporto: 1.234,50 EUR\n"
        "Valuta: EUR\nMario Rossi - Acquirente\nLucia Bianchi - Venditore"
    ))
    for fp, v, q in [("contract_number", '"44/B"', "CONTRATTO N. 44/B"),
                     ("issue_date", '"2024-03-15"', "Data emissione: 2024-03-15"),
                     ("amount_eur", "1234.5", "Importo: 1.234,50 EUR"),
                     ("currency", '"EUR"', "Valuta: EUR"),
                     ("parties[0].name", '"Mario Rossi"', "Mario Rossi"),
                     ("parties[0].role", '"Acquirente"', "Acquirente")]:
        db.upsert_extraction("d", fp, v, q, 1, None, 1, "pending")
    db.upsert_extraction("d", "parties$inventory",
                         '[{"anchor": "Mario Rossi", "page": 1}, {"anchor": "Lucia Bianchi", "page": 1}]',
                         None, None, None, 1, "validated")
    db.upsert_extraction("d", "parties[0].vat_id", None, None, None, None, 1, "pending")
    # parties[1].name: tentativo 1 senza pagina -> grounding fallisce
    db.upsert_extraction("d", "parties[1].name", '"Lucia Bianchi"', "Lucia Bianchi",
                         None, None, 1, "pending")
    db.upsert_extraction("d", "parties[1].role", '"Venditore"', "Venditore", 1, None, 1, "pending")
    db.upsert_extraction("d", "parties[1].vat_id", None, None, None, None, 1, "pending")

    schemas = []

    def leaf(v):
        return {"value": v, "quote": v, "page": 1, "bbox": None, "confidence": "high"}

    class Fixer:
        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            schemas.append(guided_json_schema)
            return {"parties": [{"name": leaf("Lucia Bianchi"), "role": leaf("Venditore"),
                                 "vat_id": {"value": None, "quote": None}}]}

    run("d", ContractStrict, db=db, settings=s, client=Fixer())

    row = db.latest_extraction("d", "parties[1].name")
    assert row["attempt"] == 2 and row["status"] == "validated"
    assert row["page_no"] == 1
    assert schemas and set(schemas[0]["properties"]) == {"name"}


# ---------------------------------------------------------------------------
# 6.2 — derivazione dichiarata dallo schema (velocità)
# ---------------------------------------------------------------------------


def test_gate_accepts_speed_derived_from_quote():
    """Lo schema CHIEDE speed_bonus.amount come differenza da 30 feet; il
    cancello deve riconoscere la derivazione, non punire l'obbedienza.

    Nella prima run reale sul Player's Handbook questi quattro casi sono usciti
    tutti `rejected` con citazione perfetta: il 59% degli elementi enumerati era
    un tratto di velocità, quindi strutturalmente impossibile da validare.
    """
    assert gate_value_quote([{"kind": "speed_bonus", "amount": 5}],
                            "Your base walking speed increases to 35 feet.")
    assert gate_value_quote([{"kind": "speed_bonus", "amount": -5}],
                            "Your base walking speed is 25 feet.")
    assert gate_value_quote(
        [{"kind": "speed_bonus", "amount": -5, "condition": "wearing_heavy_armor"}],
        "Your base walking speed is 25 feet. Your speed is not reduced by wearing heavy armor.")
    # forma già assoluta nel testo: accettata anche senza derivare
    assert gate_value_quote([{"kind": "speed_bonus", "amount": 10}],
                            "Your speed increases by 10 feet.")


def test_gate_still_rejects_wrong_speed():
    """La derivazione non è un lasciapassare: il segno sbagliato resta un errore."""
    assert not gate_value_quote([{"kind": "speed_bonus", "amount": 5}],
                                "Your base walking speed is 25 feet.")
    assert not gate_value_quote([{"kind": "speed_bonus", "amount": -10}],
                                "Your base walking speed is 25 feet.")


def test_gate_derivation_is_only_for_speed():
    """Nessun altro kind gode della derivazione: un numero inventato resta tale."""
    quote = "Your hit point maximum increases by 1, and it increases by 1 every time you gain a level."
    assert gate_value_quote([{"kind": "hp_bonus_per_level", "amount": 1}], quote)
    assert not gate_value_quote([{"kind": "hp_bonus_per_level", "amount": 2}], quote)
    # 31 = 30 + 1 non testimonia nulla se il kind non è speed_bonus
    assert not gate_value_quote([{"kind": "ac_bonus", "amount": 1}],
                                "Your armor class becomes 31.")


# ---------------------------------------------------------------------------
# 6.3 per campo — tassonomia del compendium
# ---------------------------------------------------------------------------


def test_compendium_def_resolves_from_field_path():
    from pipeline.phase6_validate import _compendium_def
    from pipeline.schema import RaceTraitsDoc

    assert _compendium_def(RaceTraitsDoc, "traits[12].effects") == "featureEffect"
    assert _compendium_def(RaceTraitsDoc, "traits[0].raceName") is None
    assert _compendium_def(RaceTraitsDoc, "inesistente") is None


def test_gate_compendium_catches_what_reached_the_export():
    """I tre effetti usciti invalidi dalla run 2. Il check sul documento intero
    li vedeva, ma arriva DOPO: le singole estrazioni restavano `validated` e
    l'export prende proprio quelle."""
    from pipeline.phase6_validate import gate_compendium

    assert gate_compendium(
        [{"kind": "resistance", "damageTypes": ["poison"]},
         {"kind": "save_bonus", "condition": "always"}], "featureEffect")
    assert gate_compendium(
        [{"kind": "resistance",
          "damageTypes": ["associated with your draconic ancestry"]}], "featureEffect")
    # quelli buoni passano
    assert gate_compendium([{"kind": "speed_bonus", "amount": -5}], "featureEffect") == []
    assert gate_compendium(
        [{"kind": "resistance", "damageTypes": ["fire"]}], "featureEffect") == []


def test_retry_falls_back_when_server_rejects_the_narrow_schema(tmp_path):
    """Alla run 6 vLLM ha risposto 400 "Unsupported JSON Schema structure" a
    OGNI retry con lo schema ristretto: la riparazione era morta e nel
    risultato non si vedeva. Se il server rifiuta la forma ristretta si
    ripiega su quella intera, che ha sempre funzionato."""
    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase6_validate import run
    from pipeline.schema import ContractStrict

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.set_status("d", "extracted")
    db.set_page("d", 1, full_text="Mario Rossi - Acquirente\nLucia Bianchi - Venditore")
    db.upsert_extraction("d", "parties$inventory",
                         '[{"anchor": "Mario Rossi", "page": 1}, '
                         '{"anchor": "Lucia Bianchi", "page": 1}]',
                         None, None, None, 1, "validated")
    for fp, v, q in [("parties[0].name", '"Mario Rossi"', "Mario Rossi"),
                     ("parties[0].role", '"Acquirente"', "Acquirente")]:
        db.upsert_extraction("d", fp, v, q, 1, None, 1, "pending")
    db.upsert_extraction("d", "parties[0].vat_id", None, None, None, None, 1, "pending")
    db.upsert_extraction("d", "parties[1].vat_id", None, None, None, None, 1, "pending")
    db.upsert_extraction("d", "parties[1].role", '"Venditore"', "Venditore", 1, None, 1, "pending")
    # pagina assente -> grounding fallisce -> retry
    db.upsert_extraction("d", "parties[1].name", '"Lucia Bianchi"', "Lucia Bianchi",
                         None, None, 1, "pending")

    def leaf(v):
        return {"value": v, "quote": v, "page": 1, "bbox": None, "confidence": "high"}

    tried = []

    class PickyServer:
        """Rifiuta ogni schema che non sia quello della lista intera."""

        def extract(self, prompt, images_b64=None, guided_json_schema=None):
            props = set(guided_json_schema["properties"])
            tried.append(props)
            if props != {"parties"}:
                raise RuntimeError("400 Unsupported JSON Schema structure false")
            return {"parties": [{"name": leaf("Lucia Bianchi"), "role": leaf("Venditore"),
                                 "vat_id": {"value": None, "quote": None}}]}

    run("d", ContractStrict, db=db, settings=s, client=PickyServer())

    assert {"name"} in tried, "la forma ristretta va provata per prima"
    assert {"parties"} in tried, "senza ripiego il retry resta morto"
    row = db.latest_extraction("d", "parties[1].name")
    assert row["attempt"] == 2 and row["status"] == "validated"
    db.close()
