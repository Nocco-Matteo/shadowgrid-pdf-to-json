"""Schema come fonte unica di verità.

- ``Extracted[T]``: ogni campo foglia è un oggetto con provenienza (value/quote/page/bbox/confidence).
- ``SchemaLoose``: derivato dallo strict, passato al decoder come JSON schema (``response_format``) (solo tipi/struttura,
  tutto Optional, niente pattern/ge/le/validator) -> grammatica semplice, compilazione xgrammar veloce.
- ``SchemaStrict``: validazione a valle con tutti i vincoli e validator cross-field.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")

Confidence = Literal["high", "low"]
BBox = tuple[float, float, float, float]


class Extracted(BaseModel, Generic[T]):
    """Campo foglia con provenienza verificabile."""

    model_config = ConfigDict(extra="forbid")

    value: T | None = None
    quote: str | None = None
    page: int | None = None
    bbox: BBox | None = None
    confidence: Confidence = "high"

    @model_validator(mode="after")
    def quote_required_with_value(self) -> Extracted[T]:
        # value e quote devono essere entrambi nulli o entrambi presenti.
        if (self.value is None) != (self.quote is None):
            raise ValueError(
                "value e quote devono essere entrambi nulli o entrambi presenti"
            )
        return self


# ---------------------------------------------------------------------------
# Schema di esempio (contratto). Sostituibile dall'utente con il proprio schema.
# ---------------------------------------------------------------------------


class Party(BaseModel):
    """Elemento di lista nidificata (esercita la Fase 4)."""

    model_config = ConfigDict(extra="forbid")

    name: Extracted[str]
    role: Extracted[str]
    vat_id: Extracted[str | None] = Field(default_factory=lambda: Extracted(value=None, quote=None))


def anchors(*labels: str) -> Any:
    """Etichette con cui il campo compare nel documento (es. "Importo"): la
    Fase 5 le usa per trovare pagina e regioni del campo invece di mandare al
    modello l'intero documento. Il nome del campo resta come ultimo tentativo."""
    return Field(json_schema_extra={"anchors": list(labels)})


def field_anchors(model: type[BaseModel], name: str) -> list[str]:
    """Etichette di ancoraggio del campo, in ordine di priorità."""
    extra = model.model_fields[name].json_schema_extra
    labels = list(extra.get("anchors", [])) if isinstance(extra, dict) else []
    return [*labels, name.replace("_", " ")]


class ContractStrict(BaseModel):
    """SchemaStrict di esempio: contratto con campi piatti + lista parti."""

    model_config = ConfigDict(extra="forbid")

    contract_number: Extracted[str] = anchors("contratto n", "contract no")
    issue_date: Extracted[str] = anchors("data emissione", "issue date")  # YYYY-MM-DD validato a valle
    amount_eur: Extracted[float] = anchors("importo", "amount")
    currency: Extracted[Literal["EUR", "USD", "GBP"]] = anchors("valuta", "currency")
    parties: list[Party]

    @model_validator(mode="after")
    def check_parties_nonempty_when_amount(self) -> ContractStrict:
        # regola cross-field di esempio
        if self.amount_eur.value is not None and self.amount_eur.value > 0 and not self.parties:
            raise ValueError("importo positivo richiede almeno una parte")
        return self


# ---------------------------------------------------------------------------
# Compendium Shadow Grid (schema/compendium.schema.json): raceTraits.json
# ---------------------------------------------------------------------------


def compendium(def_name: str, description: str) -> Any:
    """Campo il cui valore è una lista di oggetti ``$defs/<def_name>`` del
    compendium: struttura vincolata in decodifica, validata con jsonschema."""
    return Field(description=description, json_schema_extra={"compendium": def_name})


# Velocità base di un personaggio, in piedi. Lo schema chiede
# speed_bonus.amount come differenza da qui, e il cancello 6.2 usa la STESSA
# costante per riconoscere la derivazione nella citazione: finché la fonte è
# una sola le due cose non possono divergere (v. phase6_validate).
BASE_WALKING_SPEED = 30


class RaceTrait(BaseModel):
    """Un tratto razziale che cambia un numero del personaggio (raceTraits.json)."""

    model_config = ConfigDict(extra="forbid")

    raceName: Extracted[str] = Field(description=(
        "nome della razza come scritto nel titolo della sua sezione (es. 'Dwarf', "
        "'Genasi (Fire)'); quote = quel titolo"))
    subraceName: Extracted[str | None] = Field(
        default_factory=lambda: Extracted(value=None, quote=None),
        description=("sottorazza se il tratto è di una sottorazza (titolo 'Hill Dwarf' "
                     "-> 'Hill'); altrimenti value=null e quote=null"))
    featureName: Extracted[str] = Field(description=(
        "nome del tratto come scritto, senza il punto finale (es. 'Dwarven Resilience')"))
    grantedAtLevel: Extracted[int | None] = Field(
        default_factory=lambda: Extracted(value=None, quote=None),
        description=("livello a cui si ottiene il tratto SOLO se il testo lo dice "
                     "(es. 'when you reach 5th level' -> 5); altrimenti value=null e quote=null"))
    effects: Extracted[list[dict[str, Any]]] = compendium("featureEffect", (
        "effetti numerici del tratto nella tassonomia chiusa del compendium "
        "(resistance, ac_bonus, ac_formula, speed_bonus, hp_bonus_per_level, ...). "
        "Solo ciò che cambia un numero del personaggio; quote = la frase della regola. "
        f"Velocità: speed_bonus.amount è la differenza da {BASE_WALKING_SPEED} feet "
        f"('35 feet' -> 5, '25 feet' -> -5); una differenza di 0 NON è un effetto, "
        f"ometti l'elemento. "
        "condition solo se è la regola stessa a limitare l'effetto a quella "
        "situazione; 'la velocità NON è ridotta dall'armatura pesante' non è una "
        "condition sul valore base. "
        "Vantaggio, competenza e immunità non numeriche NON sono effetti: non "
        "metterli nella lista e non inventare campi per far quadrare lo schema "
        "(meglio nessun effetto che uno inventato). "
        "Se il tratto non cambia nessun numero: value=null e quote=null"))

    @model_validator(mode="after")
    def effects_match_compendium(self) -> RaceTrait:
        from .compendium import validate_def

        for i, effect in enumerate(self.effects.value or []):
            errors = validate_def(effect, "featureEffect")
            if errors:
                raise ValueError(f"effects[{i}] non valido per il compendium: {errors[:3]}")
        return self


class RaceTraitsDoc(BaseModel):
    """Schema di estrazione per il seed raceTraits.json del compendium."""

    model_config = ConfigDict(extra="forbid")
    seed_file: ClassVar[str] = "raceTraits.json"

    traits: list[RaceTrait] = Field(default_factory=list, description=(
        "tratti razziali: paragrafi che iniziano con il nome del tratto in grassetto "
        "(es. 'Dwarven Resilience.') nelle sezioni '<Razza> Traits' di razze e "
        "sottorazze. SOLO tratti che cambiano un numero del personaggio: resistenze o "
        "immunità ai danni, classe armatura, velocità, punti ferita. ESCLUSI gli "
        "aumenti di caratteristica ('Ability Score Increase'): non sono tratti, il "
        "compendium li tiene in un seed separato (raceAsi.json) e li cura a mano. "
        "Esclusi anche i tratti che danno solo vantaggio o competenza senza un "
        "numero, e le voci descrittive (Age, Alignment, Size, Languages, Names). "
        "anchor = il nome del tratto; section = il titolo della razza o sottorazza "
        "a cui appartiene"))


# Schemi selezionabili dalla CLI (--schema)
SCHEMAS: dict[str, type[BaseModel]] = {
    "race_traits": RaceTraitsDoc,
    "contract": ContractStrict,
}


# ---------------------------------------------------------------------------
# Derivazione SchemaLoose da SchemaStrict
# ---------------------------------------------------------------------------


def _is_extracted_model(annotation: Any) -> bool:
    try:
        return isinstance(annotation, type) and issubclass(annotation, Extracted)
    except TypeError:
        return False


def _list_inner_type(ann: Any) -> Any | None:
    import typing as t

    origin = t.get_origin(ann)
    if origin in (list, list):
        args = t.get_args(ann)
        return args[0] if args else None
    return None


def loose(model: type[BaseModel]) -> type[BaseModel]:
    """API pubblica: ritorna la variante loose di uno schema strict.

    Implementazione semplice e robusta: ricrea i campi con annotation Optional e
    rimuove i vincoli (Field senza pattern/ge/le). Per Extracted[T] usa Extracted[Any].
    """

    annotations: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    for name, fi in model.model_fields.items():
        ann = fi.annotation
        inner = _list_inner_type(ann)
        if inner is not None and isinstance(inner, type) and issubclass(inner, BaseModel):
            loose_inner = loose(inner)
            annotations[name] = list[loose_inner] | None  # type: ignore[valid-type]
            defaults[name] = None
        elif _is_extracted_model(ann):
            annotations[name] = Extracted[Any] | None
            defaults[name] = None
        else:
            annotations[name] = ann | None
            defaults[name] = None

    namespace: dict[str, Any] = {
        "__annotations__": annotations,
        "model_config": ConfigDict(extra="forbid"),
    }
    namespace.update(defaults)
    return type(f"{model.__name__}Loose", (BaseModel,), namespace)


_CONSTRAINT_KEYS = {
    "anchors", "compendium",  # metadati della pipeline, non vincoli per il decoder
    "pattern", "format", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
}


def _strip_constraints(node: Any, in_properties: bool = False) -> Any:
    """Rimuove dai JSON schema i vincoli di valore (verificati a valle dallo strict).
    Le chiavi di un oggetto ``properties`` sono nomi di campo, non vincoli: un
    campo chiamato "format" o "minimum" resta."""
    if isinstance(node, dict):
        return {k: _strip_constraints(v, in_properties=k == "properties" and not in_properties)
                for k, v in node.items() if in_properties or k not in _CONSTRAINT_KEYS}
    if isinstance(node, list):
        return [_strip_constraints(v) for v in node]
    return node


_CMP_PREFIX = "cmp__"


def _prefix_refs(node: Any, prefix: str) -> Any:
    if isinstance(node, dict):
        return {k: (f"#/$defs/{prefix}{v.removeprefix('#/$defs/')}" if k == "$ref" else
                    _prefix_refs(v, prefix)) for k, v in node.items()}
    if isinstance(node, list):
        return [_prefix_refs(v, prefix) for v in node]
    return node


def _apply_compendium_refs(schema: dict[str, Any]) -> None:
    """Campi ``Extracted[list[dict]]`` marcati ``compendium=<def>``: il valore
    diventa una lista di oggetti dello schema vero del compendium (sotto-schema
    per il decoder, con le sue $defs prefissate)."""
    from .compendium import decoding_schema

    defs = schema.setdefault("$defs", {})
    holders = [schema, *list(defs.values())]
    for holder in holders:
        for prop in holder.get("properties", {}).values():
            def_name = prop.get("compendium")
            if not def_name:
                continue
            sub_schema = _prefix_refs(decoding_schema(def_name), _CMP_PREFIX)
            for k, v in sub_schema.pop("$defs").items():
                defs.setdefault(f"{_CMP_PREFIX}{k}", v)
            ref = prop["$ref"].removeprefix("#/$defs/")
            leaf_name = f"{ref}__{def_name}"
            leaf = json.loads(json.dumps(defs[ref]))
            leaf["properties"]["value"] = {"anyOf": [
                {"type": "array", "items": sub_schema}, {"type": "null"}]}
            defs[leaf_name] = leaf
            prop["$ref"] = f"#/$defs/{leaf_name}"


def guided_schema(
    model: type[BaseModel],
    fields: list[str] | None = None,
    single_item: bool = False,
) -> dict[str, Any]:
    """JSON schema per la generazione vincolata di un task di estrazione.

    Parte dallo schema strict senza i vincoli pesanti per la grammatica
    (pattern, min/max, lunghezze): restano i tipi dei valori (number, enum...),
    altrimenti il modello restituisce p.es. un importo come stringa "1.234,50"
    e la validazione strict finale fallisce. Rispetto a ``loose`` obbliga il
    modello a rispondere a ogni campo richiesto:
    - la radice contiene solo ``fields`` (default: tutti), tutti required;
    - ogni campo foglia è l'oggetto Extracted (non null) con value, quote e
      confidence required: l'assenza si dichiara con value=null e quote=null,
      non omettendo la chiave (che in Fase 5 è un task fallito);
    - ``single_item``: le liste hanno esattamente un elemento (task per
      singolo elemento di lista)."""
    schema = model.model_json_schema()
    _apply_compendium_refs(schema)
    schema = _strip_constraints(schema)
    defs = schema.get("$defs", {})

    def tighten(obj: dict[str, Any]) -> None:
        props = obj.get("properties", {})
        for prop in props.values():
            prop.pop("default", None)
            options = [o for o in prop.get("anyOf", []) if o.get("type") != "null"]
            if len(options) == 1 and ("$ref" in options[0] or options[0].get("type") == "array"):
                prop.pop("anyOf")
                prop.update(options[0])
            if single_item and prop.get("type") == "array":
                prop["minItems"] = prop["maxItems"] = 1
        obj["required"] = list(props)

    for name, d in defs.items():
        if name.startswith(_CMP_PREFIX):
            continue  # schema del compendium: required/opzionali sono i suoi
        if str(d.get("title", "")).startswith("Extracted"):
            for prop in d.get("properties", {}).values():
                prop.pop("default", None)
            # confidence obbligatoria: se il modello la omette vale il default
            # "high", cioè il valore più rassicurante scelto da nessuno. Nella
            # prima run TUTTE e 153 le estrazioni sono uscite 'high', il che
            # rende il tasso di errore silenzioso (valore sbagliato + confidence
            # alta) identico al tasso di errore, e la metrica inutile.
            d["required"] = ["value", "quote", "confidence"]
        else:
            tighten(d)
    if fields is not None:
        schema["properties"] = {k: v for k, v in schema["properties"].items() if k in fields}
    tighten(schema)
    return schema


# ---------------------------------------------------------------------------
# Flatten per scrittura su tabella extractions
# ---------------------------------------------------------------------------


_EXTRACTED_KEYS = {"value", "quote"}


def _leaf_row(obj: Any, prefix: str) -> list[dict[str, Any]]:
    """Riga per un campo foglia (istanza Extracted o dict {value, quote, ...}).

    I null espliciti del modello (value=null E quote=null) producono una riga
    con value_json/quote NULL: è un'assenza DICHIARATA, distinta dal campo mai
    prodotto (nessuna riga). Serve a valle per non confondere "il modello ha
    verificato che il dato non c'è" con "l'estrazione è fallita"."""
    if isinstance(obj, Extracted):
        value, quote, page = obj.value, obj.quote, obj.page
        bbox, confidence = obj.bbox, obj.confidence
    else:
        value = obj.get("value")
        quote = obj.get("quote")
        page = obj.get("page")
        bbox = obj.get("bbox")
        confidence = obj.get("confidence", "high")
    if value is None and quote is not None:
        return []  # incoerente (value nullo, quote presente): scarta
    return [{
        "field_path": prefix.rstrip(".") or "<root>",
        "value_json": json.dumps(value, ensure_ascii=False)
                      if value is not None else None,
        "quote": quote,
        "page": page,
        "bbox": bbox,
        "confidence": confidence,
    }]


def _is_extracted_leaf_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(_EXTRACTED_KEYS & obj.keys())


def flatten_extracted(obj: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Appiattisce uno schema di Extracted (o il dict JSON equivalente prodotto
    dall'estrattore) in righe (field_path, value_json, quote, page, bbox).
    I null espliciti del foglio producono una riga con value_json=None
    (assenza dichiarata); i campi mai presenti nel payload nessuna riga."""
    if isinstance(obj, Extracted) or _is_extracted_leaf_dict(obj):
        return _leaf_row(obj, prefix)
    if isinstance(obj, BaseModel):
        obj = {name: getattr(obj, name) for name in type(obj).model_fields}
    if isinstance(obj, dict):
        rows: list[dict[str, Any]] = []
        for k, v in obj.items():
            if v is None:
                continue
            if isinstance(v, list):
                for i, item in enumerate(v):
                    rows.extend(flatten_extracted(item, f"{prefix}{k}[{i}]."))
            else:
                rows.extend(flatten_extracted(v, f"{prefix}{k}."))
        return rows
    return []


__all__ = [
    "Extracted",
    "BBox",
    "Confidence",
    "Party",
    "ContractStrict",
    "loose",
    "flatten_extracted",
]
