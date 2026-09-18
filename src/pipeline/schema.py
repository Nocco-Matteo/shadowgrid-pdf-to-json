"""Schema come fonte unica di verità.

- ``Extracted[T]``: ogni campo foglia è un oggetto con provenienza (value/quote/page/bbox/confidence).
- ``SchemaLoose``: derivato dallo strict, passato al decoder come JSON schema (``response_format``) (solo tipi/struttura,
  tutto Optional, niente pattern/ge/le/validator) -> grammatica semplice, compilazione xgrammar veloce.
- ``SchemaStrict``: validazione a valle con tutti i vincoli e validator cross-field.
"""

from __future__ import annotations

import json
from typing import Any, Generic, Literal, TypeVar

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


class ContractStrict(BaseModel):
    """SchemaStrict di esempio: contratto con campi piatti + lista parti."""

    model_config = ConfigDict(extra="forbid")

    contract_number: Extracted[str]
    issue_date: Extracted[str]  # formato YYYY-MM-DD validato a valle
    amount_eur: Extracted[float]
    currency: Extracted[Literal["EUR", "USD", "GBP"]]
    parties: list[Party]

    @model_validator(mode="after")
    def check_parties_nonempty_when_amount(self) -> ContractStrict:
        # regola cross-field di esempio
        if self.amount_eur.value is not None and self.amount_eur.value > 0 and not self.parties:
            raise ValueError("importo positivo richiede almeno una parte")
        return self


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


def guided_schema(
    model: type[BaseModel],
    fields: list[str] | None = None,
    single_item: bool = False,
) -> dict[str, Any]:
    """JSON schema per la generazione vincolata di un task di estrazione.

    Parte da ``loose(model)`` ma, a differenza di quello (dove tutto è
    opzionale), obbliga il modello a rispondere a ogni campo richiesto:
    - la radice contiene solo ``fields`` (default: tutti), tutti required;
    - ogni campo foglia è l'oggetto Extracted (non null) con value e quote
      required: l'assenza si dichiara con value=null e quote=null, non
      omettendo la chiave (che in Fase 5 è un task fallito);
    - ``single_item``: le liste hanno esattamente un elemento (task per
      singolo elemento di lista)."""
    schema = loose(model).model_json_schema()
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

    for d in defs.values():
        if str(d.get("title", "")).startswith("Extracted"):
            for prop in d.get("properties", {}).values():
                prop.pop("default", None)
            d["required"] = ["value", "quote"]
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
