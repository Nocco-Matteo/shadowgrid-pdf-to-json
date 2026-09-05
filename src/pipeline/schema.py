"""Schema come fonte unica di verità.

- ``Extracted[T]``: ogni campo foglia è un oggetto con provenienza (value/quote/page/bbox/confidence).
- ``SchemaLoose``: derivato dallo strict, passato al decoder come ``guided_json`` (solo tipi/struttura,
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


# ---------------------------------------------------------------------------
# Flatten per scrittura su tabella extractions
# ---------------------------------------------------------------------------


def flatten_extracted(obj: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Appiattisce uno schema di Extracted in righe (field_path, value_json, quote, page, bbox).
    Salta i campi Extracted con value=None (dati assenti)."""
    rows: list[dict[str, Any]] = []
    if isinstance(obj, Extracted):
        if obj.value is None:
            return rows  # dato assente: non produrre riga
        rows.append({
            "field_path": prefix.rstrip(".") or "<root>",
            "value_json": json.dumps(obj.value, ensure_ascii=False),
            "quote": obj.quote,
            "page": obj.page,
            "bbox": obj.bbox,
            "confidence": obj.confidence,
        })
        return rows
    if isinstance(obj, BaseModel):
        for name, _fi in type(obj).model_fields.items():
            val = getattr(obj, name)
            child_prefix = f"{prefix}{name}."
            if val is None:
                continue
            if isinstance(val, list):
                for i, item in enumerate(val):
                    rows.extend(flatten_extracted(item, f"{child_prefix}[{i}]."))
            else:
                rows.extend(flatten_extracted(val, child_prefix))
        return rows
    if isinstance(obj, dict):
        for k, v in obj.items():
            rows.extend(flatten_extracted(v, f"{prefix}{k}."))
    return rows


__all__ = [
    "Extracted",
    "BBox",
    "Confidence",
    "Party",
    "ContractStrict",
    "loose",
    "flatten_extracted",
]
