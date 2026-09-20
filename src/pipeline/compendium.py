"""Ponte verso ``schema/compendium.schema.json`` (i seed curati di Shadow Grid).

La pipeline estrae con modelli Pydantic ``Extracted[T]`` (valore + citazione);
il compendium è un JSON Schema Draft 2020-12 con un envelope per file seed.
Questo modulo:

- estrae un sotto-schema del compendium (es. ``featureEffect``) in forma adatta
  alla generazione vincolata: solo le $defs raggiunte, riferimenti annidati
  appiattiti, ``oneOf`` -> ``anyOf``, niente vincoli di valore/default;
- valida valori ed envelope con ``jsonschema`` contro lo schema vero;
- converte le estrazioni validate nel formato del seed (``{version, definitions}``).
"""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "compendium.schema.json"

# Vincoli di valore: verificati a valle da jsonschema, non servono al decoder
# (e pattern/lunghezze rendono la grammatica lenta da compilare).
_DROP_KEYS = {
    "pattern", "format", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "minLength", "maxLength", "default", "description", "$comment",
}


@lru_cache(maxsize=4)
def load_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    return _merge_closed_allof(json.loads(Path(path).read_text(encoding="utf-8")))


def _merge_closed_allof(root: dict[str, Any]) -> dict[str, Any]:
    """Il compendium esprime `base.extend({...})` di Zod come ``allOf`` di due
    oggetti entrambi con ``additionalProperties: false``: in JSON Schema ogni
    ramo valida da solo, quindi ciascuno rifiuta i campi dell'altro e NESSUN
    oggetto è valido (nemmeno i seed). Qui i rami vengono fusi in un unico
    oggetto chiuso, che è la semantica voluta. Da correggere nel generatore
    dello schema (unire i rami o usare unevaluatedProperties)."""
    for def_name, d in root.get("$defs", {}).items():
        branches = d.get("allOf") if isinstance(d, dict) else None
        if not branches:
            continue
        resolved = [_resolve(root, b["$ref"]) if "$ref" in b else b for b in branches]
        if not all(b.get("type") == "object" and b.get("additionalProperties") is False
                   for b in resolved):
            continue
        merged: dict[str, Any] = {"type": "object", "additionalProperties": False,
                                  "properties": {}, "required": []}
        for b in resolved:
            merged["properties"].update(b.get("properties", {}))
            merged["required"] += [r for r in b.get("required", []) if r not in merged["required"]]
        root["$defs"][def_name] = {**{k: v for k, v in d.items() if k != "allOf"}, **merged}
    return root


def _resolve(root: dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise ValueError(f"$ref non locale non supportato: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def _flat_name(ref: str) -> str:
    # "#/$defs/effects/resistance" -> "effects__resistance"
    return ref.removeprefix("#/$defs/").replace("/", "__")


def decoding_schema(def_name: str, path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    """Sotto-schema autosufficiente di ``$defs/<def_name>`` per il decoder."""
    root = load_schema(path)
    defs: dict[str, Any] = {}

    def convert(node: Any) -> Any:
        if isinstance(node, list):
            return [convert(v) for v in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _DROP_KEYS:
                continue
            if k == "$ref":
                name = _flat_name(v)
                if name not in defs:
                    defs[name] = {}  # segnaposto contro i cicli
                    defs[name] = convert(_resolve(root, v))
                out["$ref"] = f"#/$defs/{name}"
            elif k == "oneOf":
                out["anyOf"] = convert(v)  # varianti con `kind` const: disgiunte comunque
            else:
                out[k] = convert(v)
        return out

    top = convert({"$ref": f"#/$defs/{def_name}"})
    return {"$defs": defs, **top}


def _errors(schema: dict[str, Any], value: Any) -> list[str]:
    from jsonschema import Draft202012Validator

    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in Draft202012Validator(schema).iter_errors(value)]


# Vincoli "esattamente uno tra" che lo schema JSON documenta nelle description
# ma non codifica: a runtime li applica il loader del compendium. Qui servono
# perché un effetto che il loader rifiuterebbe non deve uscire dalla pipeline.
# Senza, un `{"kind": "save_bonus", "condition": "always"}` senza amount passa
# indenne — il cancello 6.2 controlla i numeri presenti, e lì non ce n'erano —
# ed è esattamente quello che la prima run ha esportato su Dwarven Resilience.
EXACTLY_ONE_OF: dict[str, tuple[str, ...]] = {
    "speed_bonus": ("amount", "byLevel"),
    "ac_bonus": ("amount", "fromAbilityModifier"),
    "attack_bonus": ("amount", "byLevel"),
    "save_bonus": ("amount", "fromAbilityModifier"),
    "resource_cost": ("amount", "variableAmount"),
}


# Effetti "delta": amount è l'unica grandezza, quindi amount 0 non cambia
# niente ed è l'assenza di effetto, non un effetto. Alla run 3 l'elfo ha preso
# uno `speed_bonus: 0` (30 feet meno i 30 di base) mentre umano, dragonide,
# mezzelfo, mezzorco e tiefling — stessa velocità, stessa frase — sono usciti
# senza effetti. Non è una differenza fra le razze, è incoerenza del modello.
NO_OP_IF_ZERO = frozenset({
    "speed_bonus", "ac_bonus", "attack_bonus", "save_bonus", "damage_bonus",
    "initiative_bonus", "hp_bonus_per_level", "ability_score_bonus",
    "spell_attack_bonus", "spell_save_dc_bonus", "check_modifier",
})


def _damage_types(path: str | Path = DEFAULT_SCHEMA_PATH) -> set[str]:
    """I 13 damage type chiusi. Lo schema li definisce in `damageTypeClosed` ma
    resistance/extra_damage_dice/spell_damage_bonus accettano ancora stringhe
    libere, e la description di damageTypeClosed dice perché è un problema:
    "una word scrapata non è un damage type". La prima run ha esportato
    `damageTypes: ["associated with your draconic ancestry"]`."""
    return set(load_schema(path)["$defs"]["damageTypeClosed"]["enum"])


def effect_kinds(def_name: str = "featureEffect",
                 path: str | Path = DEFAULT_SCHEMA_PATH) -> list[tuple[str, list[str]]]:
    """I kind ammessi da un `$defs` del compendium, con i loro campi.

    Serve a GENERARE l'elenco per il prompt invece di scriverlo a mano. La
    descrizione di `effects` ne elencava 13 su 25 — li avevo digitati io e ne
    avevo dimenticati dodici, fra cui `resource_cost`, che e` il kind piu`
    frequente nelle capacita' di classe curate. L'istruzione diceva di
    rispondere null quando nessun kind corrisponde, quindi il modello
    rispondeva null: su 61 capacita' misurate contro il compendium, 35
    sbagliavano solo per questo.

    Generandolo l'elenco e` completo per costruzione e segue la tassonomia se
    cambia, invece di restare indietro in silenzio.
    """
    defs = load_schema(path)["$defs"]
    node = defs.get(def_name, {})
    refs = [r["$ref"].split("/")[-1] for r in node.get("oneOf", []) if "$ref" in r]
    out: list[tuple[str, list[str]]] = []
    for ref in refs:
        d = defs.get("effects", {}).get(ref.split("/")[-1]) or defs.get(ref, {})
        if not isinstance(d, dict):
            continue
        props = d.get("properties", {})
        kind = (props.get("kind") or {}).get("const") or ref.rsplit("/", 1)[-1]
        out.append((kind, [k for k in props if k not in ("kind", "condition")]))
    return out


def effect_kinds_text(def_name: str = "featureEffect",
                      path: str | Path = DEFAULT_SCHEMA_PATH) -> str:
    """L'elenco dei kind formattato per il prompt: `nome(campi)`, separati."""
    return "; ".join(f"{k}({', '.join(f)})" if f else k
                     for k, f in effect_kinds(def_name, path))


def strip_defaults(value: Any, path: str | Path = DEFAULT_SCHEMA_PATH) -> Any:
    """Toglie dalle strutture del compendium le proprietà scritte esplicitamente
    al loro valore di default.

    Lo schema dichiara `condition` con default "always": ometterla e scriverla
    sono la stessa cosa, e il seed curato a mano non la scrive mai. Senza questa
    normalizzazione un `{"kind": "resistance", "damageTypes": ["fire"],
    "condition": "always"}` risulta diverso dall'identico senza condition — e
    alla run 5 sette effetti su nove sono stati contati sbagliati per questo,
    nascondendo che erano giusti.
    """
    if isinstance(value, list):
        return [strip_defaults(v, path) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: strip_defaults(v, path) for k, v in value.items()}
    props = (load_schema(path)["$defs"]["effects"]
             .get(str(value.get("kind")), {}).get("properties", {}))
    for k, spec in props.items():
        if k in out and "default" in spec and out[k] == spec["default"]:
            del out[k]
    return out


def _loader_errors(value: Any, where: str = "") -> list[str]:
    """Regole che lo schema documenta ma non codifica e che a runtime applica
    il loader del compendium. Ricorsivo: un envelope va controllato fino agli
    effects annidati nelle definizioni."""
    out: list[str] = []
    if isinstance(value, dict):
        keys = EXACTLY_ONE_OF.get(value.get("kind"))
        if keys:
            present = [k for k in keys if value.get(k) is not None]
            if len(present) != 1:
                got = ", ".join(present) if present else "nessuno dei due"
                out.append(f"{where or '<root>'}: {value['kind']} richiede "
                           f"esattamente uno tra {' e '.join(keys)} ({got})")
        amount = value.get("amount")
        if (value.get("kind") in NO_OP_IF_ZERO
                and isinstance(amount, (int, float)) and not isinstance(amount, bool)
                and amount == 0
                and not set(value) - {"kind", "condition", "amount"}):
            out.append(f"{where or '<root>'}: {value['kind']} con amount 0 non cambia "
                       f"niente; è l'assenza di effetto, non un effetto")
        known = None
        for key in ("damageType", "damageTypes"):
            raw = value.get(key)
            if raw is None:
                continue
            known = _damage_types() if known is None else known
            for dt in ([raw] if isinstance(raw, str) else raw):
                if isinstance(dt, str) and dt.lower() not in known:
                    out.append(f"{where}/{key}: {dt!r} non è uno dei 13 damage "
                               f"type; è prosa copiata dalla regola")
        for k, v in value.items():
            out.extend(_loader_errors(v, f"{where}/{k}" if where else k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out.extend(_loader_errors(v, f"{where}[{i}]"))
    return out


def validate_def(value: Any, def_name: str, path: str | Path = DEFAULT_SCHEMA_PATH) -> list[str]:
    """Errori di validazione di ``value`` contro ``$defs/<def_name>`` (vuota = ok)."""
    root = load_schema(path)
    return _errors({"$schema": root.get("$schema"), "$defs": root["$defs"],
                    "$ref": f"#/$defs/{def_name}"}, value) + _loader_errors(value)


def envelope_schema(seed_file: str, path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    """Schema dell'envelope di un file seed (es. "raceTraits.json")."""
    root = load_schema(path)
    for env in root["oneOf"]:
        if env.get("title") == seed_file:
            return {"$schema": root.get("$schema"), "$defs": root["$defs"], **env}
    raise KeyError(f"envelope {seed_file!r} non presente nel compendium")


def validate_envelope(doc: dict[str, Any], seed_file: str,
                      path: str | Path = DEFAULT_SCHEMA_PATH) -> list[str]:
    return _errors(envelope_schema(seed_file, path), doc) + _loader_errors(doc)


# ---------------------------------------------------------------------------
# Export raceTraits
# ---------------------------------------------------------------------------


def _tokens(s: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def race_trait_id(race: str, subrace: str | None, feature: str) -> str:
    """Regola dei seed: race_<razza>_<sottorazza>_<tratto>, senza ripetere
    token consecutivi ("Genasi (Fire)" + "Fire Resistance" -> race_genasi_fire_resistance)."""
    out: list[str] = []
    for tok in ["race", *_tokens(race), *_tokens(subrace), *_tokens(feature)]:
        if not out or out[-1] != tok:
            out.append(tok)
    return "_".join(out)


def _key(race: str | None, subrace: str | None, feature: str | None) -> tuple[str, ...]:
    return (" ".join(_tokens(race)), " ".join(_tokens(subrace)), " ".join(_tokens(feature)))


def export_race_traits(
    doc: dict[str, Any],
    sources: list[str],
    seed: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Documento validato (dict stile Extracted, campo ``traits``) -> envelope
    raceTraits.json. Ritorna (envelope, note).

    - tratti con effects dichiarati assenti (nessun effetto numerico) sono
      esclusi: raceTraits.json contiene solo ciò che cambia un numero;
    - l'id riusa quello del seed quando razza/sottorazza/tratto coincidono
      (gli id sono persistiti sui personaggi), altrimenti è generato;
    - le sources del seed vengono unite a quelle del libro processato."""
    known = {}
    for d in (seed or {}).get("definitions", []):
        known[_key(d.get("raceName"), d.get("subraceName"), d.get("featureName"))] = d

    notes: list[str] = []
    definitions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, trait in enumerate(doc.get("traits") or []):
        def val(name: str, trait: dict = trait) -> Any:
            leaf = trait.get(name) or {}
            return leaf.get("value") if isinstance(leaf, dict) else None

        race, subrace, feature = val("raceName"), val("subraceName"), val("featureName")
        effects = val("effects")
        if not effects:
            notes.append(f"traits[{i}] {race}/{feature}: nessun effetto numerico, escluso")
            continue
        prev = known.get(_key(race, subrace, feature))
        tid = prev["id"] if prev else race_trait_id(race, subrace, feature)
        if tid in seen:
            notes.append(f"traits[{i}] {tid}: duplicato, escluso")
            continue
        seen.add(tid)
        entry: dict[str, Any] = {"id": tid, "raceName": race}
        if subrace:
            entry["subraceName"] = subrace
        entry["featureName"] = feature
        entry["grantedAtLevel"] = val("grantedAtLevel") or 1
        entry["sources"] = sorted(set(sources) | set(prev["sources"] if prev else []))
        # senza strip il seed si riempie di `condition: "always"`, che il
        # seed curato non scrive mai: ogni riestrazione segnalerebbe
        # "effects diversi dal seed" su tratti in realtà identici
        entry["effects"] = strip_defaults(copy.deepcopy(effects))
        if prev and prev.get("notes"):
            entry["notes"] = prev["notes"]
        definitions.append(entry)
        if prev and strip_defaults(prev.get("effects")) != entry["effects"]:
            notes.append(f"{tid}: effects diversi dal seed (seed={json.dumps(prev['effects'])})")

    return {"version": 1, "definitions": definitions}, notes


__all__ = [
    "effect_kinds",
    "effect_kinds_text",
    "DEFAULT_SCHEMA_PATH",
    "load_schema",
    "decoding_schema",
    "validate_def",
    "envelope_schema",
    "validate_envelope",
    "race_trait_id",
    "export_race_traits",
]
