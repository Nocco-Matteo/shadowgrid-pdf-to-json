"""Normalizzazione testo per fuzzy match (Fase 3 e Fase 6)."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(s: str) -> str:
    """lowercase, collassa spazi, unifica punteggiatura, rimuove diacritici."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()


def similarity(a: str, b: str) -> float:
    """Similarità normalizzata in [0,1] tra due testi."""
    from rapidfuzz.fuzz import ratio

    return ratio(normalize(a), normalize(b)) / 100.0


__all__ = ["normalize", "similarity"]
