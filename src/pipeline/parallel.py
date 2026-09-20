"""Chiamate al modello in parallelo, risultati in ordine.

Il server vLLM è avviato con `--max-num-seqs 8` ma le fasi gli mandavano una
richiesta per volta: nel log di una enumerazione reale 183 righe di statistiche
su 183 riportavano `Running: 1 reqs`. A flusso singolo la decodifica di un 27B
è limitata dalla banda di memoria — per ogni token si rileggono tutti i pesi —
e si ferma sui 18 token/s. In batch quel costo si ammortizza sulle richieste
contemporanee e la resa aggregata sale quasi linearmente.

Regola che questo modulo impone: si parallelizza SOLO la chiamata di rete. I
risultati tornano nell'ordine di partenza, così tutto ciò che dipende
dall'ordine — deduplica degli elementi, numerazione dei task, scritture su
SQLite — resta seriale e il risultato è identico a quello sequenziale.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


def in_order(
    fn: Callable[[T], Any], items: Iterable[T], workers: int
) -> Iterator[tuple[int, T, Any, Exception | None]]:
    """Esegue `fn` su ogni elemento con al più `workers` in volo e restituisce
    `(indice, elemento, risultato, eccezione)` NELL'ORDINE di partenza.

    Il consumo in ordine non serializza il lavoro: mentre si attende il
    risultato i-esimo gli altri continuano. Serve a mantenere l'elaborazione
    incrementale — un elemento alla volta, appena il suo turno è pronto —
    invece di aspettare che finiscano tutti, così un'interruzione a metà lascia
    dietro di sé il lavoro già registrato.
    """
    elems = list(items)
    if not elems:
        return
    if workers <= 1 or len(elems) == 1:
        for i, it in enumerate(elems):
            try:
                yield i, it, fn(it), None
            except Exception as e:  # noqa: BLE001 - l'errore lo gestisce il chiamante
                yield i, it, None, e
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, it) for it in elems]
        for i, (it, fut) in enumerate(zip(elems, futures, strict=True)):
            try:
                yield i, it, fut.result(), None
            except Exception as e:  # noqa: BLE001
                yield i, it, None, e


def _hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class Progress:
    """Avanzamento per i log: `12/46 26% · 3m12s · ~8m40s rimasti`.

    Serve perché una fase lunga scrive una riga per unità di lavoro e basta:
    con 46 finestre si conta a mano, con 400 task di estrazione non si conta
    affatto. La stima del rimanente è la media dei completati finora — con le
    richieste in parallelo le prime unità sembrano lente (si attende che il
    primo lotto si riempia), quindi all'inizio sovrastima.
    """

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.start = time.monotonic()

    def step(self) -> str:
        self.done += 1
        elapsed = time.monotonic() - self.start
        pct = 100 * self.done / self.total if self.total else 100
        out = f"{self.done}/{self.total} {pct:.0f}% · {_hms(elapsed)}"
        if self.done < self.total and self.done:
            out += f" · ~{_hms(elapsed / self.done * (self.total - self.done))} rimasti"
        return out


__all__ = ["in_order", "Progress"]
