"""Chiamate in parallelo: il guadagno c'è e l'ordine non cambia."""
from __future__ import annotations

import time

from pipeline.parallel import in_order


def test_runs_concurrently():
    """Otto attese da 0,2s devono costare ~0,2s, non 1,6s.

    Il server vLLM è avviato con --max-num-seqs 8 ma le fasi gli mandavano una
    richiesta per volta: nel log di una enumerazione reale 183 righe di
    statistiche su 183 riportavano `Running: 1 reqs`."""
    t = time.monotonic()
    out = [r for _, _, r, _ in in_order(lambda x: (time.sleep(0.2), x)[1], range(8), 8)]
    assert out == list(range(8))
    assert time.monotonic() - t < 0.8, "le chiamate non sono andate in parallelo"


def test_results_keep_input_order_not_completion_order():
    """Il primo elemento è il più lento: deve comunque uscire per primo.

    Da questo dipende tutto il resto — deduplica degli elementi in Fase 4,
    numerazione dei task e scritture su SQLite in Fase 5 restano identiche al
    sequenziale solo se l'ordine è quello di partenza."""
    delays = {0: 0.30, 1: 0.01, 2: 0.20, 3: 0.02}

    def slow(x):
        time.sleep(delays[x])
        return x

    assert [i for i, _, _, _ in in_order(slow, range(4), 4)] == [0, 1, 2, 3]
    assert [r for _, _, r, _ in in_order(slow, range(4), 4)] == [0, 1, 2, 3]


def test_failure_is_per_item_and_does_not_stop_the_rest():
    """Una finestra che fallisce non deve portarsi via le altre: l'errore
    torna accanto al suo elemento e il chiamante decide."""
    def boom(x):
        if x == 2:
            raise ValueError("finestra rotta")
        return x * 10

    got = [(i, r, None if e is None else str(e)) for i, _, r, e in in_order(boom, range(4), 4)]
    assert got == [(0, 0, None), (1, 10, None), (2, None, "finestra rotta"), (3, 30, None)]


def test_single_worker_stays_sequential():
    """Con concorrenza 1 non si crea nessun thread: utile per isolare un
    problema senza cambiare altro."""
    seen: list[int] = []

    def note(x):
        seen.append(x)
        return x

    assert [r for _, _, r, _ in in_order(note, range(5), 1)] == [0, 1, 2, 3, 4]
    assert seen == [0, 1, 2, 3, 4]


def test_empty_input():
    assert list(in_order(lambda x: x, [], 8)) == []
