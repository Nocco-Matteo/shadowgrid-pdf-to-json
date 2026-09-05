"""SQLite in WAL + macchina a stati.

Regola non negoziabile: ogni fase legge lo stato e salta ciò che è già fatto.
Devi poter uccidere il processo in qualsiasi momento e ripartire senza perdere lavoro.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any

from .config import Settings, get_settings

# Macchina a stati
STATES = [
    "ingested",
    "rasterized",
    "ocr_a",
    "ocr_b",
    "reconciled",
    "enumerated",
    "extracted",
    "validated",
    "done",
]
TERMINAL_ALT = {"needs_review", "failed"}

# Archi ammessi (src -> dst)
EDGES: dict[str, set[str]] = {
    "ingested": {"rasterized"},
    "rasterized": {"ocr_a"},
    "ocr_a": {"ocr_b", "reconciled"},  # si può saltare a reconciled se niente OCR B
    "ocr_b": {"reconciled"},
    "reconciled": {"enumerated"},
    "enumerated": {"extracted"},
    "extracted": {"validated"},
    "validated": {"done", "needs_review"},
    "needs_review": {"done", "failed"},
}
# Da qualsiasi stato non terminale si può andare in failed
for s in STATES:
    EDGES.setdefault(s, set()).add("failed")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents(
  doc_id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE,
  n_pages INTEGER,
  status TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pages(
  doc_id TEXT NOT NULL,
  page_no INTEGER NOT NULL,
  image_path TEXT,
  dpi INTEGER,
  deskew_angle REAL,
  full_text TEXT,
  PRIMARY KEY (doc_id, page_no),
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
CREATE TABLE IF NOT EXISTS regions(
  region_id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT NOT NULL,
  page_no INTEGER NOT NULL,
  bbox TEXT,            -- JSON [x0,y0,x1,y1]
  region_type TEXT,     -- text|table|formula|stamp
  text TEXT,
  engine TEXT,          -- a|b|resolver
  order_idx INTEGER,
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
CREATE INDEX IF NOT EXISTS idx_regions_doc_page ON regions(doc_id, page_no);
CREATE TABLE IF NOT EXISTS region_conflicts(
  region_id INTEGER PRIMARY KEY,
  text_a TEXT,
  text_b TEXT,
  resolved_text TEXT,
  resolver TEXT
);
CREATE TABLE IF NOT EXISTS extractions(
  doc_id TEXT NOT NULL,
  field_path TEXT NOT NULL,
  value_json TEXT,
  quote TEXT,
  page_no INTEGER,
  bbox TEXT,
  attempt INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'pending',  -- pending|validated|rejected|needs_review
  confidence TEXT,
  PRIMARY KEY (doc_id, field_path, attempt)
);
CREATE INDEX IF NOT EXISTS idx_extractions_doc ON extractions(doc_id);
CREATE TABLE IF NOT EXISTS runs(
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  git_sha TEXT,
  model TEXT,
  prompt_version TEXT,
  started_at REAL NOT NULL,
  finished_at REAL,
  metrics_json TEXT
);
"""


class DB:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.conn = sqlite3.connect(self.s.db_path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self):
        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            yield cur
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    # --- documents --------------------------------------------------------
    def upsert_document(self, doc_id: str, path: str, sha256: str, n_pages: int | None) -> None:
        with self.tx() as cur:
            cur.execute(
                """INSERT INTO documents(doc_id, path, sha256, n_pages, status, created_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(doc_id) DO UPDATE SET path=excluded.path""",
                (doc_id, str(path), sha256, n_pages, "ingested", time.time()),
            )

    def get_document(self, doc_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()

    def find_by_sha256(self, sha256: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM documents WHERE sha256=?", (sha256,)
        ).fetchone()

    def get_status(self, doc_id: str) -> str | None:
        row = self.get_document(doc_id)
        return row["status"] if row else None

    def transition(self, doc_id: str, src: str, dst: str) -> None:
        if dst not in EDGES.get(src, set()):
            raise ValueError(f"Transizione non ammessa: {src} -> {dst}")
        with self.tx() as cur:
            cur.execute(
                "UPDATE documents SET status=? WHERE doc_id=? AND status=?",
                (dst, doc_id, src),
            )
            if cur.rowcount == 0:
                raise ValueError(
                    f"Impossibile transire {src}->{dst}: doc {doc_id} non in stato {src}"
                )

    def set_status(self, doc_id: str, dst: str) -> None:
        # Forza lo stato (usato per needs_review/failed)
        with self.tx() as cur:
            cur.execute("UPDATE documents SET status=? WHERE doc_id=?", (dst, doc_id))

    def get_pending(self, status: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM documents WHERE status=?", (status,)
        ).fetchall()

    # --- pages ------------------------------------------------------------
    def set_page(
        self,
        doc_id: str,
        page_no: int,
        image_path: str | None = None,
        dpi: int | None = None,
        deskew_angle: float | None = None,
        full_text: str | None = None,
    ) -> None:
        with self.tx() as cur:
            cur.execute(
                """INSERT INTO pages(doc_id, page_no, image_path, dpi, deskew_angle, full_text)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(doc_id, page_no) DO UPDATE SET
                     image_path=COALESCE(excluded.image_path, pages.image_path),
                     dpi=COALESCE(excluded.dpi, pages.dpi),
                     deskew_angle=COALESCE(excluded.deskew_angle, pages.deskew_angle),
                     full_text=COALESCE(excluded.full_text, pages.full_text)""",
                (doc_id, page_no, image_path, dpi, deskew_angle, full_text),
            )

    def get_page(self, doc_id: str, page_no: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM pages WHERE doc_id=? AND page_no=?", (doc_id, page_no)
        ).fetchone()

    def get_pages(self, doc_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM pages WHERE doc_id=? ORDER BY page_no", (doc_id,)
        ).fetchall()

    # --- regions ----------------------------------------------------------
    def add_region(
        self,
        doc_id: str,
        page_no: int,
        bbox: tuple[float, float, float, float] | None,
        region_type: str,
        text: str,
        engine: str,
        order_idx: int,
    ) -> int:
        with self.tx() as cur:
            cur.execute(
                """INSERT INTO regions(doc_id, page_no, bbox, region_type, text, engine, order_idx)
                   VALUES(?,?,?,?,?,?,?)""",
                (doc_id, page_no, json.dumps(bbox) if bbox else None, region_type, text, engine, order_idx),
            )
            return cur.lastrowid

    def clear_regions(self, doc_id: str, page_no: int, engine: str) -> None:
        with self.tx() as cur:
            cur.execute(
                "DELETE FROM regions WHERE doc_id=? AND page_no=? AND engine=?",
                (doc_id, page_no, engine),
            )

    def get_regions(self, doc_id: str, page_no: int, engine: str | None = None) -> list[sqlite3.Row]:
        if engine:
            return self.conn.execute(
                "SELECT * FROM regions WHERE doc_id=? AND page_no=? AND engine=? ORDER BY order_idx",
                (doc_id, page_no, engine),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM regions WHERE doc_id=? AND page_no=? ORDER BY engine, order_idx",
            (doc_id, page_no),
        ).fetchall()

    def set_page_full_text(self, doc_id: str, page_no: int, full_text: str) -> None:
        self.set_page(doc_id, page_no, full_text=full_text)

    # --- conflicts --------------------------------------------------------
    def add_conflict(
        self, region_id: int, text_a: str, text_b: str, resolved_text: str, resolver: str
    ) -> None:
        with self.tx() as cur:
            cur.execute(
                """INSERT INTO region_conflicts(region_id, text_a, text_b, resolved_text, resolver)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(region_id) DO UPDATE SET
                     text_a=excluded.text_a, text_b=excluded.text_b,
                     resolved_text=excluded.resolved_text, resolver=excluded.resolver""",
                (region_id, text_a, text_b, resolved_text, resolver),
            )

    # --- extractions ------------------------------------------------------
    def upsert_extraction(
        self,
        doc_id: str,
        field_path: str,
        value_json: str | None,
        quote: str | None,
        page_no: int | None,
        bbox: tuple[float, float, float, float] | None,
        attempt: int,
        status: str,
        confidence: str | None = None,
    ) -> None:
        with self.tx() as cur:
            cur.execute(
                """INSERT INTO extractions(doc_id, field_path, value_json, quote, page_no, bbox,
                                           attempt, status, confidence)
                   VALUES(?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(doc_id, field_path, attempt) DO UPDATE SET
                     value_json=excluded.value_json, quote=excluded.quote,
                     page_no=excluded.page_no, bbox=excluded.bbox,
                     status=excluded.status, confidence=excluded.confidence""",
                (doc_id, field_path, value_json, quote, page_no,
                 json.dumps(bbox) if bbox else None, attempt, status, confidence),
            )

    def get_extractions(self, doc_id: str, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self.conn.execute(
                "SELECT * FROM extractions WHERE doc_id=? AND status=? ORDER BY field_path, attempt",
                (doc_id, status),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM extractions WHERE doc_id=? ORDER BY field_path, attempt",
            (doc_id,),
        ).fetchall()

    def latest_extraction(self, doc_id: str, field_path: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM extractions WHERE doc_id=? AND field_path=? ORDER BY attempt DESC LIMIT 1",
            (doc_id, field_path),
        ).fetchone()

    # --- runs -------------------------------------------------------------
    def start_run(self, git_sha: str, model: str, prompt_version: str) -> int:
        with self.tx() as cur:
            cur.execute(
                "INSERT INTO runs(git_sha, model, prompt_version, started_at) VALUES(?,?,?,?)",
                (git_sha, model, prompt_version, time.time()),
            )
            return cur.lastrowid

    def finish_run(self, run_id: int, metrics: dict[str, Any] | None = None) -> None:
        with self.tx() as cur:
            cur.execute(
                "UPDATE runs SET finished_at=?, metrics_json=? WHERE run_id=?",
                (time.time(), json.dumps(metrics) if metrics else None, run_id),
            )

    def last_run(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runs WHERE finished_at IS NOT NULL ORDER BY run_id DESC LIMIT 1"
        ).fetchone()


def skip_if_done(db: DB, doc_id: str, required: str, next_state: str) -> bool:
    """Ritorna True se la fase va saltata (già fatta). Se lo stato è precedente a
    quello richiesto, lancia (la fase precedente non è stata eseguita)."""
    status = db.get_status(doc_id)
    if status is None:
        raise ValueError(f"Documento {doc_id} sconosciuto")
    if status == next_state or status in STATES and STATES.index(status) > STATES.index(next_state):
        return True
    if status != required:
        raise ValueError(
            f"Stato atteso {required} per transire a {next_state}, trovato {status}"
        )
    return False


__all__ = ["DB", "STATES", "EDGES", "skip_if_done"]
