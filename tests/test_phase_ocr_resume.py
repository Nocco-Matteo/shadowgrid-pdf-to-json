"""Test kill-safety OCR A/B e del testo canonico riconciliato (P1-4/P1-5)."""
from __future__ import annotations

import pytest

from pipeline.config import Settings
from pipeline.db import DB
from pipeline.ocr_clients import Region


@pytest.fixture
def env(tmp_path):
    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "work")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    # PNG reale: i crop di risoluzione conflitti la leggono con cv2
    from PIL import Image

    img_path = tmp_path / "p1.png"
    Image.new("RGB", (200, 100), "white").save(img_path)
    db = DB(s)
    db.upsert_document("d", "/a.pdf", "sha", 1)
    db.set_status("d", "rasterized")
    db.set_page("d", 1, image_path=str(img_path), dpi=300, deskew_angle=0.0)
    yield db, s
    db.close()


def _regions(n: int, prefix: str = "riga") -> list[Region]:
    return [
        Region(page_no=1, bbox=(0, i * 10, 100, i * 10 + 10),
               region_type="text", text=f"{prefix} {i}", order_idx=i)
        for i in range(n)
    ]


class CountingOCR:
    """OCR stub che simula un'interruzione dopo la prima regione salvata."""

    def __init__(self, n_regions: int):
        self.n = n_regions
        self.calls = 0

    def ocr_page(self, image_path, page_no):
        self.calls += 1
        return _regions(self.n)


def test_phase2_partial_page_is_reocrd(env):
    """Kill dopo la prima regione di due: la pagina NON è completa, al resume
    viene ri-OCR-ata per intero (full_text scritto solo col commit atomico)."""
    db, s = env
    from pipeline import phase2_ocr_a

    client = CountingOCR(2)
    phase2_ocr_a.run("d", db=db, settings=s, client=client)
    assert client.calls == 1
    assert len(db.get_regions("d", 1, engine="a")) == 2
    assert db.get_page("d", 1)["full_text"] == "riga 0\nriga 1"

    # stato simulato di interruzione: una regione salvata, full_text NULL
    db.set_status("d", "ocr_a")
    db.set_status("d", "rasterized")
    db.clear_regions("d", 1, "a")
    db.add_region("d", 1, (0, 0, 100, 10), "text", "riga 0", "a", 0)
    page = db.get_page("d", 1)
    db.conn.execute(
        "UPDATE pages SET full_text=NULL, canonical_text=NULL WHERE doc_id='d' AND page_no=1"
    )
    assert page is not None

    client2 = CountingOCR(2)
    phase2_ocr_a.run("d", db=db, settings=s, client=client2)
    # la pagina incompleta è stata ri-OCR-ata, non saltata
    assert client2.calls == 1
    assert len(db.get_regions("d", 1, engine="a")) == 2
    assert db.get_status("d") == "ocr_a"


def test_phase2_completed_page_not_reocrd(env):
    db, s = env
    from pipeline import phase2_ocr_a

    client = CountingOCR(2)
    phase2_ocr_a.run("d", db=db, settings=s, client=client)
    db.set_status("d", "rasterized")  # forza ri-run
    phase2_ocr_a.run("d", db=db, settings=s, client=client)
    assert client.calls == 1  # pagina completa -> mai ri-OCR-ata


def test_phase2_empty_page_completes(env):
    """Pagina vuota: zero regioni ma pagina completa (full_text vuoto, non NULL)."""
    db, s = env
    from pipeline import phase2_ocr_a

    class Empty:
        calls = 0

        def ocr_page(self, image_path, page_no):
            Empty.calls += 1
            return []

    phase2_ocr_a.run("d", db=db, settings=s, client=Empty())
    page = db.get_page("d", 1)
    assert page["full_text"] == ""
    assert page["full_text"] is not None
    assert db.get_status("d") == "ocr_a"
    db.set_status("d", "rasterized")
    phase2_ocr_a.run("d", db=db, settings=s, client=Empty())
    assert Empty.calls == 1  # non ri-OCR-ata


def test_phase2_full_text_includes_tables(env):
    """Una citazione presente solo in tabella deve essere nel full_text."""
    db, s = env
    from pipeline import phase2_ocr_a

    regions = [
        Region(page_no=1, bbox=(0, 0, 100, 10), region_type="text",
               text="CONTRATTO N. 44/B", order_idx=0),
        Region(page_no=1, bbox=(0, 20, 100, 60), region_type="table",
               text="<table><tr><td>Mario Rossi</td></tr></table>", order_idx=1),
    ]

    class Stub:
        def ocr_page(self, image_path, page_no):
            return regions

    phase2_ocr_a.run("d", db=db, settings=s, client=Stub())
    full_text = db.get_page("d", 1)["full_text"]
    assert "44/B" in full_text
    assert "Mario Rossi" in full_text  # tabella inclusa


def test_phase3_b_partial_page_reocrd(env):
    """Kill a metà OCR B: al resume la pagina B viene ri-fatta per intero."""
    db, s = env
    from pipeline import phase2_ocr_a, phase3_ocr_b

    phase2_ocr_a.run("d", db=db, settings=s, client=CountingOCR(2))
    assert db.get_status("d") == "ocr_a"

    class StubB:
        calls = 0

        def ocr_page(self, image_path, page_no):
            StubB.calls += 1
            return _regions(2)

    class NoResolver:
        def read_crop(self, image_path):
            raise AssertionError("nessun conflitto atteso")

    phase3_ocr_b.run("d", db=db, settings=s, client_b=StubB(), resolver=NoResolver())
    assert StubB.calls == 1
    assert len(db.get_regions("d", 1, engine="b")) == 2
    assert db.get_page("d", 1)["ocr_b_done"] == 1
    assert db.get_status("d") == "reconciled"

    # interruzione simulata: regioni B a metà senza flag
    db.set_status("d", "ocr_a")
    db.clear_regions("d", 1, "b")
    db.add_region("d", 1, (0, 0, 100, 10), "text", "riga 0", "b", 0)
    db.conn.execute("UPDATE pages SET ocr_b_done=0 WHERE doc_id='d' AND page_no=1")

    StubB.calls = 0
    phase3_ocr_b.run("d", db=db, settings=s, client_b=StubB(), resolver=NoResolver())
    assert StubB.calls == 1  # ri-OCR-ata
    assert len(db.get_regions("d", 1, engine="b")) == 2


def test_reconciled_text_is_canonical_downstream(env):
    """Il vincitore majority diventa il testo usato a valle; A e B restano originali."""
    pytest.importorskip("cv2")
    db, s = env
    from pipeline import phase2_ocr_a, phase3_ocr_b

    # OCR A legge male, OCR B legge bene
    regions_a = [Region(page_no=1, bbox=(0, 0, 100, 10), region_type="text",
                        text="Impofto: 100 EUR", order_idx=0)]
    regions_b = [Region(page_no=1, bbox=(0, 0, 100, 10), region_type="text",
                        text="Importo: 100 EUR", order_idx=0)]

    class StubA:
        def ocr_page(self, image_path, page_no):
            return regions_a

    class StubB:
        def ocr_page(self, image_path, page_no):
            return regions_b

    class Resolver:
        def read_crop(self, image_path):
            return "Importo: 100 EUR"  # voto 2-su-3 con B

    phase2_ocr_a.run("d", db=db, settings=s, client=StubA())
    phase3_ocr_b.run("d", db=db, settings=s, client_b=StubB(), resolver=Resolver())

    # originali preservati
    assert db.get_regions("d", 1, engine="a")[0]["text"] == "Impofto: 100 EUR"
    assert db.get_regions("d", 1, engine="b")[0]["text"] == "Importo: 100 EUR"
    # conflitto registrato con majority
    conflicts = db.get_conflicts("d")
    assert conflicts[0]["resolver"] == "majority"
    assert conflicts[0]["resolved_text"] == "Importo: 100 EUR"
    # il testo canonico usato a valle è quello riconciliato
    assert db.get_page("d", 1)["canonical_text"] == "Importo: 100 EUR"
    assert db.get_page_text("d", 1) == "Importo: 100 EUR"
    assert db.get_canonical_regions("d", 1)[0]["text"] == "Importo: 100 EUR"


def test_divergent_conflict_keeps_a_text(env):
    """Tutti divergenti: canonico resta A, conflitto in coda umana."""
    pytest.importorskip("cv2")
    db, s = env
    from pipeline import phase2_ocr_a, phase3_ocr_b

    regions_a = [Region(page_no=1, bbox=(0, 0, 100, 10), region_type="text",
                        text="testo A", order_idx=0)]
    regions_b = [Region(page_no=1, bbox=(0, 0, 100, 10), region_type="text",
                        text="testo B", order_idx=0)]

    class StubA:
        def ocr_page(self, image_path, page_no):
            return regions_a

    class StubB:
        def ocr_page(self, image_path, page_no):
            return regions_b

    class Resolver:
        def read_crop(self, image_path):
            return "testo C del resolver"

    phase2_ocr_a.run("d", db=db, settings=s, client=StubA())
    phase3_ocr_b.run("d", db=db, settings=s, client_b=StubB(), resolver=Resolver())
    assert db.get_conflicts("d")[0]["resolver"] == "divergent_low"
    assert db.get_page_text("d", 1) == "testo A"


def test_interrupted_majority_repaired_on_resume(env):
    """Punto d'interruzione di una versione precedente: conflitto registrato
    con resolved_text majority ma canonico mai aggiornato. Al resume la
    risoluzione persistita viene riapplicata (e non risolta due volte)."""
    db, s = env
    from pipeline import phase2_ocr_a, phase3_ocr_b

    regions_a = [Region(page_no=1, bbox=(0, 0, 100, 10), region_type="text",
                        text="Impofto: 100 EUR", order_idx=0)]
    regions_b = [Region(page_no=1, bbox=(0, 0, 100, 10), region_type="text",
                        text="Importo: 100 EUR", order_idx=0)]

    class StubA:
        def ocr_page(self, image_path, page_no):
            return regions_a

    class StubB:
        def ocr_page(self, image_path, page_no):
            return regions_b

    phase2_ocr_a.run("d", db=db, settings=s, client=StubA())
    rid = db.get_regions("d", 1, engine="a")[0]["region_id"]

    # interruzione esatta: conflitto registrato, canonico stantio
    db.add_conflict("d", rid, "Impofto: 100 EUR", "Importo: 100 EUR",
                    "Importo: 100 EUR", "majority")
    assert db.get_page_text("d", 1) == "Impofto: 100 EUR"

    class NoResolver:
        def read_crop(self, image_path):
            raise AssertionError("conflitto già registrato: mai risolto due volte")

    phase3_ocr_b.run("d", db=db, settings=s, client_b=StubB(), resolver=NoResolver())
    # la risoluzione persistita è stata riapplicata al testo canonico
    assert db.get_page_text("d", 1) == "Importo: 100 EUR"
    assert db.get_canonical_regions("d", 1)[0]["text"] == "Importo: 100 EUR"


# ---------------------------------------------------------------------------
# reorder: riapplica l'ordine di lettura senza rifare l'OCR
# ---------------------------------------------------------------------------


def test_reorder_fixes_order_and_rebuilds_page_text(tmp_path):
    """Le bbox sono già in DB: cambiare reading_order non deve costare un
    nuovo OCR. Riordina le regioni e ricostruisce full_text/canonical_text."""
    from pipeline.config import Settings
    from pipeline.db import DB
    from pipeline.phase2_ocr_a import reorder

    s = Settings(db_path=tmp_path / "t.db", work_dir=tmp_path / "w")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    db = DB(s)
    db.upsert_document("doc_r", "/r.pdf", "sha_r", 1)
    db.set_page("doc_r", 1, full_text="sbagliato", image_path=None)
    # il motore emette il titolo della seconda sezione PRIMA del paragrafo che
    # in pagina lo precede: è il caso Hill/Mountain Dwarf
    db.add_region("doc_r", 1, (0, 0, 100, 20), "text", "TITOLO A", "a", 0)
    db.add_region("doc_r", 1, (0, 200, 100, 220), "text", "TITOLO B", "a", 1)
    db.add_region("doc_r", 1, (0, 100, 100, 120), "text", "paragrafo di A", "a", 2)

    assert reorder("doc_r", db=db) == 1
    rows = db.get_regions("doc_r", 1, engine="a")
    assert [r["text"] for r in rows] == ["TITOLO A", "paragrafo di A", "TITOLO B"]
    assert [r["order_idx"] for r in rows] == [0, 1, 2]
    assert db.get_page_text("doc_r", 1) == "TITOLO A\nparagrafo di A\nTITOLO B"
    # idempotente: alla seconda passata non c'è più niente da cambiare
    assert reorder("doc_r", db=db) == 0
    db.close()
