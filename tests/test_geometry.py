"""Test di geometry.py."""
from __future__ import annotations

import pytest

from pipeline.geometry import (
    expand,
    invert_deskew_bbox,
    iou,
    rotate_bbox,
    union_bbox,
)


def test_iou_identical():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial():
    v = iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.1 < v < 0.4


def test_rotate_zero_idempotent():
    assert rotate_bbox((1, 2, 3, 4), 0.0, 100, 100) == (1, 2, 3, 4)


def test_invert_deskew_matches_opencv_inverse():
    """invert_deskew_bbox deve coincidere con l'inversa della matrice di
    deskew di OpenCV (cv2.invertAffineTransform), stessa convenzione di segno."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    w, h = 200, 240
    angle = 5.0
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    Minv = cv2.invertAffineTransform(np.asarray(M))
    bbox = (10, 10, 50, 20)
    x0, y0, x1, y1 = bbox
    pts = np.array(
        [[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], dtype=float
    ) @ Minv.T
    expected = (
        float(pts[:, 0].min()), float(pts[:, 1].min()),
        float(pts[:, 0].max()), float(pts[:, 1].max()),
    )
    got = invert_deskew_bbox(bbox, angle, w, h)
    assert all(abs(g - e) < 1e-6 for g, e in zip(got, expected, strict=True))


def test_invert_deskew_not_forward_rotation():
    """L'inversione non deve riapplicare la rotazione forward (vecchio bug)."""
    bbox = (10, 10, 50, 20)
    angle = 5.0
    w, h = 200, 200
    got = invert_deskew_bbox(bbox, angle, w, h)
    wrong = rotate_bbox(bbox, -angle, w, h)  # vecchia implementazione buggata
    assert got != wrong


def test_union_bbox():
    assert union_bbox((0, 0, 10, 10), (5, 5, 20, 20)) == (0, 0, 20, 20)


def test_expand_clamped():
    assert expand((5, 5, 10, 10), 100, 20, 20) == (0.0, 0.0, 20.0, 20.0)


# ---------------------------------------------------------------------------
# Ordine di lettura
# ---------------------------------------------------------------------------

from pipeline.geometry import reading_order  # noqa: E402


def test_reading_order_single_column_unchanged():
    boxes = [(0, 0, 100, 20), (0, 30, 100, 50), (0, 60, 100, 80)]
    assert reading_order(boxes) == [0, 1, 2]


def test_reading_order_fixes_vertical_inversion_in_column():
    """Il caso che conta: due blocchi della STESSA colonna emessi al contrario."""
    boxes = [(0, 0, 100, 20), (0, 60, 100, 80), (0, 30, 100, 50)]
    assert reading_order(boxes) == [0, 2, 1]


def test_reading_order_two_columns_not_interleaved():
    """Colonna sinistra completa, poi destra: mai alternate."""
    boxes = [
        (0, 0, 100, 20), (0, 40, 100, 60),        # sinistra
        (200, 0, 300, 20), (200, 40, 300, 60),    # destra
    ]
    assert reading_order(boxes) == [0, 1, 2, 3]


def test_reading_order_keeps_engine_column_order():
    """L'ordine FRA le colonne resta quello del motore: una regione isolata a
    sinistra (numero di pagina, nota di margine) non scavalca il corpo."""
    boxes = [
        (200, 0, 300, 20),      # corpo, emesso per primo
        (200, 40, 300, 60),
        (0, 500, 40, 520),      # margine sinistro, emesso per ultimo
    ]
    assert reading_order(boxes) == [0, 1, 2]


def test_reading_order_full_width_block_separates_sections():
    """Un blocco a tutta larghezza chiude la sezione sopra e apre quella sotto."""
    boxes = [
        (0, 0, 100, 20), (200, 0, 300, 20),       # due colonne, fascia alta
        (0, 40, 300, 60),                          # banda a tutta larghezza
        (0, 80, 100, 100), (200, 80, 300, 100),    # due colonne, fascia bassa
    ]
    assert reading_order(boxes) == [0, 1, 2, 3, 4]


def test_reading_order_footer_stays_last():
    """Il piè di pagina cade nella colonna sinistra per posizione: senza
    page_height finisce a metà pagina, con page_height resta in fondo."""
    boxes = [
        (0, 100, 100, 120),      # colonna sinistra
        (200, 100, 300, 120),    # colonna destra
        (0, 960, 120, 980),      # piè di pagina
    ]
    assert reading_order(boxes) == [0, 2, 1]
    assert reading_order(boxes, page_height=1000) == [0, 1, 2]


def test_reading_order_without_bbox_keeps_engine_order():
    boxes = [(0, 60, 100, 80), None, (0, 0, 100, 20)]
    assert reading_order(boxes) == [0, 1, 2]


def test_reading_order_phb_page16_hill_vs_mountain_dwarf():
    """Regressione dalla run reale sul Player's Handbook p.16.

    PaddleOCR-VL ha emesso il titolo MOUNTAIN DWARF (y=1478) PRIMA del tratto
    Dwarven Toughness (y=1308), che in pagina gli sta sopra e appartiene quindi
    a HILL DWARF. La Fase 4 ha attribuito il tratto al titolo che lo precedeva
    nel testo e l'export ne ha fatto `race_dwarf_mountain_dwarven_toughness`:
    validato, confidence alta, e sbagliato (è un tratto Hill Dwarf).
    """
    # (id, bbox) come in DB, nell'ordine emesso dal motore
    regions = [
        (457, (1431, 934, 2380, 970)),    # HILL DWARF
        (459, (1429, 985, 2380, 1190)),   # As a hill dwarf, ...
        (460, (1431, 1215, 2380, 1250)),  # Ability Score Increase (Wis)
        (461, (1434, 1478, 2380, 1514)),  # MOUNTAIN DWARF
        (462, (1430, 1308, 2380, 1440)),  # Dwarven Toughness
        (463, (1429, 1528, 2380, 1760)),  # As a mountain dwarf, ...
    ]
    order = [regions[i][0] for i in reading_order([b for _, b in regions],
                                                  page_height=3300)]
    assert order == [457, 459, 460, 462, 461, 463]
    assert order.index(462) < order.index(461)  # Toughness sotto HILL DWARF


def test_empty_region_does_not_weld_two_columns():
    """Regressione dal Player's Handbook p.41 (apertura del capitolo Barbarian).

    Una regione SENZA TESTO larga il 58,8% della pagina — appena sotto la soglia
    di spanner — cominciava dove finiva la colonna sinistra e arrivava in fondo
    alla destra. Nel raggruppamento per sovrapposizione orizzontale faceva da
    ponte: le due colonne diventavano una, la pagina veniva ordinata per y e il
    testo usciva interlacciato ('Quick Build' in mezzo alla colonna sinistra).

    Nel PHB le regioni vuote sono 402 su 9.535: ognuna è un ponte potenziale.
    """
    boxes = [
        (176, 200, 1139, 260),    # 0 sinistra
        (1139, 3, 2515, 60),      # 1 VUOTA, scavalca il corridoio
        (180, 2206, 1115, 2280),  # 2 sinistra
        (1228, 1994, 1518, 2040),  # 3 destra
        (1223, 2049, 2161, 2140),  # 4 destra
    ]
    flow = [True, False, True, True, True]
    order = reading_order(boxes, page_height=3300, in_flow=flow)
    assert order.index(0) < order.index(2) < order.index(3), "colonne interlacciate"
    assert order.index(2) < order.index(3), "la sinistra va chiusa prima della destra"
    assert order[-1] == 1, "la regione senza testo va in coda, non nel flusso"

    # senza in_flow il ponte fa ancora danno: è il comportamento che si correggeva
    assert reading_order(boxes, page_height=3300) != order


def test_in_flow_ignored_when_everything_has_text():
    boxes = [(0, 0, 100, 20), (0, 60, 100, 80), (0, 30, 100, 50)]
    assert (reading_order(boxes, in_flow=[True, True, True])
            == reading_order(boxes) == [0, 2, 1])
