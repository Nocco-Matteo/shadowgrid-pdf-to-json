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
