"""Test di geometry.py."""
from __future__ import annotations

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


def test_invert_deskew_roundtrip():
    bbox = (10, 10, 50, 20)
    angle = 5.0
    w, h = 200, 200
    rotated = rotate_bbox(bbox, angle, w, h)
    back = invert_deskew_bbox(rotated, angle, w, h)
    # approssimazione AABB: non è esatto ma vicino per piccoli angoli
    assert abs(back[0] - bbox[0]) < 5
    assert abs(back[2] - bbox[2]) < 5


def test_union_bbox():
    assert union_bbox((0, 0, 10, 10), (5, 5, 20, 20)) == (0, 0, 20, 20)


def test_expand_clamped():
    assert expand((5, 5, 10, 10), 100, 20, 20) == (0.0, 0.0, 20.0, 20.0)
