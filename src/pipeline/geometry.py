"""Utility geometriche: rotazione bbox e riproiezione inversa del deskew."""

from __future__ import annotations

import math

BBox = tuple[float, float, float, float]  # (x0, y0, x1, y1)


def _bbox_corners(bbox: BBox) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = bbox
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _aabb(points: list[tuple[float, float]]) -> BBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def rotate_bbox(bbox: BBox, angle_deg: float, img_w: int, img_h: int) -> BBox:
    """Ruota una bbox attorno al centro dell'immagine di `angle_deg` (gradi),
    in convenzione matematica (antioraria con y verso l'alto). Attenzione: è
    la rotazione OPPOSTA a cv2.getRotationMatrix2D con lo stesso angolo.
    Restituisce l'AABB del rettangolo ruotato."""
    if angle_deg == 0.0:
        return bbox
    cx, cy = img_w / 2.0, img_h / 2.0
    theta = math.radians(angle_deg)
    cos, sin = math.cos(theta), math.sin(theta)

    def rot(px: float, py: float) -> tuple[float, float]:
        dx, dy = px - cx, py - cy
        return (cx + dx * cos - dy * sin, cy + dx * sin + dy * cos)

    return _aabb([rot(x, y) for x, y in _bbox_corners(bbox)])


def invert_deskew_bbox(
    bbox: BBox, deskew_angle: float, img_w: int, img_h: int
) -> BBox:
    """Riproietta una bbox (sull'immagine deskewed, cioè ruotata in Fase 1 con
    cv2.getRotationMatrix2D(center, deskew_angle)) sulle coordinate
    dell'immagine originale.

    cv2.getRotationMatrix2D(θ) mappa (dx,dy) -> (cos·dx + sin·dy, -sin·dx + cos·dy);
    la sua inversa (θ -> -θ) mappa (dx,dy) -> (cos·dx - sin·dy, sin·dx + cos·dy),
    che è esattamente rotate_bbox(+θ) per la convenzione di segno opposta di
    rotate_bbox. Quindi l'inversione è rotate_bbox con l'angolo NON negato
    (equivalente a cv2.invertAffineTransform della matrice di deskew).

    Nota: la pipeline ora usa un unico sistema di coordinate (quello
    dell'immagine deskewed salvata su disco), quindi questa funzione serve
    solo per mappare verso le coordinate pre-deskew quando servono."""
    return rotate_bbox(bbox, deskew_angle, img_w, img_h)


def iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def union_bbox(a: BBox, b: BBox) -> BBox:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def expand(bbox: BBox, margin_px: int, img_w: int, img_h: int) -> BBox:
    x0, y0, x1, y1 = bbox
    return (
        max(0.0, x0 - margin_px),
        max(0.0, y0 - margin_px),
        min(float(img_w), x1 + margin_px),
        min(float(img_h), y1 + margin_px),
    )


__all__ = [
    "BBox",
    "rotate_bbox",
    "invert_deskew_bbox",
    "iou",
    "union_bbox",
    "expand",
]
