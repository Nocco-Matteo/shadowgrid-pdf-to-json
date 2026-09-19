"""Utility geometriche: rotazione bbox, riproiezione inversa del deskew, ordine di lettura."""

from __future__ import annotations

import math
from collections.abc import Sequence

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


# --- Ordine di lettura -----------------------------------------------------

# Una regione più larga di questa frazione del contenuto attraversa le colonne
# (titolo di capitolo, banda, tabella a tutta pagina): apre una nuova sezione.
SPAN_WIDTH_RATIO = 0.6
# Spazio orizzontale (frazione della larghezza del contenuto) che separa due
# colonne. Basso di proposito: unire due colonne ne interlaccia il testo, mentre
# spezzarne una in due è quasi impossibile (serve spazio bianco verticale vero).
COLUMN_GAP_RATIO = 0.02
# Fascia in alto e in basso alla pagina occupata dagli elementi di servizio
# (numero di pagina, titolo corrente): vanno tenuti fuori dal flusso a colonne.
MARGIN_RATIO = 0.05


def _columns(idxs: list[int], boxes: list[BBox], width: float) -> list[list[int]]:
    """Raggruppa gli indici in colonne: si apre una colonna nuova solo quando
    c'è spazio bianco verticale fra il bordo destro visto finora e la regione
    successiva (indici ordinati per x0)."""
    if not idxs:
        return []
    order = sorted(idxs, key=lambda i: boxes[i][0])
    cols: list[list[int]] = [[order[0]]]
    right = boxes[order[0]][2]
    for i in order[1:]:
        if boxes[i][0] > right + COLUMN_GAP_RATIO * width:
            cols.append([i])
            right = boxes[i][2]
        else:
            cols[-1].append(i)
            right = max(right, boxes[i][2])
    return cols


def reading_order(
    boxes: Sequence[BBox | None],
    page_height: float | None = None,
    in_flow: Sequence[bool] | None = None,
) -> list[int]:
    """Indici delle regioni in ordine di lettura.

    Serve perché i motori OCR emettono i blocchi nell'ordine che preferiscono:
    su un impaginato a due colonne un titolo di sezione può uscire PRIMA del
    paragrafo che in pagina lo precede. Chi legge il testo a valle eredita
    l'errore in silenzio — la Fase 4 assegna ogni elemento al titolo che lo
    precede, quindi un titolo fuori posto riattribuisce il tratto alla sezione
    sbagliata e l'errore esce validato e ad alta confidenza.

    Regola CONSERVATIVA: si corregge l'ordine verticale DENTRO ogni colonna,
    non l'ordine delle colonne fra loro, che resta quello del motore (una
    colonna vale quanto la prima delle sue regioni nell'output originale).
    Ordinare anche le colonne per posizione sembra più pulito ma regredisce:
    una riga di piè di pagina o un numero di pagina nel margine forma una
    colonna propria e, ordinata per x, scavalca il corpo del testo finendo in
    testa alla pagina. Il motore la posizione delle colonne la azzecca quasi
    sempre; è dentro la colonna che sbaglia.

    `in_flow` marca le regioni che partecipano al flusso: quelle senza testo
    (figure, fregi) non ne fanno parte e vanno TOLTE dal calcolo delle colonne,
    non solo ignorate alla fine. Una regione vuota larga che comincia dove
    finisce la colonna sinistra e arriva in fondo alla destra salda le due
    colonne in una sola, e l'intera pagina finisce ordinata per y: sulla pagina
    41 del Player's Handbook le due colonne uscivano interlacciate per colpa di
    una regione vuota larga il 58,8% della pagina, appena sotto la soglia di
    spanner. Nel PHB ce ne sono 402 su 9.535.

    `page_height` serve solo a riconoscere gli elementi di servizio nei margini
    (numero di pagina, titolo corrente): restano dove li ha messi il motore,
    in testa o in coda. Senza, un piè di pagina finisce in fondo alla colonna
    in cui cade — cioè in mezzo alla pagina — e si presenta a valle come un
    titolo di sezione fasullo.

    Se anche una sola bbox manca si restituisce l'ordine del motore: senza
    geometria non c'è niente di meglio su cui basarsi.
    """
    n = len(boxes)
    if n < 2 or any(b is None for b in boxes):
        return list(range(n))
    if in_flow is not None and not all(in_flow):
        keep = [i for i in range(n) if in_flow[i]]
        rest = [i for i in range(n) if not in_flow[i]]
        if len(keep) < 2:
            return list(range(n))
        sub = reading_order([boxes[i] for i in keep], page_height)
        # le regioni fuori flusso non portano testo: in coda, nell'ordine del motore
        return [keep[k] for k in sub] + rest
    bs: list[BBox] = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                      for b in boxes]  # type: ignore[index]
    left = min(b[0] for b in bs)
    right = max(b[2] for b in bs)
    width = right - left
    if width <= 0:
        return list(range(n))

    head: list[int] = []
    foot: list[int] = []
    body = list(range(n))
    if page_height:
        top, bottom = MARGIN_RATIO * page_height, (1 - MARGIN_RATIO) * page_height
        head = [i for i in body if bs[i][3] <= top]
        foot = [i for i in body if bs[i][1] >= bottom]
        margin = set(head) | set(foot)
        body = [i for i in body if i not in margin]
        if not body:
            return head + foot

    spanners = sorted((i for i in body if bs[i][2] - bs[i][0] >= SPAN_WIDTH_RATIO * width),
                      key=lambda i: (bs[i][1], bs[i][0]))
    if len(spanners) == len(body):  # nessuna colonna: solo blocchi a tutta larghezza
        return head + spanners + foot

    # Ogni regione appartiene alla sezione aperta dall'ultimo blocco a tutta
    # larghezza che le sta sopra (confronto sul centro verticale).
    cuts = [bs[i][3] for i in spanners]  # bordo inferiore degli spanner
    sections: dict[int, list[int]] = {}
    span_at = set(spanners)
    for i in body:
        if i in span_at:
            continue
        y_mid = (bs[i][1] + bs[i][3]) / 2.0
        sections.setdefault(sum(1 for c in cuts if c <= y_mid), []).append(i)

    out: list[int] = list(head)
    for k in range(len(spanners) + 1):
        if k > 0:
            out.append(spanners[k - 1])
        cols = _columns(sections.get(k, []), bs, width)
        # colonne nell'ordine del motore (la prima regione che ne fa parte),
        # regioni dentro la colonna per posizione verticale
        for col in sorted(cols, key=min):
            out.extend(sorted(col, key=lambda i: (bs[i][1], bs[i][0])))
    return out + foot


__all__ = [
    "BBox",
    "rotate_bbox",
    "invert_deskew_bbox",
    "iou",
    "union_bbox",
    "expand",
    "reading_order",
]
