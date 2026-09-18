"""Deskew: l'angolo stimato non deve dipendere dalla convenzione di
cv2.minAreaRect ([-90, 0) prima di OpenCV 4.5.1, (0, 90] dopo)."""

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from pipeline.phase1_ingest import estimate_deskew_angle, rotate_image  # noqa: E402


def _page():
    img = np.full((1400, 1000), 255, np.uint8)
    for i in range(20):
        cv2.putText(img, "Lorem ipsum dolor sit amet contratto", (60, 100 + i * 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2)
    return img


def test_straight_page_has_no_skew():
    # Con OpenCV >= 4.5.1 la vecchia normalizzazione dava 90° (pagina ruotata)
    assert abs(estimate_deskew_angle(_page())) < 0.5


@pytest.mark.parametrize("skew", [3, -3, 7, -7])
def test_skewed_page_is_corrected(skew):
    angle = estimate_deskew_angle(rotate_image(_page(), skew))
    assert abs(angle + skew) < 1.0
    assert abs(estimate_deskew_angle(rotate_image(rotate_image(_page(), skew), angle))) < 1.0
