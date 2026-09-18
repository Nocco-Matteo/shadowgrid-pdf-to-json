"""Test dell'adapter _parse_paddle_result (formato reale PaddleOCR-VL 1.6).

Il formato documentato: predict() -> iteratore di risultati per pagina; ogni
risultato espone `.json` con {input_path, page_index, parsing_res_list:
[{block_label, block_bbox, block_content, ...}]}.
"""
from __future__ import annotations

import pytest

from pipeline.ocr_clients import OCRParseError, _parse_paddle_result


class _VOResult:
    """Mimica l'oggetto risultato di PaddleOCR (ha l'attributo .json)."""

    def __init__(self, payload: dict):
        self.json = payload


BLOCKS = [
    {
        "block_label": "title",
        "block_bbox": [52, 69, 270, 105],
        "block_content": "CONTRATTO N. 44/B",
        "index": 1,
    },
    {
        "block_label": "text",
        "block_bbox": [52, 120, 270, 200],
        "block_content": "Data emissione: 2024-03-15",
        "index": 2,
    },
    {
        "block_label": "table",
        "block_bbox": [52, 220, 500, 400],
        "block_content": "<table><tr><td>Parte</td><td>Ruolo</td></tr></table>",
        "index": 3,
    },
    {
        "block_label": "formula",
        "block_bbox": [52, 410, 200, 440],
        "block_content": "E = mc^2",
        "index": 4,
    },
    {
        "block_label": "stamp",
        "block_bbox": [400, 50, 480, 110],
        "block_content": "TIMBRO",
        "index": 5,
    },
]


def _page(blocks):
    return {"input_path": "page.png", "page_index": 0, "parsing_res_list": blocks}


def test_parse_result_object_with_json():
    """Il caso reale: oggetto con .json e parsing_res_list."""
    regions = _parse_paddle_result(_VOResult(_page(BLOCKS)), page_no=3)
    assert len(regions) == 5
    r0 = regions[0]
    assert r0.page_no == 3
    assert r0.bbox == (52, 69, 270, 105)
    assert r0.region_type == "text"  # title -> text
    assert r0.text == "CONTRATTO N. 44/B"
    assert r0.order_idx == 0
    table = regions[2]
    assert table.region_type == "table"
    assert "<table>" in table.text  # HTML conservato
    assert regions[3].region_type == "formula"
    assert regions[4].region_type == "stamp"


def test_parse_result_object_with_json_res_wrapper():
    """PaddleX JsonMixin incapsula il risultato: .json -> {"res": {...}}."""
    wrapped = _VOResult({"res": _page(BLOCKS)})
    regions = _parse_paddle_result(wrapped, page_no=2)
    assert len(regions) == 5
    assert regions[0].text == "CONTRATTO N. 44/B"
    assert regions[0].bbox == (52, 69, 270, 105)


def test_parse_bare_dict_res_wrapper():
    regions = _parse_paddle_result({"res": _page(BLOCKS)}, page_no=1)
    assert len(regions) == 5


def test_parse_res_wrapper_empty_page():
    regions = _parse_paddle_result(_VOResult({"res": _page([])}), page_no=1)
    assert regions == []


def test_parse_generator_of_results():
    """predict() restituisce un generatore: non deve più produrre 0 regioni."""
    def gen():
        yield _VOResult(_page(BLOCKS))

    regions = _parse_paddle_result(gen(), page_no=1)
    assert len(regions) == 5
    assert regions[0].text == "CONTRATTO N. 44/B"


def test_parse_bare_dict():
    regions = _parse_paddle_result(_page(BLOCKS), page_no=1)
    assert len(regions) == 5


def test_parse_list_of_results():
    regions = _parse_paddle_result([_VOResult(_page(BLOCKS))], page_no=1)
    assert len(regions) == 5


def test_parse_json_as_string():
    import json

    regions = _parse_paddle_result(
        _VOResult(json.dumps(_page(BLOCKS[:1]))), page_no=1
    )
    assert len(regions) == 1


def test_empty_page_is_not_an_error():
    """Pagina realmente vuota: parsing_res_list == [] -> zero regioni."""
    regions = _parse_paddle_result(_VOResult(_page([])), page_no=1)
    assert regions == []


def test_unrecognized_result_raises():
    with pytest.raises(OCRParseError):
        _parse_paddle_result("qualcosa", page_no=1)
    with pytest.raises(OCRParseError):
        _parse_paddle_result(None, page_no=1)
    with pytest.raises(OCRParseError):
        _parse_paddle_result({"altro": "formato"}, page_no=1)
    with pytest.raises(OCRParseError):
        _parse_paddle_result(_VOResult({"niente": True}), page_no=1)


def test_legacy_list_format_still_works():
    legacy = [
        {"bbox": [1, 2, 3, 4], "type": "text", "text": "ciao"},
        {"bbox": None, "type": "table", "text": None, "table": {"rows": 2}},
    ]
    regions = _parse_paddle_result(legacy, page_no=7)
    assert len(regions) == 2
    assert regions[0].text == "ciao"
    assert regions[0].bbox == (1, 2, 3, 4)
    assert regions[1].region_type == "table"
    assert "rows" in regions[1].text  # struttura conservata in JSON


def test_bad_block_bbox_raises():
    blocks = [{"block_label": "text", "block_bbox": [1, 2, 3], "block_content": "x"}]
    with pytest.raises(OCRParseError):
        _parse_paddle_result(_VOResult(_page(blocks)), page_no=1)


# ---------------------------------------------------------------------------
# DeepSeek-OCR 2: output di grounding <|ref|>/<|det|> (non JSON)
# ---------------------------------------------------------------------------

from pipeline.ocr_clients import _parse_deepseek_grounding  # noqa: E402

DEEPSEEK_OUT = (
    "<|ref|>title<|/ref|><|det|>[[100, 50, 899, 90]]<|/det|>\n"
    "# CONTRATTO DI LOCAZIONE\n\n"
    "<|ref|>text<|/ref|><|det|>[[100, 120, 899, 300]]<|/det|>\n"
    "Il locatore concede in locazione: {\"canone\": 800}\n\n"
    "<|ref|>image<|/ref|><|det|>[[100, 400, 500, 600]]<|/det|>\n\n"
    "<|ref|>table<|/ref|><|det|>[[100, 650, 899, 800]]<|/det|>\n"
    "<table><tr><td>Canone</td><td>800</td></tr></table>"
    "<｜end▁of▁sentence｜>"
)


def test_deepseek_grounding_parse():
    regions = _parse_deepseek_grounding(DEEPSEEK_OUT, page_no=1, img_w=999, img_h=1998)
    assert [r.region_type for r in regions] == ["text", "text", "table"]  # image scartata
    assert regions[0].text == "# CONTRATTO DI LOCAZIONE"
    assert regions[0].bbox == (100, 100, 899, 180)  # scala 0-999 -> pixel
    assert regions[1].text == 'Il locatore concede in locazione: {"canone": 800}'
    assert regions[2].text.startswith("<table>") and "end" not in regions[2].text
    assert [r.order_idx for r in regions] == [0, 1, 2]


def test_deepseek_grounding_empty_page():
    assert _parse_deepseek_grounding("", page_no=1, img_w=100, img_h=100) == []


def test_deepseek_grounding_unrecognized_raises():
    # Testo senza tag (es. skip_special_tokens rimasto attivo): non è una pagina vuota
    with pytest.raises(OCRParseError):
        _parse_deepseek_grounding("CONTRATTO DI LOCAZIONE\nIl locatore...", 1, 100, 100)
