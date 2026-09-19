"""Client HTTP per i motori OCR e per l'estrattore LLM.

I client pesanti (PaddleOCR-VL via vLLM server, DeepSeek-OCR2) sono dietro import
lazy: i test possono sostituirli con stub. L'interfaccia pubblica è stabile.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import get_settings
from .geometry import BBox, reading_order

log = logging.getLogger(__name__)


class OCRError(RuntimeError):
    """Errore generico dei motori OCR."""


class OCRParseError(OCRError):
    """Il risultato del motore OCR non è in un formato riconosciuto.

    Distinto da "pagina vuota" (formato riconosciuto, zero regioni): una
    risposta non riconosciuta non deve produrre silenziosamente una pagina
    vuota e far avanzare la pipeline."""


class ExtractorError(RuntimeError):
    """L'estrattore ha risposto in modo inutilizzabile (non JSON / non oggetto).

    Non va confusa con una risposta valida che dichiara assenza (value=null)."""


@dataclass
class Region:
    page_no: int
    bbox: BBox | None
    region_type: str  # text|table|formula|stamp
    text: str
    order_idx: int
    region_id: int | None = None  # id DB, se la regione viene dal DB


def _in_reading_order(regions: list[Region], page_height: float | None) -> list[Region]:
    """Riordina le regioni di una pagina in ordine di lettura e rinumera
    order_idx. I motori emettono i blocchi nel loro ordine interno, che su un
    impaginato a colonne non è quello di lettura: v. geometry.reading_order."""
    order = reading_order([r.bbox for r in regions], page_height,
                          in_flow=[bool((r.text or "").strip()) for r in regions])
    out = [regions[i] for i in order]
    for idx, r in enumerate(out):
        r.order_idx = idx
    return out


# ---------------------------------------------------------------------------
# OCR A: PaddleOCR-VL-1.6 via vLLM server
# ---------------------------------------------------------------------------


class PaddleOCRVLClient:
    """Wrapper su PaddleOCRVL con backend vllm-server. La pipeline è costruita
    una sola volta e cachata (l'init è costoso)."""

    def __init__(self, url: str | None = None, model: str | None = None):
        s = get_settings()
        self.url = url or s.ocr_a_url
        # vLLM espone il modello col nome passato a `vllm serve` (model_a); il
        # client Paddle di default chiede "PaddleOCR-VL-1.6-0.9B" -> 404.
        self.model = model or s.model_a
        self._pipeline = None

    def _build(self):
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL  # import lazy

            self._pipeline = PaddleOCRVL(
                pipeline_version="v1.6",
                vl_rec_backend="vllm-server",
                vl_rec_server_url=self.url,
                vl_rec_api_model_name=self.model,
            )
        return self._pipeline

    def ocr_page(self, image_path: str | Path, page_no: int) -> list[Region]:
        result = self._build().predict(str(image_path))
        regions = _parse_paddle_result(result, page_no, _image_height(image_path))
        if not regions:
            log.info("OCR A pagina %d: nessuna regione (pagina vuota)", page_no)
        return regions


def _parse_paddle_result(
    result: Any, page_no: int, page_height: float | None = None
) -> list[Region]:
    """Adapter per il formato documentato di PaddleOCR-VL (pipeline_version 1.6).

    `predict()` restituisce un iteratore di risultati, uno per pagina; ogni
    risultato espone `.json` con l'involucro PaddleX JsonMixin
    `{"res": {"parsing_res_list": [...]}}` (gestito anche lo spacchettato
    senza involucro): blocchi con `block_bbox` [x0,y0,x1,y1], `block_label`
    (text/table/formula/stamp/...) e `block_content` (testo; HTML per le
    tabelle).

    Formati gestiti:
    - oggetto risultato con attributo `.json` (il caso reale);
    - dict già spacchettato con `parsing_res_list`;
    - lista/iteratore dei precedenti (più pagine);
    - legacy: lista di dict {bbox, type, text}.

    Una pagina vuota (`parsing_res_list == []`) restituisce zero regioni; un
    risultato non riconosciuto solleva OCRParseError.
    """
    regions: list[Region] = []
    for payload in _paddle_page_payloads(result):
        blocks = payload.get("parsing_res_list")
        if blocks is None:
            raise OCRParseError(
                f"payload OCR senza 'parsing_res_list': {str(payload)[:200]}"
            )
        if not isinstance(blocks, list):
            raise OCRParseError(f"parsing_res_list non è una lista: {type(blocks)!r}")
        for i, block in enumerate(blocks):
            regions.append(_paddle_block_to_region(block, page_no, len(regions) + i))
    return _in_reading_order(regions, page_height)


def _paddle_page_payloads(result: Any) -> list[dict]:
    """Normalizza il risultato di predict() in una lista di payload per pagina.
    Solleva OCRParseError per formati non riconosciuti; il formato legacy
    (lista piatta di regioni) viene accettato come payload a un blocco."""
    if result is None or isinstance(result, (str, bytes, bool, int, float)):
        raise OCRParseError(f"formato risultato OCR non riconosciuto: {type(result)!r}")

    def _payload(obj: Any) -> dict | None:
        # oggetto risultato con .json, oppure dict con parsing_res_list
        if hasattr(obj, "json"):
            j = obj.json
            if callable(j):  # alcune versioni espongono .json() come metodo
                j = j()
            if isinstance(j, str):
                try:
                    j = json.loads(j)
                except json.JSONDecodeError as e:
                    raise OCRParseError(f".json del risultato OCR non decodificabile: {e}") from e
            if not isinstance(j, dict):
                raise OCRParseError(f"risultato OCR .json inatteso: {type(j)!r}")
            obj = j
        if isinstance(obj, dict):
            # Formato già spacchettato
            if "parsing_res_list" in obj:
                return obj
            # PaddleX JsonMixin incapsula il risultato: {"res": {...}}
            inner = obj.get("res")
            if isinstance(inner, dict) and "parsing_res_list" in inner:
                return inner
        return None

    single = _payload(result)
    if single is not None:
        return [single]

    if isinstance(result, (list, tuple)) or hasattr(result, "__iter__"):
        items = list(result)
        payloads = [_payload(it) for it in items]
        if all(p is not None for p in payloads):
            return [p for p in payloads if p is not None]
        # legacy: lista piatta di regioni {bbox, type, text}
        if items and all(isinstance(it, dict) and "parsing_res_list" not in it for it in items):
            return [{"parsing_res_list": items}]
        if not items:
            return [{"parsing_res_list": []}]
        raise OCRParseError("elementi del risultato OCR non riconosciuti")

    raise OCRParseError(f"formato risultato OCR non riconosciuto: {type(result)!r}")


# Etichette PaddleOCR-VL -> region_type della pipeline
_LABEL_MAP = {
    "text": "text",
    "title": "text",
    "header": "text",
    "footer": "text",
    "line": "text",
    "paragraph": "text",
    "reference": "text",
    "algorithm": "text",
    "table": "table",
    "table_title": "table",
    "formula": "formula",
    "stamp": "stamp",
    "seal": "stamp",
}


def _image_height(image_path: str | Path) -> float | None:
    """Altezza in pixel dell'immagine di pagina; None se illeggibile (senza,
    reading_order non distingue gli elementi di servizio nei margini)."""
    try:
        from PIL import Image  # import lazy

        with Image.open(image_path) as im:
            return float(im.size[1])
    except Exception as e:
        log.debug("Altezza pagina non determinata per %s: %s", image_path, e)
        return None


def _map_label(label: Any) -> str:
    return _LABEL_MAP.get(str(label or "").lower(), "text")


def _paddle_block_to_region(block: Any, page_no: int, idx: int) -> Region:
    if not isinstance(block, dict):
        raise OCRParseError(f"blocco OCR non è un dict: {type(block)!r}")
    # Formato documentato PaddleOCR-VL
    if "block_bbox" in block or "block_label" in block or "block_content" in block:
        bbox = block.get("block_bbox")
        if bbox is not None and not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise OCRParseError(f"block_bbox inatteso: {bbox!r}")
        rtype = _map_label(block.get("block_label"))
        text = block.get("block_content")
        text = "" if text is None else str(text)
        if rtype == "table" and "table" in block and block.get("table") is not None:
            # struttura alternativa già dict: conserva senza appiattire
            text = json.dumps(block["table"], ensure_ascii=False)
        return Region(page_no=page_no,
                      bbox=tuple(bbox) if bbox else None,
                      region_type=rtype, text=text, order_idx=idx)
    # Formato legacy: {bbox, type, text}
    if "bbox" in block or "type" in block or "text" in block:
        bbox = block.get("bbox")
        rtype = str(block.get("type", "text") or "text")
        text = block.get("text", "") or ""
        if rtype == "table":
            # Conserva la struttura, non appiattire
            text = json.dumps(block.get("table", text), ensure_ascii=False)
        return Region(page_no=page_no, bbox=tuple(bbox) if bbox else None,
                      region_type=rtype, text=text, order_idx=idx)
    raise OCRParseError(f"blocco OCR senza campi riconosciuti: {str(block)[:200]}")


# ---------------------------------------------------------------------------
# OCR B: DeepSeek-OCR 2 (o dots.ocr) via vLLM server
# ---------------------------------------------------------------------------


# Prompt di grounding ufficiale: l'output è una sequenza di blocchi
# `<|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>` seguiti dal testo
# del blocco, con coordinate normalizzate su 0-999. Non è JSON.
_DEEPSEEK_PROMPT = "<|grounding|>Convert the document to markdown."
_DEEPSEEK_BLOCK_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>", re.S)
_DEEPSEEK_EOS_RE = re.compile(r"<｜end▁of▁sentence｜>|<\|end▁of▁sentence\|>")

# Etichette DeepSeek-OCR non presenti in _LABEL_MAP
_DEEPSEEK_LABEL_MAP = {
    "sub_title": "text",
    "table_caption": "text",
    "table_footnote": "text",
    "image_caption": "text",
    "equation": "formula",
}


def _parse_deepseek_grounding(content: str, page_no: int, img_w: int, img_h: int) -> list[Region]:
    """Converte l'output di grounding di DeepSeek-OCR 2 in Region (bbox in pixel)."""
    content = _DEEPSEEK_EOS_RE.sub("", content or "")
    matches = list(_DEEPSEEK_BLOCK_RE.finditer(content))
    if not matches:
        if content.strip():
            raise OCRParseError(f"output DeepSeek-OCR senza blocchi <|ref|>/<|det|>: {content[:200]!r}")
        return []
    regions: list[Region] = []
    for i, m in enumerate(matches):
        label = m.group(1).strip().lower()
        if label == "image":
            continue  # figura: nessun testo da riconciliare
        try:
            boxes = json.loads(m.group(2))
        except json.JSONDecodeError as e:
            raise OCRParseError(f"coordinate DeepSeek-OCR non valide: {m.group(2)!r}") from e
        if not (isinstance(boxes, list) and boxes and isinstance(boxes[0], list) and len(boxes[0]) == 4):
            raise OCRParseError(f"coordinate DeepSeek-OCR inattese: {m.group(2)!r}")
        # Più box per lo stesso blocco: si usa quella che li contiene tutti
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        bbox = (x1 / 999 * img_w, y1 / 999 * img_h, x2 / 999 * img_w, y2 / 999 * img_h)
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[m.end():text_end].strip()
        rtype = _DEEPSEEK_LABEL_MAP.get(label) or _map_label(label)
        regions.append(Region(page_no=page_no, bbox=bbox, region_type=rtype,
                              text=text, order_idx=len(regions)))
    return _in_reading_order(regions, float(img_h))


class DeepSeekOCRClient:
    def __init__(self, url: str | None = None, model: str | None = None):
        s = get_settings()
        self.url = url or s.ocr_b_url
        self.model = model or s.model_b

    def ocr_page(self, image_path: str | Path, page_no: int) -> list[Region]:
        import base64

        from openai import OpenAI  # import lazy
        from PIL import Image

        client = OpenAI(base_url=self.url, api_key="EMPTY")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        with Image.open(image_path) as im:
            img_w, img_h = im.size
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": _DEEPSEEK_PROMPT},
                ],
            }],
            temperature=0.0,
            # Senza questo vLLM toglie i token <|ref|>/<|det|> dall'output
            extra_body={"skip_special_tokens": False},
        )
        return _parse_deepseek_grounding(resp.choices[0].message.content or "", page_no, img_w, img_h)


# ---------------------------------------------------------------------------
# Resolver mirato (terzo passaggio per conflitti)
# ---------------------------------------------------------------------------


class ResolverClient:
    """Rilegge un crop a piena risoluzione per risolvere conflitti 2-su-3."""

    def __init__(self, url: str | None = None, model: str | None = None):
        s = get_settings()
        self.url = url or s.ocr_b_url
        self.model = model or s.model_b

    def read_crop(self, image_path: str | Path) -> str:
        import base64

        from openai import OpenAI

        client = OpenAI(base_url=self.url, api_key="EMPTY")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Transcribe verbatim the text in the image. Output only the text."},
                ],
            }],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Estrattore (Qwen3.8-27B AWQ) con output vincolato a JSON schema
# ---------------------------------------------------------------------------


class ExtractorClient:
    def __init__(self, url: str | None = None, model: str | None = None):
        s = get_settings()
        self.url = url or s.extractor_url
        self.model = model or s.extractor_model
        self.seed = s.extractor_seed
        self.temperature = s.extractor_temperature

    def extract(
        self,
        prompt: str,
        images_b64: list[str] | None = None,
        guided_json_schema: dict | None = None,
    ) -> dict:
        """Chiama l'estrattore vincolando l'output allo SchemaLoose del task."""
        from openai import OpenAI

        client = OpenAI(base_url=self.url, api_key="EMPTY")
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64 in images_b64 or []:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": get_settings().extractor_max_tokens,
            # Qwen3.x ragiona (<think>) prima di rispondere: qui serve solo il JSON
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if guided_json_schema is not None:
            # response_format standard OpenAI: vLLM (structured outputs, xgrammar)
            # e llama-server lo applicano entrambi. Il vecchio extra_body
            # `guided_json` è stato rimosso da vLLM e verrebbe ignorato in silenzio.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": guided_json_schema},
            }

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if getattr(choice, "finish_reason", None) == "length":
            raise ExtractorError(
                f"risposta troncata a max_tokens={kwargs['max_tokens']} "
                f"(alza extractor_max_tokens): {text[-120:]!r}")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            # Non trasformare un fallimento in una risposta vuota: una `{}`
            # sarebbe indistinguibile da "campo assente" a valle (Fase 5/6).
            raise ExtractorError(f"output estrattore non JSON: {text[:200]!r}") from e
        if not isinstance(parsed, dict):
            raise ExtractorError(f"output estrattore non è un oggetto: {type(parsed).__name__}")
        return parsed


__all__ = [
    "Region",
    "OCRError",
    "OCRParseError",
    "ExtractorError",
    "PaddleOCRVLClient",
    "DeepSeekOCRClient",
    "ResolverClient",
    "ExtractorClient",
]
