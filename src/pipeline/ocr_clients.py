"""Client HTTP per i motori OCR e per l'estrattore LLM.

I client pesanti (PaddleOCR-VL via vLLM server, DeepSeek-OCR2) sono dietro import
lazy: i test possono sostituirli con stub. L'interfaccia pubblica è stabile.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import get_settings
from .geometry import BBox

log = logging.getLogger(__name__)


@dataclass
class Region:
    page_no: int
    bbox: BBox | None
    region_type: str  # text|table|formula|stamp
    text: str
    order_idx: int


# ---------------------------------------------------------------------------
# OCR A: PaddleOCR-VL-1.6 via vLLM server
# ---------------------------------------------------------------------------


class PaddleOCRVLClient:
    """Wrapper su PaddleOCRVL con backend vllm-server."""

    def __init__(self, url: str | None = None):
        self.url = url or get_settings().ocr_a_url

    def _build(self):
        from paddleocr import PaddleOCRVL  # import lazy

        return PaddleOCRVL(
            pipeline_version="v1.6",
            vl_rec_backend="vllm-server",
            vl_rec_server_url=self.url,
        )

    def ocr_page(self, image_path: str | Path, page_no: int) -> list[Region]:
        pipeline = self._build()
        result = pipeline.predict(str(image_path))
        return _parse_paddle_result(result, page_no)


def _parse_paddle_result(result: Any, page_no: int) -> list[Region]:
    """Parser difensivo: PaddleOCR-VL restituisce regioni tipizzate con bbox, testo,
    ordine di lettura. Tabelle in forma strutturata (serializzate JSON in text)."""
    regions: list[Region] = []
    # Il formato esatto dipende dalla versione; gestiamo il caso comune (lista di dict).
    items = result if isinstance(result, list) else getattr(result, "json", None) or []
    for i, item in enumerate(items):
        bbox = item.get("bbox")
        rtype = item.get("type", "text")
        text = item.get("text", "")
        if rtype == "table":
            # Conserva la struttura, non appiattire
            text = json.dumps(item.get("table", text), ensure_ascii=False)
        regions.append(Region(page_no=page_no, bbox=tuple(bbox) if bbox else None,
                              region_type=rtype, text=text, order_idx=i))
    return regions


# ---------------------------------------------------------------------------
# OCR B: DeepSeek-OCR 2 (o dots.ocr) via vLLM server
# ---------------------------------------------------------------------------


class DeepSeekOCRClient:
    def __init__(self, url: str | None = None, model: str | None = None):
        s = get_settings()
        self.url = url or s.ocr_b_url
        self.model = model or s.model_b

    def ocr_page(self, image_path: str | Path, page_no: int) -> list[Region]:
        # DeepSeek-OCR2 expone un'API simile a OpenAI vision via vLLM.
        # Qui usiamo il client OpenAI generico; l'implementazione concreta può variare.
        from openai import OpenAI  # import lazy

        client = OpenAI(base_url=self.url, api_key="EMPTY")
        with open(image_path, "rb") as f:
            import base64

            b64 = base64.b64encode(f.read()).decode()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Extract text regions with bbox and type as JSON."},
                ],
            }],
            temperature=0.0,
        )
        items = json.loads(resp.choices[0].message.content or "[]")
        return [
            Region(page_no=page_no, bbox=tuple(it.get("bbox")) if it.get("bbox") else None,
                   region_type=it.get("type", "text"), text=it.get("text", ""), order_idx=i)
            for i, it in enumerate(items)
        ]


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
# Estrattore (Qwen3.8-27B AWQ) con guided_json via xgrammar
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
        """Chiama l'estrattore con guided_json (SchemaLoose del task)."""
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
            "max_tokens": 2048,
        }
        if guided_json_schema is not None:
            # vLLM supporta extra_body con guided_json / guided_decoding_backend
            kwargs["extra_body"] = {
                "guided_json": guided_json_schema,
                "guided_decoding_backend": "xgrammar",
            }

        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("Output estrattore non JSON: %s", text[:200])
            return {}


__all__ = [
    "Region",
    "PaddleOCRVLClient",
    "DeepSeekOCRClient",
    "ResolverClient",
    "ExtractorClient",
]
