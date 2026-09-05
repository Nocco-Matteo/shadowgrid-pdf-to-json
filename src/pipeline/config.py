"""Configurazione via variabili d'ambiente (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    db_path: Path = Field(default=Path("runs/pipeline.db"))
    work_dir: Path = Field(default=Path("runs/work"))

    # Modelli / server
    vllm_url: str = "http://127.0.0.1:8080/v1"
    ocr_a_url: str = "http://127.0.0.1:8080/v1"
    ocr_b_url: str = "http://127.0.0.1:8080/v1"
    extractor_url: str = "http://127.0.0.1:8080/v1"

    model_a: str = "PaddlePaddle/PaddleOCR-VL-1.6"
    model_b: str = "deepseek-ai/deepseek-ocr2"  # o dots.ocr
    extractor_model: str = "Qwen/Qwen3.8-27B-AWQ"

    # Rasterizzazione
    dpi: int = 300
    dpi_degraded: int = 400
    deskew_min_angle: float = 0.3

    # OCR / riconciliazione
    iou_align_threshold: float = 0.5
    text_conflict_threshold: float = 0.95
    conflict_margin_px: int = 10
    conflict_upscale: int = 2
    conflict_rate_warn: float = 0.05

    # Estrazione
    extractor_temperature: float = 0.0
    extractor_seed: int = 12345
    max_fields_per_task: int = 8
    max_field_per_task_min: int = 5

    # Validazione
    grounding_threshold: int = 90
    max_retries: int = 2  # 2 retry -> al 3° tentativo needs_review

    # Run
    git_sha: str = "unknown"


def get_settings() -> Settings:
    s = Settings()
    s.work_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    return s
