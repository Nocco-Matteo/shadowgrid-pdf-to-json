"""Configurazione: pipeline.yaml < .env < env PIPELINE_* < argomenti espliciti.

Il file YAML di default è ``pipeline.yaml`` nella working directory;
``PIPELINE_CONFIG=/path/altro.yaml`` per puntare altrove.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_FILE = "pipeline.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = os.environ.get("PIPELINE_CONFIG", DEFAULT_CONFIG_FILE)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file),
            file_secret_settings,
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
    model_b: str = "deepseek-ai/DeepSeek-OCR-2"
    extractor_model: str = "nicosuter/Qwen3.8-27B-AWQ"

    # Server vLLM dell'estrattore (27B su 24GB: i default di vLLM non lasciano
    # memoria per la KV cache). gpu_memory_utilization è una frazione della VRAM
    # totale: in WSL Windows ne tiene ~1.5GB, oltre ~0.93 vLLM non parte.
    extractor_max_model_len: int = 16384
    extractor_max_num_batched_tokens: int = 4096
    extractor_gpu_memory_utilization: float = 0.92
    # Il 27B è multimodale ma riceve solo testo: senza encoder visivo vLLM
    # risparmia ~0.8GB di pesi e il profiling delle immagini. Da disattivare
    # se un giorno l'estrattore riceve images_b64.
    extractor_text_only: bool = True
    extractor_max_num_seqs: int = 8
    extractor_vllm_args: list[str] = Field(default_factory=list)

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
