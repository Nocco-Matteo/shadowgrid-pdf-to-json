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

    # Schema di estrazione (chiave di schema.SCHEMAS) e codici del libro per
    # l'export verso i seed del compendium (sovrascrivibili con --schema/--source)
    schema_name: str = "race_traits"
    source_codes: list[str] = Field(default_factory=list)

    # Server vLLM dell'estrattore (27B su 24GB: i default di vLLM non lasciano
    # memoria per la KV cache). gpu_memory_utilization è una frazione della VRAM
    # totale: in WSL Windows ne tiene ~1.5GB (22.45/23.99 GiB liberi), oltre
    # ~0.935 vLLM non parte.
    extractor_max_model_len: int = 16384
    extractor_max_num_batched_tokens: int = 2048
    extractor_gpu_memory_utilization: float = 0.93
    # Niente CUDA graph: più lento ma libera VRAM per la KV cache
    extractor_enforce_eager: bool = True
    # Il 27B è multimodale ma riceve solo testo: senza encoder visivo vLLM
    # risparmia ~0.8GB di pesi e il profiling delle immagini. Da disattivare
    # se un giorno l'estrattore riceve images_b64.
    extractor_text_only: bool = True
    extractor_max_num_seqs: int = 16
    # Richieste in volo dal CLIENT. Le fasi mandavano una richiesta per volta a
    # un server da 8 posti: a flusso singolo un 27B sta sui 18 token/s perche'
    # per ogni token rilegge tutti i pesi, mentre in batch quel costo si
    # ammortizza. Non ha senso superare extractor_max_num_seqs: il resto
    # verrebbe accodato dal server.
    extractor_concurrency: int = 16
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
    # Caratteri di TABELLE aggiunti al contesto di un elemento (la sua pagina e
    # quella della sua sezione). Il manuale mette la regola nella prosa e il
    # numero in tabella: senza, l'estrattore dichiara un'assenza. Mediana
    # osservata sul PHB: ~1.100 caratteri per pagina, peggior caso ~6.300.
    item_table_context_chars: int = 8000
    # Caratteri di PROSA dell'elemento: dalla sua anchor a dove comincia il
    # successivo. Con gli elementi a intestazione (capacità di classe) il corpo
    # della regola sta nei paragrafi dopo il titolo, non nel titolo.
    item_span_chars: int = 6000
    # Token massimi di risposta dell'estrattore
    extractor_max_tokens: int = 2048
    # Fase 4: token di regioni per chiamata. None = automatico dal contesto
    # (extractor_max_model_len - extractor_max_tokens - 1024 per le istruzioni)
    enumerate_window_tokens: int | None = None

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
