"""Config: pipeline.yaml + precedenza env PIPELINE_*."""

import os

from pipeline.config import Settings


def test_yaml_file_loading(tmp_path, monkeypatch):
    cfg = tmp_path / "custom.yaml"
    cfg.write_text('model_a: "org/CustomOCR"\ndpi: 400\n')
    monkeypatch.setenv("PIPELINE_CONFIG", str(cfg))
    monkeypatch.delenv("PIPELINE_MODEL_A", raising=False)

    s = Settings()
    assert s.model_a == "org/CustomOCR"
    assert s.dpi == 400
    # non toccati -> default
    assert s.model_b == "deepseek-ai/DeepSeek-OCR-2"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    cfg = tmp_path / "custom.yaml"
    cfg.write_text('model_a: "org/FromYaml"\n')
    monkeypatch.setenv("PIPELINE_CONFIG", str(cfg))
    monkeypatch.setenv("PIPELINE_MODEL_A", "org/FromEnv")

    s = Settings()
    assert s.model_a == "org/FromEnv"


def test_missing_yaml_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_CONFIG", str(tmp_path / "non_esiste.yaml"))
    s = Settings()
    assert s.extractor_model == "nicosuter/Qwen3.8-27B-AWQ"
    assert s.db_path.name == "pipeline.db"


def test_default_pipeline_yaml_in_repo():
    # il file pipeline.yaml in repo deve caricarsi senza errori e dare i modelli di default
    repo_yaml = os.path.join(os.path.dirname(__file__), "..", "pipeline.yaml")
    os.environ["PIPELINE_CONFIG"] = os.path.abspath(repo_yaml)
    try:
        s = Settings()
        assert s.model_a.startswith("PaddlePaddle/")
    finally:
        del os.environ["PIPELINE_CONFIG"]
