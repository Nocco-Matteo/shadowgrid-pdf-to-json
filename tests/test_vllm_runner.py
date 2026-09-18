"""VLLMRunner: argomenti di `vllm serve` per l'estrattore (limiti VRAM)."""

from pipeline import vllm_runner
from pipeline.config import Settings


def _capture(monkeypatch, tmp_path):
    calls = []

    class _Proc:
        def poll(self):
            return None

    def fake_popen(args, **kw):
        calls.append(args)
        return _Proc()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vllm_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(vllm_runner.VLLMRunner, "_wait_ready", lambda self, url, timeout=900: None)
    return calls


def _flag(args, name):
    return args[args.index(name) + 1]


def test_start_extractor_passes_memory_limits(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, tmp_path)
    s = Settings(extractor_model="org/Estrattore", extractor_max_model_len=8192,
                 extractor_max_num_batched_tokens=2048,
                 extractor_gpu_memory_utilization=0.9,
                 extractor_vllm_args=["--enforce-eager"])
    url = vllm_runner.VLLMRunner(s).start_extractor(port=8080, phase="extract")
    args = calls[0]
    assert url == "http://127.0.0.1:8080/v1"
    assert args[:3] == ["vllm", "serve", "org/Estrattore"]
    assert _flag(args, "--max-model-len") == "8192"
    assert _flag(args, "--max-num-batched-tokens") == "2048"
    assert _flag(args, "--gpu-memory-utilization") == "0.9"
    assert args.count("--max-num-batched-tokens") == 1
    assert args[-1] == "--enforce-eager"


def test_ocr_start_unchanged(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, tmp_path)
    vllm_runner.VLLMRunner(Settings()).start("org/Ocr", port=8080, phase="ocr_a")
    args = calls[0]
    assert _flag(args, "--max-num-batched-tokens") == "16384"
    assert "--max-model-len" not in args
