"""VLLMRunner: argomenti di `vllm serve` per l'estrattore (limiti VRAM)."""

import json

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
    assert _flag(args, "--max-num-seqs") == "8"
    assert json.loads(_flag(args, "--limit-mm-per-prompt")) == {"image": 0, "video": 0}
    assert "--enforce-eager" in args


def test_ocr_start_unchanged(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, tmp_path)
    vllm_runner.VLLMRunner(Settings()).start("org/Ocr", port=8080, phase="ocr_a")
    args = calls[0]
    assert _flag(args, "--max-num-batched-tokens") == "16384"
    assert "--max-model-len" not in args


OLD_RUN = (
    "(EngineCore pid=1) ERROR 09-18 20:00:00 [core.py:1] ValueError: errore del run precedente\n"
)
FAILED_RUN = """\
(EngineCore pid=3709) INFO 09-18 21:36:10 [gpu_model_runner.py:3120] Model loading took 18.21 GiB memory and 190.2 seconds
(EngineCore pid=3709) ERROR 09-18 21:38:02 [core.py:1374]   File "/x/vllm/v1/core/kv_cache_utils.py", line 859, in _check_enough_kv_cache_memory
(EngineCore pid=3709) ERROR 09-18 21:38:02 [core.py:1374]     raise ValueError(
(EngineCore pid=3709) ERROR 09-18 21:38:02 [core.py:1374] ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`.
(EngineCore pid=3709) Traceback (most recent call last):
(EngineCore pid=3709) ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`.
(APIServer pid=3653) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
"""


def test_failure_report_shows_root_cause_of_current_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = vllm_runner.VLLMRunner(Settings())
    runner._log_file = tmp_path / "vllm_extract.log"
    runner._log_file.write_text(OLD_RUN + FAILED_RUN)
    runner._log_offset = len(OLD_RUN.encode())

    msg = runner._failure_report("vLLM terminato con codice 1")

    assert msg.startswith("vLLM terminato con codice 1\nErrore vLLM:")
    assert msg.count("No available memory for the cache blocks") == 1  # niente duplicati
    assert "Engine core initialization failed" in msg
    assert "run precedente" not in msg
    assert "Model loading took 18.21 GiB" in msg
    assert "raise ValueError(" not in msg and "File " not in msg
    assert str(runner._log_file) in msg


def test_start_extractor_multimodal(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, tmp_path)
    vllm_runner.VLLMRunner(Settings(extractor_text_only=False)).start_extractor()
    assert "--limit-mm-per-prompt" not in calls[0]


def test_start_extractor_default_limits(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, tmp_path)
    vllm_runner.VLLMRunner(Settings()).start_extractor()
    args = calls[0]
    assert _flag(args, "--gpu-memory-utilization") == "0.93"
    assert _flag(args, "--max-num-batched-tokens") == "2048"
    assert args.count("--enforce-eager") == 1
