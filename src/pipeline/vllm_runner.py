"""Gestione vita dei server vLLM tra le fasi.

VRAM: PaddleOCR-VL ~3GB; Qwen3.8-27B AWQ 4-bit ~18GB + KV cache.
32GB RAM sistema -> niente offload -> fasi in serie, vLLM si spegne/riparte.
Stato su SQLite -> non perdi nulla.
"""

from __future__ import annotations

import logging
import os
import platform
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from .config import Settings, get_settings

log = logging.getLogger(__name__)


def _in_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower()


def _vllm_env() -> dict[str, str]:
    """Ambiente del processo `vllm serve`.

    In WSL2 il V2 model runner fallisce con "UVA is not available" (niente
    pinned memory) e il sampler FlashInfer con "Could not find nvcc" (c'è il
    driver ma non il CUDA Toolkit). Si usano i default sicuri, ma un valore
    già impostato dall'utente ha la precedenza."""
    env = os.environ.copy()
    if _in_wsl():
        env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return env


class VLLMRunner:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.proc: subprocess.Popen | None = None
        self.log_dir = Path("runs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        model: str,
        port: int = 8080,
        extra_args: list[str] | None = None,
        phase: str = "phase",
        max_num_batched_tokens: int = 16384,
    ) -> str:
        """Avvia `vllm serve <model>` e aspetta health-check su /v1/models."""
        if self.proc is not None and self.proc.poll() is None:
            log.warning("vLLM già in esecuzione, skip start")
            return f"http://127.0.0.1:{port}/v1"

        args = [
            "vllm", "serve", model,
            "--port", str(port),
            "--trust-remote-code",
            "--max-num-batched-tokens", str(max_num_batched_tokens),
            "--no-enable-prefix-caching",
            "--mm-processor-cache-gb", "0",
        ]
        if extra_args:
            args.extend(extra_args)

        log_file = self.log_dir / f"vllm_{phase}.log"
        log.info("Avvio vLLM: %s (log: %s)", " ".join(args), log_file)
        lf = log_file.open("a", buffering=1)
        self.proc = subprocess.Popen(
            args, stdout=lf, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid, env=_vllm_env(),
        )
        url = f"http://127.0.0.1:{port}/v1"
        self._wait_ready(url, timeout=900)
        return url

    def start_extractor(self, port: int = 8080, phase: str = "extract") -> str:
        """Avvia l'estrattore con i limiti di memoria da config."""
        s = self.s
        return self.start(
            s.extractor_model,
            port=port,
            phase=phase,
            max_num_batched_tokens=s.extractor_max_num_batched_tokens,
            extra_args=[
                "--max-model-len", str(s.extractor_max_model_len),
                "--gpu-memory-utilization", str(s.extractor_gpu_memory_utilization),
                *s.extractor_vllm_args,
            ],
        )

    def _wait_ready(self, url: str, timeout: int = 900) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"vLLM terminato con codice {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{url}/models", timeout=5) as r:
                    if r.status == 200:
                        log.info("vLLM ready: %s", url)
                        return
            except Exception:
                time.sleep(5)
        raise TimeoutError(f"vLLM non pronto entro {timeout}s su {url}")

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            log.info("Stop vLLM (pid=%s)", self.proc.pid)
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        self.proc = None


__all__ = ["VLLMRunner"]
