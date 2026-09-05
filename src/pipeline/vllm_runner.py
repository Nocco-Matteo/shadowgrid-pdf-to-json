"""Gestione vita dei server vLLM tra le fasi.

VRAM: PaddleOCR-VL ~3GB; Qwen3.8-27B AWQ 4-bit ~18GB + KV cache.
32GB RAM sistema -> niente offload -> fasi in serie, vLLM si spegne/riparte.
Stato su SQLite -> non perdi nulla.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from .config import Settings, get_settings

log = logging.getLogger(__name__)


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
    ) -> str:
        """Avvia `vllm serve <model>` e aspetta health-check su /v1/models."""
        if self.proc is not None and self.proc.poll() is None:
            log.warning("vLLM già in esecuzione, skip start")
            return f"http://127.0.0.1:{port}/v1"

        args = [
            "vllm", "serve", model,
            "--port", str(port),
            "--trust-remote-code",
            "--max-num-batched-tokens", "16384",
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
            preexec_fn=os.setsid,
        )
        url = f"http://127.0.0.1:{port}/v1"
        self._wait_ready(url, timeout=900)
        return url

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
