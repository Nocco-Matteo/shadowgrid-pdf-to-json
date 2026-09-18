"""Gestione vita dei server vLLM tra le fasi.

VRAM: PaddleOCR-VL ~3GB; Qwen3.8-27B AWQ 4-bit ~18GB + KV cache.
32GB RAM sistema -> niente offload -> fasi in serie, vLLM si spegne/riparte.
Stato su SQLite -> non perdi nulla.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from .config import Settings, get_settings

log = logging.getLogger(__name__)

# Righe del log di vLLM che spiegano un avvio fallito o lo stato della VRAM.
_ERROR_RE = re.compile(r"\b\w*(?:Error|Exception): |CUDA out of memory")
_MEMORY_RE = re.compile(
    r"Model loading took|Available KV cache memory|GPU KV cache size|"
    r"Maximum concurrency|Free memory on device|max seq len|"
    r"peak activation|non-torch|torch peak|CUDA ?graph memory|Memory profiling",
    re.I,
)
# Prefisso tipo "(EngineCore pid=12) ERROR 09-18 21:38:02 [core.py:1374] "
_PREFIX_RE = re.compile(r"^(?:\([^)]*\)\s*)?(?:[A-Z]+ \d\d-\d\d [\d:]+ \[[^\]]*\]\s*)?")


def _log_excerpt(path: Path, offset: int) -> tuple[list[str], list[str]]:
    """Dal log di vLLM (a partire da `offset`, cioè solo il run corrente)
    estrae le righe di errore e quelle sulla memoria, senza duplicati."""
    try:
        with path.open("rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return [], []
    errors: list[str] = []
    memory: list[str] = []
    for raw in text.splitlines():
        line = _PREFIX_RE.sub("", raw).strip()
        if _ERROR_RE.search(line) and not line.startswith(("File ", "Traceback")):
            if line not in errors:
                errors.append(line)
        elif _MEMORY_RE.search(line) and line not in memory:
            memory.append(line)
    return errors, memory


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
        self._log_file: Path = self.log_dir / "vllm_phase.log"
        self._log_offset = 0

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
        self._log_file, self._log_offset = log_file, lf.tell()
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
        extra = [
            "--max-model-len", str(s.extractor_max_model_len),
            "--gpu-memory-utilization", str(s.extractor_gpu_memory_utilization),
            "--max-num-seqs", str(s.extractor_max_num_seqs),
        ]
        if s.extractor_enforce_eager:
            extra.append("--enforce-eager")
        if s.extractor_text_only:
            extra += ["--limit-mm-per-prompt", '{"image": 0, "video": 0}']
        return self.start(
            s.extractor_model,
            port=port,
            phase=phase,
            max_num_batched_tokens=s.extractor_max_num_batched_tokens,
            extra_args=[*extra, *s.extractor_vllm_args],
        )

    def _wait_ready(self, url: str, timeout: int = 900) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(self._failure_report(
                    f"vLLM terminato con codice {self.proc.returncode}"))
            try:
                with urllib.request.urlopen(f"{url}/models", timeout=5) as r:
                    if r.status == 200:
                        log.info("vLLM ready: %s", url)
                        for line in _log_excerpt(self._log_file, self._log_offset)[1]:
                            log.info("vLLM: %s", line)
                        return
            except Exception:
                time.sleep(5)
        raise TimeoutError(self._failure_report(f"vLLM non pronto entro {timeout}s su {url}"))

    def _failure_report(self, headline: str) -> str:
        """Messaggio d'errore con la causa letta dal log del run corrente."""
        errors, memory = _log_excerpt(self._log_file, self._log_offset)
        parts = [headline]
        if errors:
            parts.append("Errore vLLM:\n  " + "\n  ".join(errors[-5:]))
        if memory:
            parts.append("Memoria:\n  " + "\n  ".join(memory))
        if not errors:
            parts.append("Nessun errore riconosciuto nel log")
        parts.append(f"Log completo: {self._log_file}")
        return "\n".join(parts)

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
