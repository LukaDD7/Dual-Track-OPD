"""vLLM server command helpers."""

from __future__ import annotations


def build_vllm_command(model_path: str, port: int = 8000) -> list[str]:
    """Build a minimal vLLM OpenAI server command."""

    return ["vllm", "serve", model_path, "--port", str(port)]

