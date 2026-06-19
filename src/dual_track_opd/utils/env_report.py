"""Runtime environment reporting."""

from __future__ import annotations

import platform
import socket
import sys


def get_env_report() -> dict[str, object]:
    """Return basic Python, PyTorch, CUDA, and host information."""

    report: dict[str, object] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch": "not_installed",
        "cuda_available": False,
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return report

