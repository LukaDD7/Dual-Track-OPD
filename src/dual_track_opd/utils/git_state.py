"""Best-effort Git state reporting."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(args: list[str], cwd: str | Path | None) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def get_git_state(cwd: str | Path | None = None) -> dict[str, str | bool]:
    """Return commit, branch, and dirty status without raising on failure."""

    state: dict[str, str | bool] = {
        "commit": "unknown",
        "branch": "unknown",
        "dirty": "unknown",
    }
    try:
        state["commit"] = _git(["rev-parse", "HEAD"], cwd)
        state["branch"] = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
        state["dirty"] = bool(_git(["status", "--porcelain"], cwd))
    except Exception:
        return state
    return state

