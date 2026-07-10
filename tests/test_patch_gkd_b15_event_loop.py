import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path("scripts/hpc/patch_gkd_b15_event_loop.py")
SPEC = importlib.util.spec_from_file_location("patch_gkd_b15_event_loop", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
patch_source = MODULE.patch_source


BROKEN_SOURCE = """\
import asyncio

class Worker:
    def sync_rollout_weights(self):
        for tensors in []:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            loop.run_until_complete(self.rollout.update_weights(iter(tensors)))
"""


def test_patch_is_minimal_valid_and_idempotent():
    patched, status = patch_source(BROKEN_SOURCE)
    assert status == "patched"
    assert "with asyncio.Runner() as runner:" in patched
    assert "runner.run(self.rollout.update_weights(iter(tensors)))" in patched
    assert "get_event_loop" not in patched
    assert patch_source(patched) == (patched, "already_patched")


def test_unknown_backend_revision_fails_closed():
    with pytest.raises(ValueError, match="Refusing to patch"):
        patch_source("class Worker: pass\n")


def test_cli_apply_then_check(tmp_path: Path):
    target = tmp_path / "megatron_workers.py"
    target.write_text(BROKEN_SOURCE, encoding="utf-8")
    subprocess.run([sys.executable, str(SCRIPT), str(target)], check=True)
    subprocess.run([sys.executable, str(SCRIPT), str(target), "--check"], check=True)
