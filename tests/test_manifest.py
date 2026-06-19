import json
from pathlib import Path

from dual_track_opd.eval.summarize import summarize_jsonl


def test_summarize_jsonl(tmp_path: Path):
    path = tmp_path / "raw.jsonl"
    rows = [
        {"dataset": "a", "finish_reason": "stop"},
        {"dataset": "a", "finish_reason": "length", "error": "timeout"},
        {"dataset": "b"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\nnot-json\n", encoding="utf-8")

    summary = summarize_jsonl(path)
    assert summary["rows"] == 4
    assert summary["errors"] == 2
    assert summary["finish_reason"] == {"stop": 1, "length": 1, "unknown": 1}
    assert summary["datasets"] == {"a": 2, "b": 1}

