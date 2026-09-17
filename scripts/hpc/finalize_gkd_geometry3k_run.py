#!/usr/bin/env python3
"""Append terminal validation results to a GKD run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--requested-steps", type=int, required=True)
    parser.add_argument("--loss-lines", type=int, required=True)
    parser.add_argument("--finite-grad-lines", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    passed = (
        args.exit_code == 0
        and args.updates >= args.requested_steps
        and args.loss_lines > 0
        and args.finite_grad_lines > 0
    )
    payload["result"] = {
        "passed": passed,
        "exit_code": args.exit_code,
        "actor_updates": args.updates,
        "requested_steps": args.requested_steps,
        "distillation_loss_log_lines": args.loss_lines,
        "finite_grad_log_lines": args.finite_grad_lines,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
    }
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Run manifest finalized: passed={passed} - {args.manifest}")


if __name__ == "__main__":
    main()
