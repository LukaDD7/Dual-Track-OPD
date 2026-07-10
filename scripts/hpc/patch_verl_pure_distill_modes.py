#!/usr/bin/env python3
"""Idempotently extend an already-applied verl actor patch with pure GKD/JSD modes."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = 'pure_fc_opd = fc_opd_mode in ("vgg_opd", "va_opd")'
NEW = """pure_fc_opd = fc_opd_mode in (
                    \"gkd\", \"gkd_forward\", \"vgg_opd\", \"va_opd\", \"va_opd_jsd\"
                )"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor", type=Path)
    args = parser.parse_args()
    actor = args.actor.resolve()
    text = actor.read_text(encoding="utf-8")
    if NEW in text:
        print(f"pure distillation modes: already current - {actor}")
        return
    if OLD not in text:
        raise SystemExit(f"cannot locate the expected FC-OPD pure-mode line in {actor}")
    actor.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"pure distillation modes: patched - {actor}")


if __name__ == "__main__":
    main()
