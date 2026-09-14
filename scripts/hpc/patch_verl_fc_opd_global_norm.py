#!/usr/bin/env python3
"""Idempotently upgrade an applied FC-OPD actor overlay to global DP normalization."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor", type=Path)
    args = parser.parse_args()
    path = args.actor.resolve()
    text = path.read_text(encoding="utf-8")

    if "fc_opd_global_normalizer" in text:
        print(f"FC-OPD global normalization: already current - {path}")
        return

    old_import = "    fc_opd_batch_denominator,\n    has_fc_opd_tensors,"
    new_import = "    fc_opd_batch_denominator,\n    fc_opd_global_normalizer,\n    has_fc_opd_tensors,"
    old_loss = """                        denominator = fc_opd_denominator.to(fc_opd_loss_sum.device)
                        fc_opd_loss = fc_opd_loss_sum / denominator"""
    new_loss = """                        denominator, dp_world_size = fc_opd_global_normalizer(
                            fc_opd_denominator.to(fc_opd_loss_sum.device)
                        )
                        fc_opd_loss = fc_opd_loss_sum * dp_world_size / denominator"""
    old_metric = '                        micro_batch_metrics["actor/fc_opd_denominator"] = denominator.detach().item()'
    new_metric = old_metric + '\n                        micro_batch_metrics["actor/fc_opd_dp_world_size"] = float(dp_world_size)'

    for old, label in ((old_import, "import"), (old_loss, "loss"), (old_metric, "metric")):
        if old not in text:
            raise SystemExit(f"cannot locate expected FC-OPD {label} block in {path}")
    text = text.replace(old_import, new_import, 1)
    text = text.replace(old_loss, new_loss, 1)
    text = text.replace(old_metric, new_metric, 1)
    path.write_text(text, encoding="utf-8")
    print(f"FC-OPD global normalization: patched - {path}")


if __name__ == "__main__":
    main()
