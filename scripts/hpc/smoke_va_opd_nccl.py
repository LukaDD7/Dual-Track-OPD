#!/usr/bin/env python3
"""Exercise the historical VA-OPD all-gather sizes on exactly four actor ranks."""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--elements",
        default="4198742,83678470",
        help="Comma-separated BF16 element counts; defaults cover the historical small and root-shard gathers.",
    )
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    sizes = [int(value) for value in args.elements.split(",") if value.strip()]
    if not sizes or min(sizes) < 1 or args.iterations < 1:
        raise ValueError("element sizes and iterations must be positive")

    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError(f"VA-OPD actor collective smoke requires exactly 4 ranks, got {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    for elements in sizes:
        source = torch.full((elements,), rank + 1, dtype=torch.bfloat16, device="cuda")
        gathered = torch.empty((world_size * elements,), dtype=torch.bfloat16, device="cuda")
        for iteration in range(args.iterations):
            dist.barrier()
            dist.all_gather_into_tensor(gathered, source)
            dist.all_reduce(source)
            torch.cuda.synchronize()
            expected_reduce_value = float((world_size + 1) * world_size / 2)
            actual_reduce_value = float(source[0].item())
            gathered_rows = gathered.view(world_size, elements)
            gather_ok = all(
                float(gathered_rows[source_rank, position].item()) == float(source_rank + 1)
                for source_rank in range(world_size)
                for position in (0, elements - 1)
            )
            if not gather_ok or actual_reduce_value != expected_reduce_value:
                raise RuntimeError(
                    f"collective corruption at elements={elements}, iteration={iteration}: "
                    f"gather_boundary_check={gather_ok}, "
                    f"reduce={actual_reduce_value}/{expected_reduce_value}"
                )
            # Restore rank-local input after all_reduce for the next iteration.
            source.fill_(rank + 1)
        if rank == 0:
            gib = elements * 2 / (1024**3)
            print(f"PASS elements={elements} input_per_rank={gib:.3f} GiB iterations={args.iterations}", flush=True)

    dist.destroy_process_group()
    if rank == 0:
        print("VA-OPD four-rank NCCL smoke: PASS", flush=True)


if __name__ == "__main__":
    main()
