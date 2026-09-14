# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TailSFT dataset: MultiTurnSFTDataset + a per-row `init_ce` float.

The init_ce column is precomputed offline by
scripts/sft_rl/annotate_tailsft_init_ce.py (ℓ0 = base-model length-normalized
mean CE over assistant target tokens, exact verl tokenization). This module
only re-emits it through __getitem__ so SFTTensorCollator packs it as a
NonTensorStack and the tailsft_loss can read it via tu.get(data, "init_ce").

Loaded by the trainer through data.custom_cls.{path,name}:

    data.custom_cls.path=scripts/sft_rl/tailsft_dataset.py
    data.custom_cls.name=TailsFTDataset

(parquet rows WITHOUT init_ce — e.g. plain warmup pools — emit init_ce=None,
which the collator packs as a NonTensorStack of Nones; tailsft_loss treats
that as the plain-SFT path.)
"""

from __future__ import annotations

from typing import Any

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

INIT_CE_KEY = "init_ce"


class TailsFTDataset(MultiTurnSFTDataset):
    def _read_files_and_process(self):
        # MultiTurnSFTDataset._read_files_and_process builds self.dataframe via
        # pd.concat over ALL parquet columns (so an init_ce column survives)
        # and then applies its max_samples row selection to self.dataframe
        # itself. Reading init_ce AFTER super() therefore stays row-aligned
        # with the selected dataframe for both shuffle and sequential modes.
        super()._read_files_and_process()

        if INIT_CE_KEY in self.dataframe.columns:
            self.init_ce = [None if v is None else float(v) for v in self.dataframe[INIT_CE_KEY].tolist()]
        else:
            self.init_ce = [None] * len(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, Any]:
        res = super().__getitem__(item)
        res[INIT_CE_KEY] = self.init_ce[item]
        return res
