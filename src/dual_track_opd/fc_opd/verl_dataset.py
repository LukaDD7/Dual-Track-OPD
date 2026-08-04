"""Custom verl RLHFDataset that preserves FC-OPD fields through the pipeline.

verl's AgentLoop drops non-standard non-tensor columns from the batch.
The only reliable channel is ``extra_info``.  This dataset subclass ensures
FC-OPD fields (question, choices, answer, condition_inputs) are always
available inside ``extra_info``, regardless of the raw parquet layout.

For VA-OPD: student prompts are normalized to the native Qwen3-VL-Instruct
image + question contract.
"""

from __future__ import annotations

from typing import Any, Optional

from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset

from dual_track_opd.fc_opd.prompt_contracts import (
    clean_geometry3k_question_rows,
    get_geometry3k_prompt_builder,
)


class FCOPDDataset(RLHFDataset):
    """RLHFDataset that normalises FC-OPD fields into ``extra_info``.

    Raw parquet files may store FC-OPD data at the top level or inside
    ``extra_info``.  This class copies every relevant field into
    ``extra_info`` so the post-rollout hook only needs to read one dict.
    """

    FC_OPD_KEYS = (
        "question",
        "choices",
        "answer",
        "answer_metadata",
        "condition_inputs",
        "fc_opd_condition_inputs",
    )

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: Any,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        prompt_version = str(config.get("prompt_version", "v1"))
        get_geometry3k_prompt_builder(prompt_version)  # fail fast on unknown version
        self.prompt_version = prompt_version
        self.dataframe = self._clean_prompts(self.dataframe, prompt_version)

    @staticmethod
    def _clean_prompts(dataframe, prompt_version: str):
        """Normalize prompts to the shared Qwen3-VL-Instruct contract.

        Do not inject literal ``<think>`` or boxed-answer instructions here:
        reasoning mode belongs to the model/chat template, and the same prompt
        must reach student rollout and teacher forced-forward scoring.
        """
        def _clean(row: dict) -> dict:
            return clean_geometry3k_question_rows([row], prompt_version)[0]
        return dataframe.map(_clean)

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        extra = row_dict.get("extra_info")
        if not isinstance(extra, dict):
            extra = {}
        # Copy any top-level FC-OPD keys that are not yet in extra_info.
        for key in self.FC_OPD_KEYS:
            if key not in extra and key in row_dict and row_dict[key] is not None:
                extra[key] = row_dict[key]
        # Handle nested condition_inputs column (non-dict in raw parquet).
        if "condition_inputs" not in extra and "condition_inputs" in row_dict:
            ci = row_dict["condition_inputs"]
            if isinstance(ci, dict):
                extra["condition_inputs"] = ci
            elif isinstance(ci, (list, tuple)):
                try:
                    extra["condition_inputs"] = dict(ci)
                except (TypeError, ValueError):
                    pass
        row_dict["extra_info"] = extra
        return row_dict

    def maybe_filter_out_long_prompts(self, dataframe=None):
        """Skip verl's chat-template-based length filter.

        Geometry3K prompts are pre-formatted and safely within the 6144-token
        budget.  The base-class filter uses closures that fail under
        multiprocessing pickling for our subclass.
        """
        max_len = getattr(self, "max_prompt_length", 6144)
        print(
            f"FCOPDataset: skipping chat-template-based filter; "
            f"keeping all {len(dataframe)} samples (pre-filtered <= {max_len})"
        )
        return dataframe
