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

import traceback

from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import build_multimodal_processor_inputs

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

        Prompt wording is selected only through the explicit version registry;
        the same versioned prompt must reach student rollout and teacher
        forced-forward scoring.  No version injects literal ``<think>`` tags.
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
        """Filter prompts using exact processor tokenization without multiprocessing.

        The base implementation uses a closure that can fail under multiprocessing
        pickling. This serial path keeps the exact text/multimodal length contract
        while avoiding that failure mode.
        """
        if not getattr(self, "filter_overlong_prompts", False):
            return dataframe

        keep_indices = []
        for index in range(len(dataframe)):
            doc = dataframe[index]
            try:
                messages = self._build_messages(doc, key=self.prompt_key)
                apply_kwargs = dict(**self.apply_chat_template_kwargs)
                if self.tool_schemas is not None:
                    apply_kwargs["tools"] = self.tool_schemas

                raw_prompt = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **apply_kwargs,
                )
                images, videos, audios = self._process_multi_modal_info(
                    messages,
                    self.image_patch_size,
                    self.config,
                )
                if images is None and videos is None and audios is None:
                    tokenized = self.processor.tokenizer(
                        text=raw_prompt,
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )
                    token_count = len(tokenized["input_ids"])
                else:
                    processed = build_multimodal_processor_inputs(
                        self.processor,
                        text=[raw_prompt],
                        images=images,
                        videos=videos,
                        audio=audios,
                        mm_processor_kwargs=self.mm_processor_kwargs,
                    )
                    token_count = len(processed["input_ids"][0])
            except Exception:
                traceback.print_exc()
                token_count = self.max_prompt_length + 1

            if token_count <= self.max_prompt_length:
                keep_indices.append(index)

        original_count = len(dataframe)
        filtered = dataframe.select(keep_indices)
        print(
            f"FCOPDataset: exact prompt filter kept {len(filtered)}/{original_count} "
            f"samples (max_prompt_length={self.max_prompt_length})"
        )
        return filtered
