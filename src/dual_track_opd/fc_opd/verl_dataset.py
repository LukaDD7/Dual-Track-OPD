"""Custom verl RLHFDataset that preserves FC-OPD fields through the pipeline.

verl's AgentLoop drops non-standard non-tensor columns from the batch.
The only reliable channel is ``extra_info``.  This dataset subclass ensures
FC-OPD fields (question, choices, answer, condition_inputs) are always
available inside ``extra_info``, regardless of the raw parquet layout.

For VA-OPD: student prompts are cleaned to raw image + question (no XML
format instructions, no answer choices).
"""

from __future__ import annotations

from typing import Any, Optional

from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset


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
        self.dataframe = self._clean_prompts(self.dataframe)

    @staticmethod
    def _clean_prompts(dataframe):
        """Replace prompts with industry-standard Geometry3K format.

        Uses the canonical math-RL prompt template (EasyR1, veRL, GAO_grpo,
        ARPO, MindSpeed-MM, Oumi all converge on this format):
          - ``<think>`` tags for chain-of-thought reasoning
          - ``\\boxed{}`` for the final answer (Qwen3 pre-training convention)

        VA-OPD: student sees only the raw image + question, no answer choices.
        """
        def _clean(row: dict) -> dict:
            question = str(row.get("question", "")).strip()
            if question:
                row["prompt"] = [{
                    "role": "user",
                    "content": (
                        f"<image>\n{question}\n\n"
                        "You FIRST think about the reasoning process as an "
                        "internal monologue and then provide the final answer. "
                        "The reasoning process MUST BE enclosed within "
                        "<think> </think> tags. "
                        "The final answer MUST BE put in \\boxed{}."
                    ),
                }]
            return row
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
