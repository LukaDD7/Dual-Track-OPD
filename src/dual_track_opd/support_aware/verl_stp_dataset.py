"""verl dataset subclass carrying STP-OPD scaffold prefix data.

Extends ``FCOPDDataset``: loads ``stp_prefixes.json`` and copies, into every
row's ``extra_info``, the verified answer-free teacher prefix token ids and
their decoded text.  The rollout-side patch (patches/verl/0003-stp-opd-rollout-
scaffold.patch) renders the assistant prefix for scaffolded samples using
``stp_scaffolded`` + ``stp_prefix_text``.
"""

from __future__ import annotations

from typing import Any, Optional

from transformers import PreTrainedTokenizer, ProcessorMixin

from dual_track_opd.fc_opd.verl_dataset import FCOPDDataset


class STPTransitionDataset(FCOPDDataset):
    """FCOPDDataset that attaches STP-OPD prefix ids/text to extra_info."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: Any,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
        prefix_manifest: str | None = None,
        scaffold_flags: dict[str, bool] | None = None,
    ):
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        from .support_transition_dataset import load_prefixes

        # verl instantiates the dataset with only
        # data_files/tokenizer/processor/config/max_samples, so the prefix
        # manifest and scaffold policy must come from data_config.
        config_get = getattr(config, "get", None)
        resolved_manifest = (
            prefix_manifest
            or (config_get("prefix_manifest") if config_get else None)
        )
        if not resolved_manifest:
            raise ValueError(
                "STPTransitionDataset requires data.prefix_manifest "
                "(verified answer-free teacher prefixes)"
            )
        self.prefixes = load_prefixes(resolved_manifest)
        if scaffold_flags is None and config_get is not None:
            scaffold_all = config_get("scaffold_all", False)
            if scaffold_all:
                scaffold_flags = {prompt_id: True for prompt_id in self.prefixes}
        self.scaffold_flags = scaffold_flags or {}
        self._prefix_text_cache: dict[str, str] = {}

    def _prefix_text(self, prompt_id: str) -> str:
        if prompt_id not in self._prefix_text_cache:
            token_ids = self.prefixes.get(prompt_id, ())
            text = self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            self._prefix_text_cache[prompt_id] = text
        return self._prefix_text_cache[prompt_id]

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        extra = row_dict.get("extra_info")
        if not isinstance(extra, dict):
            extra = {}
        prompt_id = str(extra.get("sample_uid") or row_dict.get("sample_uid") or "")
        extra["stp_prefix_token_ids"] = list(self.prefixes.get(prompt_id, ()))
        extra["stp_prefix_text"] = self._prefix_text(prompt_id)
        extra["stp_scaffolded"] = bool(self.scaffold_flags.get(prompt_id, False))
        row_dict["extra_info"] = extra
        # The verl agent loop receives dataset row fields as **kwargs, so the
        # scaffold fields must also live at the top level of the row dict.
        row_dict["stp_prefix_text"] = extra["stp_prefix_text"]
        row_dict["stp_scaffolded"] = extra["stp_scaffolded"]
        return row_dict
