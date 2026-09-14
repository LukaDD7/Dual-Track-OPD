"""Data-side scaffold construction for the STP-OPD pilot (CPU-testable).

Scaffold contract (see patches/verl/README.md): a scaffolded sample renders the
fixed verified answer-free teacher prefix as the assistant message content, so
the model continues generating the suffix after it; an unscaffolded sample has
no assistant prefix.  The FKL prefix region is scored by the actor on the
prefix tokens (path A), which requires the rollout to carry the prefix token
ids in ``extra_info``.

No verl import here so the message/token logic is unit-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class STPTransitionSample:
    prompt_id: str
    question: str
    image: Any
    scaffolded: bool
    prefix_token_ids: tuple[int, ...]

    @property
    def messages(self) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = [
            {"type": "image", "image": self.image},
            {"type": "text", "text": self.question},
        ]
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]
        if self.scaffolded and self.prefix_token_ids:
            # Prefix is rendered by the caller into assistant text (tokenizer
            # decode) — keeping token ids separate keeps identity auditable.
            messages.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
        return messages


def load_prefixes(path: str | Path) -> dict[str, tuple[int, ...]]:
    """Load ``{prompt_id: [token_ids]}`` from the prefix builder output."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(prompt_id): tuple(int(value) for value in token_ids)
        for prompt_id, token_ids in raw.items()
    }


def build_samples(
    rows: Sequence[Mapping[str, Any]],
    prefixes: Mapping[str, tuple[int, ...]],
    *,
    scaffold_flags: Mapping[str, bool],
) -> list[STPTransitionSample]:
    """Construct scaffolded/unscaffolded samples for a batch of prompt rows.

    ``scaffold_flags`` maps prompt_id -> scaffolded (from
    ``prefix_scaffold.assign_scaffold`` / ``paired_batch``); every prompt is
    emitted once per branch so comparisons stay prompt-paired.
    """

    samples: list[STPTransitionSample] = []
    for row in rows:
        prompt_id = str(row["sample_uid"])
        scaffolded = bool(scaffold_flags.get(prompt_id, False))
        for branch in (True, False):
            samples.append(
                STPTransitionSample(
                    prompt_id=prompt_id,
                    question=str(row["question"]).strip(),
                    image=row.get("images"),
                    scaffolded=scaffolded and branch,
                    prefix_token_ids=prefixes.get(prompt_id, ()),
                )
            )
    return samples
