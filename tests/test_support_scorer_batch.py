"""CPU-level regression tests for StudentScorer.score_batch input construction.

The batch path regressed by reusing single-image tensors (from the prompt-only
encoding) together with a multi-row padded ``input_ids``.  Qwen3-VL then fails
in ``get_placeholder_mask`` with "Image features and image tokens do not match"
— the batch holds n image-token blocks but only one image's features.  These
tests exercise the processor-level construction (no LLM weights) and a stubbed
forward call, so they run on CPU.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from dual_track_opd.support_aware.scorer import StudentScorer


MODEL_DIR = "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-4B-Instruct"

requires_model = pytest.mark.skipif(
    not os.path.isdir(MODEL_DIR),
    reason="Qwen3-VL-4B model directory not available on this machine",
)


def _spatial_merge_size() -> int:
    with open(os.path.join(MODEL_DIR, "config.json"), encoding="utf-8") as fh:
        return int(json.load(fh)["vision_config"]["spatial_merge_size"])


def _chat_text(processor, image, text="Find x. Think step by step.") -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@pytest.fixture(scope="module")
def processor():
    if not os.path.isdir(MODEL_DIR):
        pytest.skip("Qwen3-VL-4B model directory not available on this machine")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL_DIR)


def _responses() -> list[str]:
    return [
        "Let me compute this carefully. The answer is 7.",
        "",
        "A longer response with more reasoning tokens: " + "x " * 80,
    ]


def _image() -> Image.Image:
    return Image.new("RGB", (600, 400), (200, 120, 30))


@requires_model
def test_batch_encoding_invariants(processor):
    """Each row must carry its own image tensors, padded to a common length."""
    image = _image()
    chat = _chat_text(processor, image)
    responses = _responses()
    n = len(responses)

    batch = processor(
        text=[chat + r for r in responses],
        images=[image] * n,
        return_tensors="pt",
        padding=True,
    )

    # All text-side tensors share the padded (n, max_len) shape.
    assert batch["input_ids"].shape[0] == n
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape

    # One image per row.
    assert batch["image_grid_thw"].shape == (n, 3)
    assert batch["pixel_values"].shape[0] == int(
        batch["image_grid_thw"].prod(dim=1).sum()
    )

    # Qwen3-VL checks that total image-pad tokens equal total vision features;
    # each row contributes (grid product / spatial_merge_size**2) features.
    merge = _spatial_merge_size()
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    per_row_tokens = (batch["input_ids"] == image_token_id).sum(dim=1)
    per_row_features = batch["image_grid_thw"].prod(dim=1) // (merge * merge)
    assert torch.equal(per_row_tokens, per_row_features)


@requires_model
def test_batch_encoding_matches_serial(processor):
    """Batched rows must tokenize identically to the per-item encodings."""
    image = _image()
    chat = _chat_text(processor, image)
    responses = _responses()
    n = len(responses)

    prompt_enc = processor(text=[chat], images=[image], return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    serial = [
        processor(text=[chat + r], images=[image], return_tensors="pt")
        for r in responses
    ]
    batch = processor(
        text=[chat + r for r in responses],
        images=[image] * n,
        return_tensors="pt",
        padding=True,
    )

    for i, enc in enumerate(serial):
        length = enc["input_ids"].shape[1]
        assert torch.equal(batch["input_ids"][i][:length], enc["input_ids"][0])
        # Response slice must be identical too (log-prob extraction window).
        if responses[i]:
            serial_slice = enc["input_ids"][0][prompt_len:]
            batch_slice = batch["input_ids"][i][prompt_len:length]
            assert torch.equal(batch_slice, serial_slice)


class _StubModel:
    """Records the forward kwargs and returns zero logits (uniform probs)."""

    def __init__(self):
        self.received: dict[str, torch.Tensor] = {}
        self._param = torch.zeros(1)

    def parameters(self):
        return iter([self._param])

    def __call__(self, **kwargs):
        self.received = {k: v.clone() for k, v in kwargs.items()}
        n, max_len = kwargs["input_ids"].shape
        vocab = int(kwargs["input_ids"].max().item()) + 2
        logits = torch.zeros((n, max_len, vocab), dtype=torch.float16)
        return SimpleNamespace(logits=logits)


def _stub_scorer(processor) -> tuple[StudentScorer, _StubModel]:
    scorer = object.__new__(StudentScorer)  # bypass the 4B model load
    scorer._processor = processor
    scorer._tokenizer = processor.tokenizer
    model = _StubModel()
    scorer._model = model
    return scorer, model


@requires_model
def test_score_batch_forward_receives_per_row_multimodal_tensors(processor):
    """score_batch must hand the model n images / grids and padded token ids."""
    image = _image()
    chat = _chat_text(processor, image)
    responses = _responses()
    n = len(responses)
    questions = ["Find x."] * n
    prompt_texts = ["Find x. Think step by step."] * n
    token_ids_list = [
        tuple(int(t) for t in processor.tokenizer.encode(r)) if r else ()
        for r in responses
    ]

    scorer, model = _stub_scorer(processor)
    results = scorer.score_batch(
        questions=questions,
        images=[image] * n,
        prompt_texts=prompt_texts,
        response_texts=responses,
        response_token_ids_list=token_ids_list,
    )

    received = model.received
    assert received["input_ids"].shape[0] == n
    assert received["mm_token_type_ids"].shape == received["input_ids"].shape
    assert received["image_grid_thw"].shape == (n, 3)
    assert received["pixel_values"].shape[0] == int(
        received["image_grid_thw"].prod(dim=1).sum()
    )
    assert received["attention_mask"].shape == received["input_ids"].shape

    # Empty responses produce an error result; non-empty rows get a token_count.
    assert len(results) == n
    assert results[1].error == "empty response"
    assert results[0].token_count > 0
    assert results[2].token_count > 0
    assert all(r.mean_logp == results[0].mean_logp for r in results if r.token_count)
