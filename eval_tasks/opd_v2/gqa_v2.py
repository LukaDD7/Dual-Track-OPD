import sys
from pathlib import Path

import datasets
from datasets import load_dataset

# Sibling-module import: lmms-eval loads task utils via spec_from_file_location
# (no package context), so put this directory on sys.path deterministically.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _extraction import extract_answer_tag, short_answer  # noqa: E402,F401


def gqa_v2_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    gqa_raw_image_dataset = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
    gqa_id2image = {}
    for row in gqa_raw_image_dataset:
        gqa_id2image[row["id"]] = row["image"].convert("RGB")

    def _process_doc(doc):
        image = gqa_id2image[doc["imageId"]]
        return {
            "image": image,
        }

    return dataset.map(_process_doc, num_proc=8)


def gqa_v2_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def gqa_v2_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def gqa_v2_process_results(doc, results):
    # GQA gold answers are single words/phrases.  Model may answer directly
    # (short-response checkpoints) or finish chain-of-thought and emit an
    # <answer> tag (long-CoT checkpoints) — both are valid here.
    pred = short_answer(results[0])
    return {"exact_match": pred}
