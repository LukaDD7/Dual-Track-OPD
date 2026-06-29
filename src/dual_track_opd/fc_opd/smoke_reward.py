"""Minimal reward function for FC-OPD smoke test.

The smoke only needs to pass the reward step so the post-rollout hook fires.
Returns 1.0 for every response — real reward is irrelevant for FC-OPD validation.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, ground_truth, extra_info, kwargs
    return {"score": 1.0}
