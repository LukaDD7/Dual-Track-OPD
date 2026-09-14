"""Ray-actord StudentScorer for verl FC-OPD hook.

The verl post-rollout hook runs in the TaskRunner (a CPU-only Ray actor).
Standalone :class:`StudentScorer` needs CUDA, so we wrap it in a Ray actor
that owns a GPU and proxy calls through a synchronous adapter.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .conditions import Condition
from .online_batch import OnlineFCOPDSample, OnlineStudentScores, StudentForcedScorer

try:
    import ray
except ModuleNotFoundError:  # pragma: no cover - exercised when optional Ray is absent.
    ray = None


if ray is not None:

    @ray.remote(num_gpus=1)
    class _RayStudentScorerActor:
        """Ray actor that owns a GPU and delegates to :class:`StudentScorer`."""

        def __init__(self, student_scorer_fqn: str, student_scorer_kwargs: dict[str, Any]):
            self._scorer = _build_scorer(student_scorer_fqn, student_scorer_kwargs)

        def score(self, sample: OnlineFCOPDSample, conditions: list[Condition]) -> OnlineStudentScores:
            result = self._scorer(sample, conditions)
            return OnlineStudentScores(
                loss_logits=result.loss_logits.detach().cpu(),
                condition_log_probs={
                    c: lp.detach().cpu() if isinstance(lp, torch.Tensor) else lp
                    for c, lp in result.condition_log_probs.items()
                },
            )

        def score_batch(
            self, samples: list[OnlineFCOPDSample], conditions: list[Condition]
        ) -> list[OnlineStudentScores]:
            results = self._scorer(samples, conditions)
            return [
                OnlineStudentScores(
                    loss_logits=r.loss_logits.detach().cpu(),
                    condition_log_probs={
                        c: lp.detach().cpu() if isinstance(lp, torch.Tensor) else lp
                        for c, lp in r.condition_log_probs.items()
                    },
                )
                for r in results
            ]

else:
    _RayStudentScorerActor = None


class RayStudentScorerProxy:
    """Synchronous proxy that forwards to a Ray GPU actor.

    Instantiating this creates a *detached* Ray actor; the proxy holds a
    handle and forwards every ``__call__`` via ``ray.get(actor.score.remote(...))``.
    """

    def __init__(
        self,
        *,
        student_scorer_fqn: str,
        student_scorer_kwargs: dict[str, Any] | None = None,
        ray_actor_options: dict[str, Any] | None = None,
    ):
        _require_ray()
        kwargs = dict(student_scorer_kwargs or {})
        actor_options = dict(ray_actor_options or {})
        actor_options.setdefault("num_gpus", 1)
        assert _RayStudentScorerActor is not None
        self._actor = _RayStudentScorerActor.options(**actor_options).remote(
            student_scorer_fqn=student_scorer_fqn,
            student_scorer_kwargs=kwargs,
        )

    def __call__(
        self, sample_or_samples: OnlineFCOPDSample | list[OnlineFCOPDSample], conditions: list[Condition]
    ) -> OnlineStudentScores | list[OnlineStudentScores]:
        _require_ray()
        assert ray is not None
        if isinstance(sample_or_samples, list):
            results: list[OnlineStudentScores] = ray.get(
                self._actor.score_batch.remote(sample_or_samples, conditions)
            )
            return [
                OnlineStudentScores(
                    loss_logits=r.loss_logits.detach().cpu(),
                    condition_log_probs={
                        c: lp.detach().cpu() if isinstance(lp, torch.Tensor) else lp
                        for c, lp in r.condition_log_probs.items()
                    },
                )
                for r in results
            ]
        result: OnlineStudentScores = ray.get(self._actor.score.remote(sample_or_samples, conditions))
        return OnlineStudentScores(
            loss_logits=result.loss_logits.detach().cpu(),
            condition_log_probs={
                c: lp.detach().cpu() if isinstance(lp, torch.Tensor) else lp
                for c, lp in result.condition_log_probs.items()
            },
        )


def build_ray_student_scorer_proxy(
    *,
    student_scorer_fqn: str,
    student_scorer_kwargs: dict[str, Any] | None = None,
    ray_actor_options: dict[str, Any] | None = None,
) -> StudentForcedScorer:
    """Create a Ray-proxied student scorer (GPU actor).

    Use this when the hook runs in a CPU-only process (e.g. verl TaskRunner).
    """
    proxy = RayStudentScorerProxy(
        student_scorer_fqn=student_scorer_fqn,
        student_scorer_kwargs=student_scorer_kwargs,
        ray_actor_options=ray_actor_options,
    )
    return proxy


def _require_ray() -> None:
    if ray is None:
        raise ModuleNotFoundError("RayStudentScorerProxy requires the optional 'ray' package")


def _build_scorer(fqn: str, kwargs: dict[str, Any]) -> Callable[..., Any]:
    import importlib

    module_name, _, attr = fqn.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"expected a fully qualified name, got: {fqn}")
    module = importlib.import_module(module_name)
    cls = getattr(module, attr)
    return cls(**kwargs)
