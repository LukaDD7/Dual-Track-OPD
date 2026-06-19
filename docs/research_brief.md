# Research Brief

Dual-Track-OPD studies on-policy distillation for vision-language models, starting with Qwen3-VL-8B-Instruct.

The working hypothesis is that standard on-policy distillation can dilute sparse visual-dependence signals. If a benchmark contains many questions answerable from language priors or weak visual evidence, uniform token-level KL may mostly reinforce text-side behavior while under-training genuinely visual reasoning.

Dual-track OPD explicitly separates and combines:

- text-side learning signals
- vision-side learning signals
- token-level visual anchors
- chunk-level visual anchors

The initial code uses simple proxy heuristics so the repository stays runnable. Later work should replace these proxies with VLM-specific visual-dependence metrics from rollouts, counterfactual image ablations, attention/gradient signals, or teacher/student disagreement under visual perturbation.

