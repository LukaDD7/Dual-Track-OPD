# Qwen3.5 Step C: Codex decisions, external evidence, and next executable gates

> Date: 2026-08-04 | Input: `docs/qwen35_stepc_decision_request_for_codex.md`
> Backend semantics checked against verl commit `334d9f8b`.

> **Outcome update:** D1/D1-L/D2 are complete.  Natural sampled thinking remained
> 83% clipped even at 8192; native non-thinking reached 28.5% clip and 9% accuracy.
> The current decision and corrected formal-teacher token path are in
> `docs/qwen35_d1_d2_codex_decision_20260804.md`.

## Executive correction

Do not treat the Step A/B/C clip rates as the training-rollout clip rate.  Those
runs used verl's validation path.  At the pinned backend commit, validation
defaults to greedy decoding (`temperature=0`, `top_p=1`, `top_k=-1`), while the
training rollout uses sampling.  Qwen3.5's own model card explicitly warns that
greedy decoding can cause endless repetition.  This is a more direct explanation
for the observed "fills either 2048 or 4096" pattern than a Geometry3K-specific
length contract.

Therefore the next gate is not `answer_only`.  First measure the current
reasoning-preserving prompt under the actual training sampling policy; then, if
needed, switch Qwen3.5's chat template to hard non-thinking mode while preserving
brief visible reasoning and the boxed answer.

## Q1: length contract and next ablation

### Decision

Keep `boxed_only` as the candidate training prompt and preserve natural thinking
until sampled evidence says otherwise.  Run these gates in order:

1. **Step D1 — sampled validation, default thinking, 2048 tokens.**  Use
   `temperature=1.0`, `top_p=0.95`, `top_k=-1`, matching the current training
   sampler rather than verl's greedy validation default.
2. **Step D1-L — naturally extended sampled validation, default thinking, 8192
   tokens.**  If D1 clips above 10%, keep the same prompt and sampler and change
   only the cap.  Run on the identical ordered 200 prompts.  This measures the
   natural completion-length distribution and whether extra sampled reasoning
   improves answer quality; it is not yet an 8192-token training decision.
3. **Step D2 — sampled validation, hard non-thinking, 2048 tokens.**  Run this
   only if D1-L still has high clip/repetition, or if its quality gain does not
   justify its token cost.  Add
   `data.apply_chat_template_kwargs.enable_thinking=False` and otherwise keep
   D1's sampler and prompt.  The prompt still requests brief visible reasoning.

For every gate, record response p50/p90/p95/p99, exact-cap clip rate, EOS rate,
boxed rate, accuracy, and repetition diagnostics (at minimum repeated 4-gram
ratio and longest repeated span).  Preserve the raw finish reason if the rollout
server exposes it.  A response whose token count equals the cap is only a proxy
for `finish_reason=length`, not proof by itself.

### Go/no-go rule

- If D1 clip rate is <=10% and accuracy does not regress, keep natural thinking
  and 2048 for the first short training smoke.
- If D1 clip is >10%, use D1-L to select the smallest cap above the sampled p95
  completion length.  Approve natural-thinking training only if at least 90% of
  responses terminate naturally, repetition is controlled, and accuracy/boxed
  rate improve enough to justify the extra generated tokens.
- If D1-L still clips above 30%, is dominated by repetition, or gains no answer
  quality, reject "just make it longer" and run D2.
- Approve `boxed_only + enable_thinking=False` only if D2 reduces clipping without
  materially degrading accuracy relative to D1.
- Do not add a stop-on-first-`\boxed{}` rule: Step C showed that boxes appearing
  mid-generation were usually wrong, so this would freeze intermediate answers.

## Q2: answer-only prompt

Do **not** approve the proposed answer-only text as the next or main training
contract.  It is syntactically fine as a secondary validation-only ablation, but
it is not Qwen3.5's reliable no-thinking control.  Qwen3.5 does not support the
Qwen3 `/think` and `/no_think` soft switch; the supported control is the chat
template argument `enable_thinking=False`.  Also, all correct non-clipped Step C
samples contained reasoning.

The prepared `answer_only` prompt and launcher may remain as a quarantined
validation-only diagnostic.  Run it only after D1/D2 if there is still a specific
need to separate "visible reasoning helps accuracy" from "thinking mode causes
length inflation".  It must not silently become a training prompt.

## Q3: `ROLLOUT_N=4`

Use this prompt-count configuration for the later `n=4` smoke:

```text
TRAIN_BATCH_SIZE=6
PPO_MINI_BATCH_SIZE=6
ROLLOUT_N=4
ROLLOUT_NUM_WORKERS=8
NGPUS_PER_NODE=3
```

The pinned backend repeats each prompt by `rollout.n` before generation and
multiplies the configured PPO mini-batch by `rollout.n` before actor update.
Thus this is 6 prompts x 4 = 24 generated sequences and a 24-sequence effective
PPO mini-batch.  `PPO_MINI_BATCH_SIZE=24` is wrong: the configured value is in
prompt units and cannot exceed `TRAIN_BATCH_SIZE=6`.

Keep eight agent-loop workers.  The 24 generated sequences divide evenly into
eight workers.  Changing to six workers is unnecessary.  The wrapper now checks
effective sequence counts and records both prompt and sequence counts.

Do not start `n=4` until D1/D2 establishes the sampling/thinking/length contract.
Then run at most 20 steps with fresh checkpoint, metadata, and validation paths.
Monitor per-step clip ratio, EOS rate, response-length quantiles, repetition,
reward, KL/distillation metrics, throughput, and peak memory.

## Q4: structurally zero format reward

Accept `format_rate=0` as a named limitation for the current line because
`USE_TASK_REWARDS=False`; do not inject literal `<think>...</think>` tags merely
to satisfy the legacy matcher.  This is especially inappropriate if D2 selects
native hard non-thinking mode.

Before any experiment enables task rewards, add a separately versioned
model-native format reward that checks the agreed final-answer contract (for
example one final boxed answer) without requiring hidden-thinking markup.  Do
not silently redefine the shared scorer, and do not claim an effective format
signal while it is structurally unreachable.

## External evidence

- Qwen3.5 defaults to thinking mode.  Its official model card warns against
  greedy decoding because it can produce endless repetition, and recommends
  sampled decoding (`temperature=1.0`, `top_p=0.95`, `top_k=20`) plus a presence
  penalty for general inference:
  https://huggingface.co/Qwen/Qwen3.5-4B
- ModelScope ms-swift's Qwen3.5 GRPO and GKD recipes explicitly disable thinking
  to avoid excessively long chain-of-thought, retain a step-by-step boxed-answer
  prompt, and use `max_completion_length=8192`:
  https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md
- EasyR1's public Geometry3K recipe uses reasoning plus `<think>`/`\boxed{}` with
  a 2048 response budget and exposes a response clip-ratio metric.  It does not
  report a Geometry3K clip rate comparable to ours:
  https://github.com/hiyouga/EasyR1/blob/main/examples/config.yaml
- RLinf's Geometry3K Qwen3-VL recipe uses a 4096-token response budget, eight
  rollouts per prompt, and truncated-importance sampling.  Its documentation
  says the importance-sampling fix controls excessive sequence-length growth
  and avoids likely reward collapse:
  https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/agentic/qwen3_vl_geo3k.html
- StableOPD reports that repetition-driven length inflation and truncation can
  emerge during OPD training and proposes a reference divergence constraint plus
  rollout-mixture distillation.  This is a later stabilization option, not an
  explanation for a step-zero greedy decoding failure:
  https://arxiv.org/abs/2604.08527

## Implementation boundary

Pinned verl already supports `data.apply_chat_template_kwargs` and passes it to
both dataset rendering and the agent loop.  It also supports rollout/validation
`temperature`, `top_p`, and `top_k`.  It does not expose Qwen's recommended
`presence_penalty` in the pinned rollout config, so do not fork the backend for
that before D1/D2.  A penalty-based sampler can alter the behavior policy and
must be audited against recomputed log-prob semantics before use in OPD training.

D2 is deliberately validation-only.  A later source audit corrected the earlier
teacher-prefix concern: the formal pinned verl agent loop sends the student's
exact `prompt_ids + response_ids` to the teacher vLLM `prompt_ids` interface, so
the teacher does not independently render raw messages.  The real precondition
is exact student/teacher token-ID semantic alignment; the Qwen35 wrapper now
fails fast on canonical tokenizer mapping mismatch and records the audit.  The
standalone FC-OPD teacher service is a separate raw-message-rendering path and
must not be used to describe formal verl behavior.
