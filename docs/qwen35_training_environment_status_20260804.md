# Dual-Track OPD current status: Qwen3.5 trainable environment and support diagnostics

> Status date: 2026-08-04
> Authoritative for the current two workstreams.  Supersedes the causal length
> interpretation in `qwen35_v1_truncation_smoke_results.md` and the proposed next
> action in `qwen35_stepc_decision_request_for_codex.md`.

## 1. Executive answer

Claude Code misidentified the core cause of the Step A/B/C truncation evidence.
The measured clip rates are real, but all three measurements came from verl's
**validation** sampler.  At pinned backend commit `334d9f8b`, validation defaults
to greedy decoding:

```yaml
actor_rollout_ref.rollout.val_kwargs:
  temperature: 0
  top_p: 1.0
  top_k: -1
```

Training rollout uses non-greedy sampling.  Qwen3.5's official model card warns
that greedy decoding may cause endless repetition and recommends sampled
decoding.  Therefore:

- `96% clip @2048` and `96% clip @4096` do **not** establish a training-rollout
  clip probability;
- they do not establish that natural longer reasoning is useless;
- `answer_only` and hard non-thinking are not the next default intervention;
- the missing experiment is natural thinking under non-greedy sampling.

We currently do not know the sampled clip probability or sampled answer quality.
Any numeric answer before Step D1/D1-L would be invented.

Official sources:

- Qwen3.5 model card and sampling guidance:
  https://huggingface.co/Qwen/Qwen3.5-4B
- ModelScope Qwen3.5 GRPO/GKD practice (non-thinking fallback, 8192 completion):
  https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md
- Geometry3K RLinf recipe (sampled rollouts, 4096 response, length-stability fix):
  https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/agentic/qwen3_vl_geo3k.html
- EasyR1 Geometry3K recipe (sampled training, 2048 response, clip metric):
  https://github.com/hiyouga/EasyR1/blob/main/examples/config.yaml

## 2. The two workstreams

### Track A — trainable Qwen3.6 -> Qwen3.5 OPD/GKD environment

Objective: obtain a reproducible, genuinely on-policy training path with online
teacher scoring, reachable task signal, controlled rollout length, and a valid
`n>1` rollout configuration.

Current state: **infrastructure works; sampling contract is the remaining gate.**

Completed evidence:

1. cu129/Qwen3.5 environment can load student, teacher, vLLM rollout, Ray, FSDP,
   and the online teacher scorer.
2. A 79-step qwen3.6-27B -> Qwen3.5-4B K1 run completed with finite loss, entropy,
   gradients, and checkpoints.
3. The FCOP dataset/prompt route is connected; validation receives the boxed
   instruction and ground truth.
4. The v1 20-step task-reward smoke completed.  Three batches contained correct
   responses and produced nonzero task policy-gradient loss, proving the reward
   reaches optimization.
5. Run manifests, resolved Hydra config, dataset/model hashes, backend commit,
   dirty diff, validation dumps, and logs are recorded outside Git.
6. Pinned-backend `rollout.n` semantics are resolved.  For 24 effective sequences
   at `n=4`, use 6 prompt rows and a six-prompt PPO mini-batch; eight rollout
   workers are valid because 24 divides by 8.

Not yet established:

- sampled clip/EOS probability for the base student;
- natural sampled response-length distribution;
- sampled accuracy/boxed quality at 2048 versus a longer cap;
- whether thinking should remain enabled for the first training run;
- teacher/student rendered-prefix policy if hard non-thinking is promoted from
  validation to training;
- a clean `n=4`, <=20-step GPU smoke under the selected sampling contract.

Readiness statement: the environment is **train-capable but not yet experiment-
ready for a long research run**.  The blocker is now an empirical decoding
contract, not CUDA/Ray/FSDP or reward wiring.

### Track B — support-aware frozen-policy diagnostic

Objective: determine whether teacher likelihood/support can identify useful rare
student rollouts before allocating bridge or support-gated training arms.

Current state: **diagnostic implementation is ready; K=32 confirmation remains a
separate frozen-policy job.  It is not training progress.**

Completed evidence:

1. K=8 full diagnostic covers 256 prompts and 2,304 exact-token scored responses.
2. Exact generated/student/teacher token identity, prompt hashes, response masks,
   EOS/truncation strata, resume, sharding, and strict merge checks are implemented.
3. A deterministic 64-prompt K=32 confirmation cohort and four-shard execution
   runbook are prepared; scoring is chunked to avoid 33-response padding/timeout
   failures.

Pending:

- run the two-prompt K=32 smoke;
- run and merge four 16-prompt K=32 shards;
- report K=8 -> K=32 reclassification, posterior pass probability, RL-ready
  prompt count, length/truncation strata, and exact-token protocol gates;
- only after that decide whether a bridge/support-gated micro-training cohort is
  scientifically justified.

Track B must not consume Track A's conclusion: a support diagnostic can be valid
while the formal Qwen3.5 training sampler is still undecided, and vice versa.

## 3. Correct experiment for natural sampled length and quality

Use the same ordered 200 Geometry3K validation prompts and the same
`boxed_only` prompt throughout.

| Gate | Thinking | Decoding | Cap | Question answered |
|---|---|---|---:|---|
| Step C (existing) | default | greedy | 2048 | Bad validation baseline; already 85.5% clipped |
| D1 | default | `T=1.0, top_p=.95, top_k=-1` | 2048 | What is the actual current training-sampler clip/quality? |
| D1-L | default | same as D1 | 8192 | Where does natural sampled reasoning finish, and does it improve answers? |
| D2 (fallback) | disabled | same as D1 | 2048 | Is long thinking, rather than sampling, the remaining problem? |

Why D1 uses `top_k=-1`: it reproduces the current training sampler exactly.  The
Qwen3.5 model card recommends `top_k=20`; changing that is a later, separately
named sampler ablation.  Do not mix it into the first greedy-versus-sampled test.

### Required metrics

For every arm, report both overall and paired-by-prompt comparisons:

- finish reason (`eos/stop/length`) where available;
- exact-cap clip proxy if finish reason is unavailable;
- response length p50/p75/p90/p95/p99/max;
- EOS rate, boxed rate, malformed rate, answer accuracy, reward mean/std;
- accuracy split by naturally finished versus length-truncated;
- repeated 4-gram fraction and longest repeated span;
- generation wall time, generated tokens, and peak memory;
- correct answers per 1,000 generated tokens as an efficiency diagnostic.

Do not call a response "naturally complete" merely because it contains a box.
Step C found boxes inside continued, usually wrong reasoning.  Completion requires
EOS/stop or a response shorter than the cap with a valid final answer.

### Decisions after measurement

1. **D1 clip <=10%, quality non-decreasing:** keep thinking, cap 2048, and use
   sampled validation for the first training smoke.
2. **D1 clip >10%, D1-L naturally finishes >=90%:** choose the smallest cap above
   D1-L's observed p95, rounded to a practical token boundary.  Accept the longer
   cap only if accuracy/boxed quality improves enough to justify token cost.
3. **D1-L clip >30%, strong repetition, or no quality gain:** reject blind
   lengthening and run D2 hard non-thinking.
4. **D2 improves termination but harms accuracy:** preserve thinking and treat
   length/termination as a training-stability problem; do not adopt answer-only.
5. Never add stop-on-first-box without a separate final-marker design: existing
   evidence shows early boxes are often intermediate wrong answers.

## 4. Track A next execution order

1. Run D1: `scripts/hpc/run_qwen35_v1_boxedonly_sampled_valonly.sh`.
2. If D1 clip is above 10%, run D1-L:
   `scripts/hpc/run_qwen35_v1_boxedonly_sampled_r8192_valonly.sh`.
3. Analyze paired quality/length and freeze the response cap in a versioned
   decision record.
4. Run D2 only if the natural sampled contract fails the rules above.
5. Before any non-thinking training, dump student and teacher rendered prefixes
   and explicitly align or document their template modes.
6. Run `n=4` for at most 20 steps with:

   ```text
   TRAIN_BATCH_SIZE=6
   PPO_MINI_BATCH_SIZE=6
   ROLLOUT_N=4
   ROLLOUT_NUM_WORKERS=8
   NGPUS_PER_NODE=3
   ```

7. Require finite OPD loss/gradient, nonzero task signal on some groups, stable
   entropy/KL, acceptable clip/repetition, correct six-prompt grouping, and no
   OOM before any longer run.
8. Only then freeze a first research configuration and launch a longer seeded
   experiment.  Raw JSONL, model weights, and checkpoints remain outside Git.

## 5. What is explicitly not decided

- We do not yet know that sampled decoding fixes truncation; D1 measures it.
- We do not yet know that 8192 improves quality; D1-L measures both quality and
  token cost.
- We do not assume Geometry3K itself causes this behavior.  Public Geometry3K
  recipes use sampled decoding and different Qwen generations, so they are
  supporting context, not an apples-to-apples result.
- We do not promote `answer_only` to training.
- We do not start Track B bridge training before K=32 confirmation.
