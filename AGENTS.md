# Agent Instructions

This repository is a research operating system for dual-track on-policy distillation in VLMs. Treat reproducibility and separation of concerns as first-class requirements.

## Working Rules

- Keep research logic in `src/dual_track_opd/`, not inside third-party backend code.
- Keep experiments configuration-driven under `configs/`.
- Prefer small, tested utilities over large framework rewrites.
- Do not commit secrets, `.env`, model weights, datasets, checkpoints, raw eval JSONL outputs, or large artifacts.
- Keep `third_party/verl/` as a placeholder/submodule area unless explicitly asked to vendor or fork.
- Store minimal patches under `patches/verl/` with clear explanations.

## Reproducibility

Every experiment should record:

- repo git commit and dirty status
- backend commit or package version
- full resolved config
- dataset manifest hash
- model checkpoint path
- raw eval output path outside Git
- summary metrics and notes

## Research Algorithm Change Protocol

Research correctness has three authority levels, in this order:

1. user-approved scientific intent, derivation, and algorithm contract
2. executable invariants and reference tests
3. implementation and backend integration code

Implementation must not redefine the algorithm. If code conflicts with an
approved invariant, report `SPEC-CODE CONFLICT`; do not weaken the invariant
or change the specification merely to make tests pass.

Treat losses, rollout weighting/grouping, masks, reductions, normalization,
gradient routing, sampling, configuration propagation, and research metrics
as algorithm-critical code. For changes in those areas:

1. Read the relevant implementation, experiment config, stored patch, and
   algorithm or implementation report before editing.
2. Write a short semantic map covering inputs, tensor shapes, reductions,
   grouping, detach/gradient behavior, and downstream consumers.
3. Separate repository-local facts from behavior that depends on an external
   backend, distributed runtime, dataset, model checkpoint, or GPU.
4. Add or update exact tests for the intended invariants before or alongside
   the smallest implementation change.
5. Verify disable-path compatibility and uniform/no-op recovery when the
   algorithm defines a baseline-equivalent case.
6. Run targeted tests first, then broader tests when the local environment
   supports them. Report missing dependencies instead of claiming success.
7. Report conclusions under `VERIFIED LOCALLY`,
   `REQUIRES SERVER VERIFICATION`, or `UNKNOWN`.

Do not infer algorithm correctness only because training starts, the loss is
finite, or a checkpoint is produced. Verify tensor dimensions, reduction
dimensions, mask semantics, group isolation, detach/gradient flow, config
propagation, and diagnostic definitions.

## NLL-TailOPD v1 Safety Island

NLL-TailOPD is the first normalized experiment chain in this repository:

- core algorithm: `src/dual_track_opd/tail_opd/weights.py`
- invariant tests: `tests/tail_opd/`
- experiment config: `configs/experiment/nll_tailopd_geometry3k_v1.yaml`
- backend contract: `configs/backend/verl_qwen35_v090_cu132.yaml`
- local verifier: `scripts/verify_backend_contract.py`
- external integration patch: `patches/verl/nll_tailopd_v1.patch`
- server handoff: `docs/NLL_TailOPD_v1_Backend_Handoff_20260918.md`

V1 is rollout-level NLL weighting only: masked sequence NLL, per-uid group
z-score, softmax weights, and detached `K*w` scaling. Do not add correctness
routing, token-level weighting, or teacher gating to V1. The core utility may
support arbitrary group sizes, but formal backend integration must fail fast
unless every uid group has exactly the configured `ROLLOUT_N` siblings.

The active training backend is an external VERL worktree. A local patch check
does not establish backend reproducibility. Do not mark a backend contract
verified until the server audit reconstructs the ordered patch stack, accounts
for all dirty diffs, passes the backend integration contract, and completes the
required semantic GPU smoke.

## Commands

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m dual_track_opd.eval.summarize path/to/raw.jsonl
python scripts/verify_backend_contract.py
```

## FC-OPD Training Notes

- Do not remove or bypass the online student scorer in the main FC-OPD training path. The core research claim depends on comparing teacher vs student sensitivity under different visual/text conditions to localize student capability deficits. A teacher-only router is only acceptable as an explicitly named ablation.
- For long HPC runs, a post-run GPU keepalive may be launched after training has completed and after Ray/teacher cleanup, preferably only when all requested steps finished successfully. The current requested command is:

```bash
CUDA_VISIBLE_DEVICES=3,4 KEEPALIVE_TARGET_UTIL=0.45 KEEPALIVE_WORK_ITERS=32 python -u /inspire/hdd/global_user/mengweicheng-240108120092/lzy/scripts/busy_keepalive.py
```

- A suitable integration point is near the end of `scripts/hpc/run_verl_fc_opd_overnight.sh`, after `ray stop -f` and teacher shutdown, before the final `exit ${VERL_EXIT}`. Keep it opt-in, for example behind `--keepalive-after-success` or an environment variable, and do not start it on failed or interrupted runs unless explicitly requested.

## Git Hygiene

Before committing, inspect:

```bash
git status --short
git diff --stat
```

Do not add large files. If a file looks like data, weights, raw output, or a secret, leave it out and update `.gitignore` instead.
