# VA-OPD Server Readiness Record

This is the small, Git-safe summary that the CPU-instance Claude Code should fill after the human runs `scripts/hpc/collect_va_opd_gpu_facts.sh` on a GPU allocation. Keep the raw `gpu_facts.txt` outside Git. Remove hostnames, UUIDs, usernames, scheduler account names, and process command lines from any committed summary.

## Facts already reported on 2026-07-21

Do not re-label these as current without a fresh allocation report:

- node inventory observed 8× H200, about 141 GB each;
- driver `570.124.06`, reported CUDA capability 12.8;
- topology was NV18 full mesh;
- physical GPUs 0–3 were idle at collection time;
- GPUs 4–7 had a user-owned keepalive PID at collection time;
- shared storage reported about 240 GB free;
- `NVIDIA_VISIBLE_DEVICES` exposed eight UUIDs.

These facts justify a cu128/SM90 build, but they do **not** prove that the next scheduler allocation has six free GPUs or that a compiled vLLM kernel runs. Refresh PID, memory, topology and mount visibility before every GPU gate.

## Record identity

- Date/time:
- Project commit:
- Backend commit:
- CPU environment build log:
- Environment prefix (must end in `cu128-v2`):
- Environment build-manifest SHA-256:
- vLLM source commit (expected `4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e`):
- vLLM wheel SHA-256:
- Raw GPU facts path outside Git:
- Prepared by:

## Storage and artifact visibility

- Shared root readable from CPU: yes/no
- Shared root readable from GPU: yes/no
- Free bytes / free inodes:
- Environment prefix readable:
- Backend checkout readable:
- Student checkpoint readable:
- Teacher checkpoint readable:
- Train/validation parquet readable:
- Full/degraded image assets readable:

## GPU allocation

- GPU model:
- Physical GPU count allocated:
- Memory per GPU:
- Driver version:
- MIG mode:
- `CUDA_VISIBLE_DEVICES`:
- Actor physical IDs (4 GPUs):
- Teacher physical IDs (2 GPUs, or documented TP=1 fallback):
- Actor topology summary:
- Pre-existing compute PIDs: none/list with ownership confirmed
- ECC/Xid/error status:

## Runtime identity

- Python:
- torch / torch CUDA runtime:
- vLLM:
- transformers:
- Ray:
- TensorDict:
- FlashInfer:
- flash-attn:
- NCCL reported by torch:
- `nvcc` on PATH:
- Approved conda CUDA 12.8 toolchain path:
- System nvcc excluded from builds: yes/no
- cu129/cu130 contamination found: yes/no
- Build kind (expected `cpu-source-build-cu128-h200-sm90`):
- Real CUDA kernel/model smoke completed: yes/no (an import-only test is insufficient)

## Required gates

| Gate | Command/run ID | Result | Evidence path | Notes |
|---|---|---|---|---|
| CPU full-image audit | `--preflight-only --audit-all-images` | pending |  |  |
| 4-rank NCCL collective smoke | `smoke_va_opd_nccl.py` | pending |  |  |
| Native OPD 3-step smoke | `gate_b_opd` | pending |  |  |
| VA-OPD 3-step smoke | `gate_c_va` | pending |  |  |
| OPD 50-step pilot | `pilot50_opd` | pending |  |  |
| VA-OPD 50-step pilot | `pilot50_va` | pending |  |  |
| OPD 5-epoch run | `full5e_opd` | blocked until pilot |  |  |
| VA-OPD 5-epoch run | `full5e_va` | blocked until pilot |  |  |

## Smoke/pilot metrics

| Metric | OPD | VA-OPD | Gate/interpretation |
|---|---:|---:|---|
| completed steps |  |  | Must equal request |
| finite distillation loss count |  |  | Must be positive |
| finite gradient count |  |  | Must be positive |
| actor entropy range |  |  | Must remain finite; inspect collapse trajectory |
| response length mean/range |  |  | Inspect growth toward 2048 |
| response clip ratio |  |  | Persistent near-1 blocks full run |
| validation score: step 0 |  |  | Baseline |
| best validation score / step |  |  | Used for checkpoint selection |
| final validation score |  |  | Report even if worse than best |
| VA mean | N/A |  | Must be present and not identically zero |
| VA positive ratio | N/A |  | Inspect extreme 0/1 behavior |
| rollout group max sum error | N/A |  | Must be ≤ 1e-5 |
| NCCL/worker/Xid errors |  |  | Any occurrence needs triage |
| CPU RSS trajectory |  |  | Sustained growth needs triage |

## Deviations from the canonical launcher

List every override. “None” is a valid and preferred answer.

- GPU layout:
- Batch/microbatch:
- Sequence lengths:
- Memory utilization:
- Teacher TP/max sequences:
- NCCL environment:
- Package/environment changes:
- Retry count and reason:

## Decision

- Readiness: PASS / FAIL / CONDITIONAL
- Allowed next gate:
- Blocking issue:
- Evidence supporting the decision:
- Early checkpoint selection rule:
- Owner of next GPU command:
