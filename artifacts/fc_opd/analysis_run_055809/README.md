# Run 055809 Analysis Artifacts

- **Date**: 2026-07-08 05:58 UTC
- **Run ID**: `fc_opd_overnight_noreshard_noreshard_20260708_055809`
- **Config**: FSDP2, reshard_after_forward=false, `<think>` + `\boxed{}` prompt, max_response_length=2048
- **Fixes applied**: trim keep-last-N (#17), prompt `\boxed{}` (#13), noreshard deadlock workaround (#15)
- **Status**: 29 steps completed, mode collapse observed (entropy 1.3→0.46, score 0.83→0.00)

## Files

- `train_*.log` — Full training metrics per step (console logger output)
- `val_0.jsonl` — val_before_train generations (score 0.35)
- `1.jsonl` – `29.jsonl` — Per-step rollout generations (48 responses × 2 conditions each)
- See `docs/VA-OPD_BUG_TRACKER.md` for bug context (#13-18)

## Key Observations

1. **Phase 1 (steps 1-11)**: Oscillating exploration, score 0-0.50, entropy 0.48→0.85
2. **Phase 2 (steps 12-20)**: 🟢 Best performance — entropy 1.0-1.3, clip 4-27%, score up to 0.83 (step 14)
3. **Phase 3 (steps 21-29)**: 🔴 Mode collapse — entropy 1.3→0.46, clip 27%→92%, score→0.00

Pattern: pure KL distillation (VA-OPD) produces healthy exploration initially, but
without reward anchoring, the distribution collapses to deterministic long outputs.

GPU: 8× H200, 6 training + 2 teacher (Qwen3-VL-32B). Total 29 steps in ~33 min.
