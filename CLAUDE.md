# Claude Code Notes

Read `AGENTS.md` first. Keep changes small, importable, tested, and reproducible.

## User-Directed Research Convergence

The research direction comes from the user's discussion with ChatGPT, captured
for execution in `docs/cc_reachability_proxy_convergence_handoff_20260814.md`.
Treat that user decision—not the current Claude Code implementation, old
handoffs, or sunk engineering work—as the source of research scope.
The execution brief is self-contained; do not depend on access to the original
ChatGPT conversation.

Claude Code's role is to verify evidence, implement the smallest offline
measurement path, run the registered proxy study, and report results.  Do not
reopen the STP-versus-proxy decision or use existing code to broaden the task.

Effective immediately:

- pause the STP canary, four-arm mechanics pilot, and all training launches;
- preserve the implementation at `489d61a` and do not delete or unwind it;
- run only the offline reachability-proxy work specified by the user brief;
- do not add a loss or start training until the proxy gate is reported and the
  user explicitly approves the next experiment.

Default verification:

```bash
pytest -q
```

Do not commit `.env`, datasets, raw JSONL outputs, checkpoints, model weights, caches, or experiment output directories.

## FC-OPD Mainline Guardrail

Do not speed up the main FC-OPD path by removing the online student scorer. The main algorithm relies on teacher-vs-student condition sensitivity to identify student capability deficits. Teacher-only routing is an ablation only.

## HPC Post-Run Keepalive

If asked to keep an H200 instance alive after a successful full FC-OPD run, add the keepalive as an opt-in post-success step near the end of `scripts/hpc/run_verl_fc_opd_overnight.sh`, after Ray and teacher cleanup and before `exit ${VERL_EXIT}`:

```bash
CUDA_VISIBLE_DEVICES=3,4 KEEPALIVE_TARGET_UTIL=0.45 KEEPALIVE_WORK_ITERS=32 python -u /inspire/hdd/global_user/mengweicheng-240108120092/lzy/scripts/busy_keepalive.py
```

Prefer a flag or env var such as `--keepalive-after-success`; do not start keepalive after failed or interrupted runs unless explicitly requested.

## GPU Instance — vLLM FlashInfer JIT Runtime Requirements

The Vision-OPD-4B checkpoint (`model_type: qwen3_5`) uses Qwen3-Next GDN (Gated Delta Net) attention, which triggers flashinfer SM90 GDN prefill kernel JIT compilation at vLLM inference time. The GPU instance has no system CUDA toolkit (`/usr/local/cuda` absent), so all vLLM servers serving the Vision-OPD checkpoint must declare the managed CUDA toolchain before startup:

```bash
export CUDA_HOME=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/cuda128-toolchain
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
```

| Variable | Why needed | Who needs it |
|---|---|---|
| `CUDA_HOME` | flashinfer `get_cuda_path()` reads this first to locate nvcc + headers + linker `-L` dirs | Vision-OPD vLLM (qwen3_5 → GDN attention) |
| `PATH` | Belt-and-suspenders: flashinfer also tries `which nvcc` as fallback | Same |
| `LIBRARY_PATH` | **Critical**: conda GCC uses sysroot-based linking; `-L` flags from flashinfer JIT are ignored. Linker searches `LIBRARY_PATH` instead to resolve `-lcudart` and `-lcuda`. | Same |
| `LD_LIBRARY_PATH` | Compiled `.so` must find `libcudart.so.*` at dlopen time (libcuda comes from NVIDIA driver) | Same |

**Design rationale**: The cuda128-toolchain was built on the CPU instance (see `scripts/hpc/setup_va_opd_native_env.sh`) for compiling vLLM and flash-attn without `/usr/bin/nvcc`. It uses the **conda CUDA layout** where real libraries live under `targets/x86_64-linux/lib/` with compat symlinks in `lib64/`. The conda GCC 14.3.0 compiler enforces a sysroot model — unlike system GCC, it does not honor `-L` on the command line, so `LIBRARY_PATH` is the only mechanism to inject library search paths at JIT link time.

**Not required for**: Judge server (Qwen3-VL-32B uses standard self-attention, no flashinfer GDN). Adding the vars is harmless.

**FlashInfer cache**: JIT-compiled kernels are cached at `~/.cache/flashinfer/` (GPU-local, not NFS). Clean with `rm -rf ~/.cache/flashinfer/` if a JIT build fails and leaves corrupted artifacts.
