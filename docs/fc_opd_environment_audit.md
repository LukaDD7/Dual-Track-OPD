# Failure-Calibrated OPD Environment Audit

Date: 2026-06-25

## Decision

Use the outer `Dual-Track-OPD` repository as the only source-of-truth checkout.
Pin official `verl` as a Git submodule at:

- tag: `v0.7.1`
- commit: `bec9ef74768dd201881cd4e54cd0385e87caae27`
- nested `recipe` commit: `6d8d5c129334f40b463dc7e383cf269358352fd6`

Do not keep the only working copy of `verl` on `/Volumes/LQ1000`. The external
disk is better used for replaceable, large local state:

```text
/Volumes/LQ1000/dual-track-opd/
  models/
  datasets/
  hf-cache/
  outputs/
  eval-runs/
```

Point environment variables at those directories while keeping code in the Git
checkout. This gives the Mac and HPC the same commit graph and avoids making the
external disk a hidden dependency.

On a new machine:

```bash
git clone --recurse-submodules git@github.com:LukaDD7/Dual-Track-OPD.git
cd Dual-Track-OPD
git submodule update --init --recursive
```

## Sources inspected

Official `verl v0.7.1`:

- `setup.py`
- `scripts/install_vllm_sglang_mcore.sh`
- `docker/verl0.6-cu128-torch2.8.0-fa2.7.4/Dockerfile.base`
- `docker/verl0.6-cu128-torch2.8.0-fa2.7.4/Dockerfile.vllm011.mcore_gpt-oss`
- `docker/Dockerfile.stable.vllm`
- `.github/workflows/vllm.yml`
- `verl/models/transformers/qwen3_vl.py`

Vision-OPD was audited from a temporary read-only clone:

- repository commit: `c2e345fcab10c806ba83e2ec6e1e246d73e7aba2`
- embedded verl version: `0.7.0.dev`
- `requirements.txt`
- `pyproject.toml`
- `scripts/run_vision_opd.sh`

No dependency was installed or changed during this audit.

## Compatibility matrix

| Component | Official verl v0.7.1 evidence | Vision-OPD environment | Decision |
| --- | --- | --- | --- |
| Python | install script uses a CPython 3.12 FlashAttention wheel; CUDA 12.8 image is Python 3.12 | Python 3.12 instructions | Start with Python 3.12 |
| CUDA | maintained image path uses CUDA 12.8 | CUDA 12.x packages, newer stack | Keep the existing CUDA 12.8 toolchain |
| PyTorch | CUDA 12.8 image pins `2.8.0`; `setup.py` does not pin | `2.10.0` | Start from `2.8.0+cu128`; do not reuse Vision-OPD's environment |
| vLLM | installer pins `0.11.0`; package extra allows `>=0.8.5,<=0.12.0` | `0.18.0` | Pin `0.11.0` for the first official-verl environment |
| Transformers | installer says `>=4.51.0`; CI enforces `<5.0`; Qwen3-VL patch cites 4.57 and imports `transformers.models.qwen3_vl` | `5.5.0` | Use a tested 4.57.x build, not 5.x |
| Ray | `>=2.41.0`, no upper bound | `2.53.0` | Record the resolved version; `2.53.0` is a plausible first pin |
| TensorDict | `>=0.8.0,<=0.10.0,!=0.9.0` | `0.10.0` | Pin `0.10.0` |
| FlashAttention | installer uses `2.8.1` wheel for torch 2.8 / CPython 3.12; CUDA image uses `2.7.4.post1` | not pinned as `flash-attn` in requirements | Prefer the official installer wheel path first |
| FlashInfer | installer pins `0.3.1` | `0.6.6` | Start with `0.3.1`; let vLLM 0.11 compatibility win |
| qwen-vl-utils | installed without a version | `0.0.14` | Pin only after the Qwen3-VL processor probe succeeds |
| Accelerate | installed without a version | `1.12.0` | Record the resolved version |

## Important inconsistencies

### 1. Vision-OPD is a separate dependency universe

Vision-OPD pins:

```text
torch==2.10.0
transformers==5.5.0
vllm==0.18.0
ray==2.53.0
tensordict==0.10.0
flashinfer-python==0.6.6
qwen-vl-utils==0.0.14
```

Official `verl v0.7.1` declares vLLM support only through `0.12.0`. Reusing the
Vision-OPD environment would turn the first experiment into an undocumented
framework port. Create a separate environment, tentatively:

```text
fc-opd-verl071-cu128
```

### 2. Official files are not one coherent lock file

The v0.7.1 installer, older CUDA 12.8 image, current stable image, and CI image
show different package generations. In particular, `Dockerfile.stable.vllm`
contains vLLM 0.17 and PyTorch 2.10, outside the package metadata's declared
vLLM range. Treat it as a moving image recipe, not the v0.7.1 lock.

The first environment must therefore be resolved from the tag-compatible
combination, then frozen in:

```text
artifacts/fc_opd/manifests/environment.lock.txt
artifacts/fc_opd/manifests/pip_freeze.txt
```

Those files should be generated on the HPC environment, not on the Mac.

### 3. Qwen3-VL needs a real model-load gate

`verl v0.7.1` contains Qwen3-VL-specific transformer patches and references the
Transformers 4.57 implementation. The generic installer lower bound
(`>=4.51.0`) is therefore too loose for this project. Before training, load
`Qwen3-VL-4B-Instruct` once with Transformers and once with vLLM.

## Proposed HPC installation order

This is an audited plan, not a command to run automatically:

1. Create a new Python 3.12 environment.
2. Install the CUDA 12.8 PyTorch 2.8 build.
3. Install vLLM 0.11.0 and its compatible FlashInfer.
4. Install a Transformers 4.57.x build and Qwen VLM utilities.
5. Install the remaining pinned runtime dependencies.
6. Install `third_party/verl` with `--no-deps -e`.
7. Install this outer repository with `--no-deps -e`.
8. Run import/version, tokenizer, Transformers model-load, and vLLM model-load
   probes.
9. Freeze the environment and record GPU/driver information.

Do not alter the NVIDIA driver, system CUDA toolkit, or system compiler during
this phase.

## Required pre-training gates

- `git submodule status --recursive` shows the recorded commits.
- `torch.version.cuda` matches the intended CUDA build.
- H200 compute capability reports `(9, 0)`.
- student and teacher tokenizer hashes are identical.
- Qwen3-VL loads through both Transformers and vLLM.
- a four-sample teacher log-prob probe returns finite values.
- the full resolved config and data manifest hash are written outside raw data.
