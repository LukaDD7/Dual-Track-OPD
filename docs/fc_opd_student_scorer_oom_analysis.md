# Student Scorer Batched Forward OOM Analysis

## Problem

With `train_batch_size=8` and `rollout_n=4`, there are 32 samples per step.
The student scorer batches all 32 samples per condition with:
```python
processor(text=[prompt_text] * 32, images=[images] * 32)
```
This crashes with CUDA OOM on the scorer's GPU (H200, 141 GB).

## Root cause

Qwen3-VL's `AutoProcessor` when called with `images=[img_list] * 32` creates
**32 independent copies of vision features** (one per batch item).  Each copy
goes through the vision encoder (ViT-G, ~2B parameters), producing ~288 vision
tokens × 5120 dims = ~3 MB of features per copy.  32 copies = ~96 MB for
vision features alone.

Combined with:
- 4B model weights: ~8 GB
- Batched hidden states [32, ~700 tokens, 2560 dims] = ~143 MB
- Batched logits [32, ~700, 151936 vocab] = ~13.6 GB
- Attention KV cache: ~few GB
- PyTorch allocator overhead: variable

Total memory pressure easily exceeds what's available when the scorer GPU
already hosts other processes (verl vLLM worker).

## Why teacher batching works but student doesn't

| | Teacher (32B) | Student (4B) |
|---|---|---|
| GPU | Dedicated (GPU 0) | Shared via Ray (GPU 3) |
| Model size | 66 GB | 8 GB |
| Forward cost per image | High (dominates) | Low |
| Available memory | 141 GB (dedicated) | ~70 GB (shared with vLLM) |

The teacher has a dedicated GPU.  The student scorer runs as a Ray actor
on the same GPU pool as verl training + vLLM.  Ray assigns it a GPU, but
that GPU may already have vLLM workers or FSDP shards.

## Solution options

### Option A: Give student scorer a dedicated GPU (recommended)

Start Ray with only TRAIN_GPUS (GPUs 1,2) and run the student scorer
as a standalone process on GPU 3 with `CUDA_VISIBLE_DEVICES=3`.

Requires: modifying `_build_student_scorer` to launch a subprocess
instead of a Ray actor when a dedicated GPU index is specified.

### Option B: Reduce sub-batch size dynamically

Instead of B=32, split into sub-batches of B=4 or B=8.
Already implemented with `_STUDENT_BATCH_MAX = 8`.
4 sub-batches × 6 conditions = 24 batched forwards, ~24 × 2.4s = 58s.

### Option C: Share vision features across batch items

Process images ONCE through vision encoder, expand vision features to
[B, num_vis_tokens, hidden], then do a single batched LLM forward.
Requires model internals access (intercept vision features before LLM).

## Current state

Option B is implemented but fc_opd timing is still ~197s (step timing shows
student path may have fallen back to per-sample).  Need to verify the
sub-batching code path is actually executing and not silently falling back.

## What we need

1. Verify sub-batching is active (add timing logs around student scorer calls)
2. If not active: fix the fallback path
3. If active but still slow: implement Option A (dedicated GPU subprocess)

## Expected performance after fix

- Teacher: 6 batched forwards of B=32 → ~25s
- Student: 24 batched forwards of B=8 (4 sub × 6 cond) → ~30s
- CPU compute (sparse_forward_kl loops) → ~60s (still bottleneck, needs vectorization separately)
- Other overhead: ~10s
- **Total fc_opd: ~125s, total step: ~140s**
- 1313 steps × 140s = **~51 hours** (still long, CPU compute needs fixing next)
