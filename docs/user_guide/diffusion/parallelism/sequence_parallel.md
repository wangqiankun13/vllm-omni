# Sequence Parallelism Guide

## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

Sequence parallelism splits the input along the sequence dimension across multiple GPUs, allowing each device to process only a portion of the sequence. vLLM-Omni provides 1.5x-3.6x speedup for large images and videos using DeepSpeed Ulysses, Ring-Attention, or hybrid approaches. Use sequence parallelism when generating high-resolution images/videos that don't fit on a single GPU or require faster inference.

See supported models list in [Diffusion Features - Supported Models](../../diffusion_features.md#supported-models).

**Supported Methods:**

- **DeepSpeed Ulysses Sequence Parallel (Ulysses-SP)** ([paper](https://arxiv.org/pdf/2309.14509)): Uses all-to-all communication for subset of attention heads per device
- **Ring-Attention** ([paper](https://arxiv.org/abs/2310.01889)): Uses ring-based P2P communication with sharded sequence dimension throughout
- **Hybrid Ulysses + Ring**: Combines both for larger scale parallelism (`ulysses_degree × ring_degree`)
- **Ulysses × AllGather-KV**: A two-dimensional topology where Ulysses shards attention heads and AllGather-KV makes K/V global (`ulysses_degree × allgather_degree`)

---

## Quick Start

### Basic Usage - Ulysses-SP

Simplest working example with Ulysses Sequence Parallel:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2)  # Enable Ulysses-SP
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50, width=1024, height=1024),
)
```

!!! note "Experimental UAA mode"
    `ulysses_mode="advanced_uaa"` is an experimental extension to Ulysses-SP. It lets Ulysses attention handle arbitrary sequence lengths and arbitrary attention head counts without relying on `attention_mask`-based token padding.

    In hybrid Ulysses + Ring mode, Ring still requires every rank in the same ring group to observe the same post-Ulysses sequence length. If that condition is not met, vLLM-Omni raises a validation error instead of entering the ring kernel with inconsistent shapes.

To enable the experimental UAA mode, use a model/configuration that requires it. For example, `Tongyi-MAI/Z-Image-Turbo` has 30 attention heads, so `ulysses_degree=4` requires UAA because 30 is not divisible by 4:

```python
omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    parallel_config=DiffusionParallelConfig(
        ulysses_degree=4,
        ulysses_mode="advanced_uaa",
    ),
)
```

### Alternative Methods

**Ring-Attention** (better for very long sequences):

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ring_degree=2)  # Enable Ring-Attention
)
```

**Hybrid Ulysses + Ring** (for larger scale):

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2, ring_degree=2)  # 4 GPUs total
)
```

**Ulysses × AllGather-KV** (4 GPUs total, no Ring in the attention computation):

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2, allgather_degree=2)
)
```

### Ulysses × AllGather-KV

This topology reaches larger SP degrees without putting Ring Attention on the
critical path. Each rank starts from `[B, S / (U × A), H, D]` and the forward is:

```text
Ulysses all-to-all (U group):  Q, K, V → [B, S / A,     H / U, D]
K/V AllGather     (A group):   K, V    → [B, S,         H / U, D]
local-Q / global-KV attention: O       → [B, S / A,     H / U, D]
reverse Ulysses all-to-all:    O       → [B, S / (U × A), H,    D]
```

The kernel always sees a local-Q/global-KV problem, so the full global sequence
is materialized in a single, ordered K/V tensor. That matters for attention
backends that need stable global K/V indexing — sparse-mask generation in
particular — which is awkward under Ring Attention because Ring keeps K/V
block-local and owns distributed attention itself.

Choosing between the two 4-GPU options is mostly about the attention backend: if
the backend can consume a global K/V tensor, prefer Ulysses × AllGather-KV; if
the sequence is large enough that materializing global K/V does not fit, use
Ulysses × Ring instead.

Constraints:

- `ring_degree` must be `1` — composing all three SP dimensions is not supported.
- Non-causal attention only.
- Every rank's region must be equally long, so the shared sequence must be evenly
  shardable across the SP group (or the model's `_sp_plan` must use `auto_pad`).
  A mismatch fails fast instead of corrupting the collective.
- 2D key masks are supported: an image mask must cover the gathered *global*
  image keys replicated across SP ranks (the same contract as plain
  AllGather-KV), and a joint text-padding mask (`joint_attn_mask`, e.g.
  Qwen-Image with unequal prompt lengths) is merged automatically in
  `joint_strategy` order. Masks covering only the local shard fail fast.
  `mask_sp_padding=True` builds exactly such a global 2D mask, so strict
  padding masking works under this topology; 4D masks keep query-range slicing.
- Not supported with Scheduler-paged KV or with `diffusion_attention_backend="TRTLLM_ATTN"`.

---

## Example Script

### Offline Inference

Use Python script under `examples/offline_inference/text_to_image/text_to_image.py`:

**Ulysses-SP:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "A cat sitting on a windowsill" \
    --ulysses-degree 2 \
    --width 1024 --height 1024
```

**Ring-Attention:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "A cat sitting on a windowsill" \
    --ring-degree 2 \
    --width 1024 --height 1024
```

**Hybrid Ulysses + Ring:**

```bash
# Hybrid: 2 Ulysses × 2 Ring = 4 GPUs total
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "A cat sitting on a windowsill" \
    --ulysses-degree 2 --ring-degree 2 \
    --width 1024 --height 1024
```

**Ulysses × AllGather-KV:**

```bash
# 2 Ulysses × 2 AllGather-KV = 4 GPUs total, ring_degree stays 1
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "A cat sitting on a windowsill" \
    --ulysses-degree 2 --allgather-degree 2 \
    --width 1024 --height 1024
```

### Online Serving

**Ulysses-SP:**

```bash
# Text-to-image (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2
```

**Ulysses-SP with UAA mode** (for models with non-divisible head counts):

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091 --usp 4 --ulysses-mode advanced_uaa
```

**Ring-Attention:**

```bash
# Text-to-image (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --ring 2
```

**Hybrid Ulysses + Ring:**

```bash
# Text-to-image (requires >= 4 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2 --ring 2
```

**Ulysses × AllGather-KV:**

```bash
# Text-to-image (requires >= 4 GPUs); --ring must stay 1
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2 --allgather-degree 2
```

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `ulysses_degree` | int | 1 | Number of GPUs for Ulysses-SP. Uses all-to-all communication. |
| `ring_degree` | int | 1 | Number of GPUs for Ring-Attention. Uses P2P ring communication. |
| `allgather_degree` | int | 1 | Number of GPUs for AllGather-KV sequence parallelism. Gathers K/V in an orthogonal group so every rank runs local-Q/global-KV attention. May be combined with `ulysses_degree`, but not with `ring_degree > 1`. |
| `ulysses_mode` | str | `"strict"` | Ulysses attention mode. Set to `"advanced_uaa"` to handle arbitrary sequence lengths and head counts without padding. |
| `ulysses_a2a_permute` | bool | `False` | Enable CUDA copy engines (CE) for strict Ulysses; disabled uses NCCL. On NPU, use platform SHMEM. |
| `mask_sp_padding` | bool | `False` | When the sequence length is not divisible by the SP size, tokens are auto-padded with zeros. Set to `True` to mask those padding tokens (strict, but uses the slower varlen attention path); the default `False` leaves them unmasked, keeping the fast path with negligible numerical impact. |

**Notes:**

- Total sequence parallel size equals `ulysses_degree × ring_degree × allgather_degree`
- Degrees must evenly divide the sequence length for optimal performance (or use `ulysses_mode="advanced_uaa"` for Ulysses-SP)
- `mask_sp_padding` is an experimental feature, currently only supported by `Wan2.2`, `Wan2.2 Vace`, `Qwen-Image`, `Flux 2`, and `HunyuanVideo 1.5`

On CUDA, set `ulysses_a2a_permute=True` (CLI: `--ulysses-a2a-permute`) to use CE.
This requires `ulysses_mode="strict"`, `ulysses_degree > 1`, and peer-accessible
GPUs on one NVLink-connected node.

```bash
vllm serve Qwen/Qwen-Image --omni --ulysses-degree 2 --ulysses-a2a-permute
```

On NPU, the flag retains the existing
[NPU symmetric-memory requirements](../../../design/feature/sequence_parallel.md#npu-symmetric-memory-ulysses).

### Overlap strict Ulysses CE with attention (experimental)

Qwen-Image-2.1 and MiniMax-H3 T2VA can pipeline Q/K/V transfers, attention,
and output transfers across groups of heads. Omit `--ulysses-head-split` to
use the existing attention path. The flag accepts two forms.

| Setting | Behavior |
| --- | --- |
| A ratio such as `3:5` or `1:12:1` | Split the heads assigned to **each rank** into two or three groups in that exact ratio. |
| `auto` | Time candidates on successive first-step layers for each request. The original whole-head CE implementation participates. |

For example, add `--ulysses-degree 4 --ulysses-a2a-permute --enforce-eager --ulysses-head-split 3:5`
to a Qwen-Image-2.1 invocation. With 32 heads this gives 3 and 5 heads per rank.
At SP2 the same ratio gives 6 and 10. For a 56-head H3 model at SP4,
`5:9` or `1:12:1` consumes the 14 heads per rank. Ratios requiring
fractional heads raise an error. The Python and stage YAML field is
`ulysses_head_split` under `parallel_config`.

Each automatic trial uses four layers. The first warms the route. The score
is the median of the three remaining attention-call times, each reduced by
maximum over the Ulysses group. Trials preserve native outputs. A candidate
must match all sampled native outputs on every rank and beat native by at
least 1% to displace it. Qwen keeps its block-causal prefill and uses its
real target rows to probe the decode shape. Cold graph capture, probing,
and accuracy checks contribute to request latency. On the 56-head H3 model
at SP4, automatic selection tests native, 7+7, 1+12+1, and 5+9. Each H3
candidate and its native control are timed around graph replay after staging
the same packed inputs. The full request pays for staging and comparison.

Both modes require peer-accessible GPUs on one node, CUDA BF16 FlashAttention-3,
pure strict CE Ulysses, and one request per forward. Qwen supports SP2 and
SP4. MiniMax-H3 requires SP4 and `task_type=t2va`. Qwen supports unpadded, block-causal prefix prefill
and cached-prefix decode. Both split modes accept H3 dense attention up to
8,576 global packed rows (including padding, summed over SP ranks). A tested
512×512, 107-frame request fits this limit. A 1344×768, 124-frame request
exceeds it. Use the original attention path for longer H3 requests.

Set `enforce_eager=True` on the outer model. After the first step, the selected
route replays the DiT block stack in one CUDA Graph per step. Qwen refreshes
the static prefix KV for each request. H3 stages exact AdaLN projections for each time embedding as graph inputs,
so the same graph can replay at different timesteps. A new geometry or route can incur
warm-up and capture latency. The graph requires resident, stationary DiT
weights. Disable DiT weight offload, sleep, LoRA, weight transfer, step execution,
outer denoise graphs, attention schedules and transformer cache acceleration. SP8+, Ring, GQA, TP, HSDP,
and masked or reference-KV H3 attention are unsupported.

To measure a fixed workload through `Omni.generate`, supply a JSON object
with `engine` (Omni engine kwargs), `prompt` (including modalities and any
reference image), and `sampling` (shape, steps and guidance). Run the native
arm first. Use the same output directory for the other arms to compare RGB
pixels and H3 audio against the corresponding seed.

```bash
for policy in native 3:5 auto; do
  python -m benchmarks.diffusion.bench_ulysses_head_split --workload qwen.json \
    --output results/qwen --policy "$policy" --rounds 9
done
```

## Best Practices

### When to Use

**Good for:**

- Large images (1024x1024 or higher) or videos
- Fast inter-GPU communication, larger bandwidth (e.g., NVLink)

**Not for:**

- Small images (<1024px) - overhead exceeds benefit, use single GPU with cache instead

---

## Compiling FA4 Ring attention

On Blackwell, Ring prefers vLLM's bundled FlashAttention dispatcher. When it
selects FA4, compiled Ring uses the bundled FA4 kernel. Ring can also use
standalone CuTe FA4 when the bundled dispatcher is unavailable.

With `torch.compile` enabled and PyTorch 2.14 or newer, set the following before
starting workers to enable experimental native P2P compilation:

```bash
export TORCHDYNAMO_ENABLE_P2P_COMPILATION=1
```

## Troubleshooting

### Common Issue 1: Performance Not Scaling

**Symptoms**: Adding GPUs doesn't improve speed proportionally, or higher parallelism degree is slower

**Diagnosis:**

```bash
# Check GPU topology
nvidia-smi topo -m

```

**Solutions:**

1. Check inter-GPU communication - NVLink is better than PCIe
2. Reduce parallelism degree if over-parallelized:

```python
# If 4 GPUs is slower than 2
parallel_config=DiffusionParallelConfig(ulysses_degree=2)
```

1. Try to switch between Ring-Attention and Ulysses-SP

- Ring-Attention has advantages, like communication-computation overlap, but the block-wise loop overhead is relatively higher, especially for short sequences
- Ulysses-SP: can benefit from larger bandwidth (such as NVLink), with two major constraints, the sequence length should be divisible by usp size, and the number of heads should be divisible by usp size (or use `ulysses_mode="advanced_uaa"`)

### Common Issue 2: Out of Memory (OOM)

**Symptoms**: CUDA OOM errors or process crashes with memory errors

**Solutions:**

1. Increase parallelism degree to split sequence more:

```python
parallel_config=DiffusionParallelConfig(ulysses_degree=4)  # From 2
```

1. Combine with other parallelism method, e.g., tensor parallel, and memory optimization methods, e.g., cpu offloading.

## Summary

1. ✅ **Enable Sequence Parallelism** - Set `ulysses_degree` or `ring_degree` for long sequence generation
2. ✅ **UAA mode** - Use `ulysses_mode="advanced_uaa"` when head count is not divisible by `ulysses_degree`
3. ✅ **Troubleshooting** - Check GPU topology with `nvidia-smi topo -m`, reduce degree if performance doesn't scale
