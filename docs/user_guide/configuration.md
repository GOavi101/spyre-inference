# Configuration

## Plugin Setup

To load the plugin, set the `VLLM_PLUGINS` environment variable before running vLLM:

```bash
export VLLM_PLUGINS=spyre_inference,spyre_inference_ops
```

`spyre_inference` activates the platform, and `spyre_inference_ops` registers the OOT
custom ops plus the Spyre Transformers backend (used for `model_impl="transformers"`).

## Usage

You can then use vLLM as usual:

```python
from vllm import LLM

llm = LLM(
    model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    max_model_len=128,
    max_num_seqs=2,
)
```

See the [Examples](../examples/offline_inference/torch_spyre_inference.md) page for more usage patterns.

## Encoder / pooling compile buckets

Spyre compile is on by default (`STOCK_TORCH_COMPILE`, `dynamic=False`). Pass
`--enforce-eager` to disable it. Body and attention are bucketed independently:

- **Body** (Linear / LN): pad the packed token count to the next 1D
  `compile_sizes` bucket `T` (same dispatch as the decoder).
- **Attention** (SDPA): gather into a dense `(B, L)` grid. This is the Spyre
  workaround until flash-style attention lands; the body is not rewritten to
  `T = B × L`.

`compile_sizes` for pooling is the body `T` ladder (`64, 128, …` up to the
token cap). Attention `L` comes from `--max-model-len` (`64, 128, …`).
Attention `B` is powers of two up to `--max-num-seqs` (same as decoder).

A 3-seq × 30-token request with `--max-num-seqs 4` pads the body to `T=128`
and attention to `(B=4, L=64)`. Masks and pooling still use the real lengths.

Compiled pooling warmup dummies 1D body sizes, then each attention `(B, L)`
at full size. Eager pooling uses one short dummy, then runtime still
1D-pads the body.

Example:

```bash
vllm serve ibm-granite/granite-embedding-125m-english \
  --runner pooling --max-num-seqs 4 --max-model-len 512
```

Runtime attention picks the **smallest warmed** `(B, L)` that fits. A 64-token
single-seq request on `--max-model-len 128` lands on `(1, 64)` when that cell
is warmed — not `(1, 128)`. Attention FLOPs still scale as `L²`, so a fair
latency A/B vs sendnn on 64-token prompts is:

```bash
vllm serve ibm-granite/granite-embedding-125m-english \
  --runner pooling --max-num-seqs 1 --max-model-len 64
```

Compare mean E2EL to the same bench with `--max-model-len 128`. If the two
are close, pad-up is not the gap; the remaining cost is per-layer pack/SDPA
and per-block launches.

At `B=1` with `T == L` (no pad slots), encoder attention skips the per-layer
`index_select` pack/unpack and only permutes. `B > 1` still gathers.

Default compile is one transformer block at a time. For a 12-layer embed
model you can A/B a single whole-model graph (do not change this for 40-layer
decoders):

```bash
SPYRE_COMPILE_GRANULARITY=model vllm serve ibm-granite/granite-embedding-125m-english \
  --runner pooling --max-num-seqs 1 --max-model-len 64
```

## pyproject.toml Reference

The `pyproject.toml` includes several key build configurations:

### Build Configuration

```toml
[tool.uv]
build-constraint-dependencies = ["torch==2.13.0"]
extra-build-variables = { vllm = { VLLM_TARGET_DEVICE = "empty", CMAKE_ARGS = "--fresh" } }
```

These settings ensure:

- All packages are built with the same PyTorch version (2.13.0)
- vLLM is built with the **empty** backend — no device-specific C kernels. This avoids
  the torch-version coupling of prebuilt CPU wheels and the dependency on `vllm._C`
  (whose CPU-optimized ops we don't need; Spyre provides its own)

### Source Repositories

The plugin pulls dependencies from specific Git repositories:

```toml
[tool.uv.sources]
vllm = { git = "https://github.com/vllm-project/vllm", rev = "..." }
torch-spyre = { git = "https://github.com/torch-spyre/torch-spyre", rev = "..." }
```

This ensures that torch-spyre and vllm are compiled/installed from source, instead of pulling pre-compiled wheels from PyPI.

### PyTorch CPU Index

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

This ensures the CPU flavor of PyTorch is installed, as CUDA support is not required.
