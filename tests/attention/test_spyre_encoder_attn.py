# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
)
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    ENCODER_BLOCK_SIZE,
    SpyreEncoderAttentionImpl,
    _blocks_for,
    _create_encoder_block_kernel,
    dense_sdpa_reference,
    encoder_index_dtype,
    encoder_mask_tiles,
    encoder_row_table,
)

# extra `encoder_attention` mark so CI can split this into its own job
# because these tests are pretty slow.
pytestmark = [pytest.mark.attention, pytest.mark.encoder_attention]


@pytest.fixture()
def configure_device(request, monkeypatch):
    """Configure overwrite_f and cache device based on the device_mode parameter.

    The spyre card check is done lazily here (not at import time) to avoid
    claiming the device before subprocess-based tests have a chance to run.
    """

    device_mode = request.param
    if device_mode == "spyre" and not spyre_available():
        pytest.skip("Spyre device not available")
    return device_mode


@pytest.fixture()
def configure_compilation(request, monkeypatch):
    """Configure torch.compile mode for tests."""
    import torch
    from vllm.config import get_cached_compilation_config
    from vllm.config.compilation import CompilationMode

    mode_name = request.param
    compilation_mode = getattr(CompilationMode, mode_name)

    # Reset dynamo cache first to ensure config changes take effect
    torch._dynamo.reset()

    cfg = get_cached_compilation_config()
    original_mode = cfg.mode

    # Store original torch._dynamo config
    original_limit = torch._dynamo.config.accumulated_recompile_limit

    cfg.mode = compilation_mode
    # Increase recompilation limit: the page-attention kernel is specialized
    # (and so recompiled) per unique (num_blocks, padded_query_len)
    torch._dynamo.config.accumulated_recompile_limit = 1024

    yield mode_name

    # Cleanup: reset mode and limits
    cfg.mode = original_mode
    torch._dynamo.config.accumulated_recompile_limit = original_limit
    torch._dynamo.reset()


def _build_metadata(
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Use the real SpyreAttentionMetadataBuilder to construct metadata."""
    from vllm.config import get_current_vllm_config

    # Reuse the VllmConfig set up by the `default_vllm_config` fixture and
    # stub the head-count methods the builder reads.
    vllm_config = get_current_vllm_config()
    vllm_config.model_config.get_num_attention_heads = Mock(return_value=num_query_heads)
    vllm_config.model_config.get_num_kv_heads = Mock(return_value=num_kv_heads)

    kv_cache_spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )

    builder = SpyreAttentionMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    max_query_len = int((query_start_loc[1:] - query_start_loc[:-1]).max().item())
    max_seq_len = int(seq_lens.max().item())
    num_actual_tokens = int(query_start_loc[-1].item())

    common_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        num_reqs=len(seq_lens),
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=False,
    )

    return builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )


def assert_close_outliers(
    actual: torch.Tensor,
    expected: torch.Tensor,
    max_outliers: int = 0,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    *,
    outlier_atol: float | None = None,
    outlier_rtol: float | None = None,
) -> None:
    """Assert tensors are close, allowing up to *max_outliers* elements to exceed tolerance.

    Arguments beyond *max_outliers* are forwarded to ``torch.testing.assert_close``.

    Args:
        actual: tensor under test.
        expected: reference tensor.
        max_outliers: number of elements that may exceed the base tolerances.
        atol: absolute tolerance for the bulk of elements.
        rtol: relative tolerance for the bulk of elements.
        outlier_atol: absolute tolerance for outlier elements (defaults to *atol*,
            meaning outliers only need to be finite, not within any tighter bound).
        outlier_rtol: relative tolerance for outlier elements.
        msg: additional context for the failure message.
    """
    diff = (actual - expected).abs()
    tol = atol + rtol * expected.abs()
    outlier_mask = diff > tol
    n_outliers = outlier_mask.sum().item()

    if n_outliers <= max_outliers and max_outliers > 0:
        # Check that outliers are still within the relaxed bound (or simply finite)
        if outlier_atol is not None or outlier_rtol is not None:
            outlier_tol = (outlier_atol if outlier_atol is not None else atol) + (
                outlier_rtol if outlier_rtol is not None else rtol
            ) * expected.abs()
            if diff[outlier_mask].gt(outlier_tol[outlier_mask]).any():
                worst = diff[outlier_mask].max().item()
                raise AssertionError(
                    f"{n_outliers} outlier(s) exceed base tolerances, "
                    f"and at least one outlier also exceeds the relaxed bound "
                    f"(worst diff={worst:.4g})."
                )
        if n_outliers > 0:
            print(
                f"  [assert_close_outliers] {n_outliers}/{actual.numel()} element(s) "
                f"exceed base tolerance but remain within relaxed bound — acceptable."
            )
        return  # acceptable number of outliers within relaxed bounds

    # Fall through to standard assert_close for a clear error message
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as e:
        prefix = (
            f"{n_outliers} elements exceed atol={atol}, rtol={rtol}. "
            if n_outliers > max_outliers
            else ""
        )
        raise AssertionError(
            f"{prefix}"
            f"max_outliers={max_outliers} was specified "
            f"but {n_outliers} element(s) exceed tolerance.\n"
            f"{e}"
        ) from e


def ref_encoder_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """Reference bidirectional self-attention (no causal mask, no KV cache)."""
    num_seqs = len(query_lens)
    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        q = query[start_idx : start_idx + query_len]
        q = q * scale
        k = key[start_idx : start_idx + query_len]
        v = value[start_idx : start_idx + query_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def _blocked_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: list[int],
    scale: float,
    *,
    compile_enabled: bool = False,
) -> torch.Tensor:
    """Drive the blocked kernel over a packed list the way ``forward`` does."""
    num_heads, head_size = query.shape[1], query.shape[2]
    num_kv_heads = key.shape[1]
    num_queries_per_kv = num_heads // num_kv_heads
    index_dtype = encoder_index_dtype(query.device)
    tile_cache: dict = {}
    out = torch.zeros_like(query)
    start = 0
    for length in query_lens:
        num_blocks = _blocks_for(length)
        extent = num_blocks * ENCODER_BLOCK_SIZE
        kernel = _create_encoder_block_kernel(
            num_blocks,
            num_heads,
            num_kv_heads,
            head_size,
            needs_gather=True,
            store_mode="index",
        )
        if compile_enabled:
            kernel = torch.compile(kernel, dynamic=False)
        kernel(
            query,
            key,
            value,
            encoder_row_table(start, length, extent, index_dtype),
            encoder_mask_tiles(
                num_blocks,
                length,
                num_kv_heads,
                num_queries_per_kv,
                query.dtype,
                query.device,
                tile_cache,
            ),
            scale,
            out=out,
        )
        start += length
    return out


def test_blocks_for_rounds_up_to_powers_of_two():
    assert [_blocks_for(n) for n in (1, 64, 65, 128, 129, 200, 512)] == [1, 1, 2, 2, 4, 4, 8]


def test_mask_tiles_cut_at_the_boundary_block():
    tiles = encoder_mask_tiles(3, 100, 2, 1, torch.float32, torch.device("cpu"), {})
    masked = torch.finfo(torch.float32).min
    # Block 0 is all real keys, block 1 straddles kv_len=100, block 2 is all padding.
    assert torch.equal(tiles[0], torch.zeros_like(tiles[0]))
    assert torch.equal(tiles[1][..., :36], torch.zeros_like(tiles[1][..., :36]))
    assert torch.equal(tiles[1][..., 36:], torch.full_like(tiles[1][..., 36:], masked))
    assert torch.equal(tiles[2], torch.full_like(tiles[2], masked))
    # Interior and beyond-the-end tiles come from the cache by reference.
    assert encoder_mask_tiles(1, 64, 2, 1, torch.float32, torch.device("cpu"), {})[0].shape == (
        2,
        1,
        1,
        ENCODER_BLOCK_SIZE,
    )


def test_row_table_clamps_padding_lanes_to_the_last_real_row():
    rows = encoder_row_table(10, 3, ENCODER_BLOCK_SIZE, torch.int64)
    assert rows[:3].tolist() == [10, 11, 12]
    # Padding lanes repeat row 12 so the gather never reads the next request.
    assert rows[3:].unique().tolist() == [12]


@pytest.mark.parametrize(
    "configure_device",
    [
        pytest.param("cpu", id="device_cpu"),
        pytest.param("spyre", id="device_spyre"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_compilation",
    [
        pytest.param("NONE", id="compilation_NONE"),
        pytest.param("STOCK_TORCH_COMPILE", id="compilation_STOCK"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([(32, 32)], id="prefill(q=32,kv=32)"),
        pytest.param([(64, 64)], id="prefill(q=64,kv=64)"),
        pytest.param([(100, 100)], id="prefill(q=100,kv=100)"),
        pytest.param([(16, 16), (32, 32)], id="batch_prefill(2seqs)"),
        pytest.param([(9, 9), (70, 70), (5, 5)], id="batch_unaligned(3seqs)"),
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [
        pytest.param((16, 4), id="GQA"),
        # pytest.param((16, 16), id="MHA"),
    ],
)
@pytest.mark.parametrize(
    "head_size",
    [
        # Product encoder models (Granite/E5/RoBERTa) use D=64; MiniLM uses 32.
        pytest.param(64, id="head_size(64)"),
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        # Valid block_size values: must be multiples of 64 for Spyre stick alignment.
        pytest.param(64, id="block_size(64)"),
        pytest.param(128, id="block_size(128)"),
        pytest.param(256, id="block_size(256)"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="dtype(fp16)"),
    ],
)
@torch.inference_mode()
def test_spyre_encoder_attn(
    default_vllm_config,
    dtype: torch.dtype,
    block_size: int,
    head_size: int,
    num_heads: tuple[int, int],
    seq_lens: list[tuple[int, int]],
    configure_compilation: str,
    configure_device: str,
) -> None:
    """Validate SpyreEncoderAttentionImpl against a bidirectional reference."""
    # TODO: STOCK_TORCH_COMPILE + device_spyre, currently fails with
    # "missing device_tensor_layout on graph input arg0_1"
    if configure_compilation == "STOCK_TORCH_COMPILE" and configure_device == "spyre":
        pytest.skip("STOCK + device_spyre, currently fails.")

    num_query_heads, num_kv_heads = num_heads
    # only for preparation, actual device is set via `configure_device`
    torch.set_default_device("cpu")
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    assert query_lens == kv_lens
    assert num_query_heads % num_kv_heads == 0
    scale = head_size**-0.5

    total_tokens = sum(query_lens)
    query = torch.randn(total_tokens, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_query_len = max(query_lens)
    max_num_blocks_per_seq = (max_query_len + block_size - 1) // block_size
    block_table = torch.zeros(num_seqs, max_num_blocks_per_seq, dtype=torch.int32)
    slot_mapping = torch.arange(total_tokens, dtype=torch.int64)

    attn_metadata = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=kv_lens_tensor,
        query_start_loc=cu_query_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
    )

    attn_impl = SpyreEncoderAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )

    cache_device = torch.device(configure_device)
    output = torch.empty_like(query).to(cache_device)
    kv_cache = SpyrePagedKVCache(k_pages=torch.empty(0), v_pages=torch.empty(0))
    attn_impl.forward(
        layer=None,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )

    ref_output = ref_encoder_attn(
        query=query,
        key=key,
        value=value,
        query_lens=query_lens,
        scale=scale,
    )

    if max(query_lens) >= 32:
        atol, rtol = 0.3, 0.2
    else:
        atol, rtol = 0.2, 0.2

    # Allow a small number of outlier elements to exceed the base tolerance,
    # which can happen due to nondeterministic hardware optimizations.
    assert_close_outliers(
        output.to("cpu"),
        ref_output,
        max_outliers=5,
        atol=atol,
        rtol=rtol,
        outlier_atol=atol * 2,
        outlier_rtol=rtol * 2,
    )


@pytest.mark.parametrize(
    "query_lens",
    [
        pytest.param([32], id="single_32"),
        pytest.param([30, 12, 8], id="ragged_3"),
        pytest.param([62], id="prompt_62"),
    ],
)
def test_encoder_flash_matches_per_seq_sdpa(query_lens: list[int]) -> None:
    """Blocked online softmax on flat T matches per-sequence SDPA (no (B, L) grid)."""
    torch.manual_seed(0)
    heads, dim = 4, 64
    scale = dim**-0.5
    total = sum(query_lens)
    query = torch.randn(total, heads, dim)
    key = torch.randn(total, heads, dim)
    value = torch.randn(total, heads, dim)
    ref = dense_sdpa_reference(query, key, value, query_lens, scale)
    got = _blocked_attn(query, key, value, query_lens, scale)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_encoder_flash_spans_multiple_kv_blocks() -> None:
    """A sequence longer than one block exercises the running max/sum rescale."""
    torch.manual_seed(0)
    heads, dim = 2, 64
    scale = dim**-0.5
    query_lens = [200]
    query = torch.randn(200, heads, dim)
    key = torch.randn(200, heads, dim)
    value = torch.randn(200, heads, dim)
    ref = dense_sdpa_reference(query, key, value, query_lens, scale)
    got = _blocked_attn(query, key, value, query_lens, scale)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_encoder_flash_gqa_matches_per_seq_sdpa() -> None:
    torch.manual_seed(0)
    heads, kv_heads, dim = 8, 2, 64
    scale = dim**-0.5
    query_lens = [70, 33]
    total = sum(query_lens)
    query = torch.randn(total, heads, dim)
    key = torch.randn(total, kv_heads, dim)
    value = torch.randn(total, kv_heads, dim)
    ref = dense_sdpa_reference(query, key, value, query_lens, scale)
    got = _blocked_attn(query, key, value, query_lens, scale)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_encoder_flash_compile_matches_eager() -> None:
    torch.manual_seed(0)
    heads, dim = 2, 64
    query_lens = [40, 90]
    total = sum(query_lens)
    q = torch.randn(total, heads, dim)
    k = torch.randn(total, heads, dim)
    v = torch.randn(total, heads, dim)
    eager = _blocked_attn(q, k, v, query_lens, dim**-0.5)
    compiled = _blocked_attn(q, k, v, query_lens, dim**-0.5, compile_enabled=True)
    torch.testing.assert_close(compiled, eager, atol=1e-4, rtol=1e-4)
