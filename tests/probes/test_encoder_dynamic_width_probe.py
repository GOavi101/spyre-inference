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

"""Probe for #1525: one encoder layer graph per length, with the batch width declared.

The rows of the encoder body ``[B * L, H]`` are reserved on dimension 0 with
granularity ``L``, and the whole layer runs as the body of a ``for_each_tile`` loop
with one sequence per tile. torch-spyre refuses a plain op over a declared
dimension, so the linears and layer norms sit inside the loop with attention, and
the weights are untiled operands.

What it answers before the bucketer, upload and warmup are changed:

* the layer compiles once and is right at every width in the range, against CPU;
* how long that compile takes next to today's static rectangle graph at the
  maximum width, printed with ``-s``.

Every sequence is full length, so there is no key-padding mask. ``F.layer_norm``
inside the loop is a strict xfail; the ``arith`` variant runs the rest of the layer.
"""

import time

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available

pytestmark = pytest.mark.probe

# (length, max width, hidden, heads, intermediate). The second is a Granite
# Embedding 125m layer on its 2048-row body at max_model_len 512.
CASES = {
    "small": (64, 8, 256, 4, 1024),
    "granite-125m-512": (512, 4, 768, 12, 3072),
}

# fp16 on device against an fp32 reference, after two layer norms.
ATOL = 5e-2
RTOL = 5e-2


@pytest.fixture()
def spyre_device():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


def _params(hidden, intermediate, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def w(*shape):
        return (torch.randn(*shape, generator=gen) * 0.05).to(dtype)

    ones, zeros = torch.ones(hidden, dtype=dtype), torch.zeros(hidden, dtype=dtype)
    return (
        w(hidden, hidden), w(hidden),  # q
        w(hidden, hidden), w(hidden),  # k
        w(hidden, hidden), w(hidden),  # v
        w(hidden, hidden), w(hidden),  # attention output
        ones, zeros,  # attention layer norm
        w(intermediate, hidden), w(intermediate),  # up
        w(hidden, intermediate), w(hidden),  # down
        ones.clone(), zeros.clone(),  # output layer norm
    )  # fmt: skip


def _arith_layer_norm(x, normalized_shape, weight, bias, eps=1e-5):
    """Layer norm as mean/var/rsqrt, which never lowers to torch-spyre's EXX2 ops.

    Inside a ``for_each_tile`` body ``validate_ops`` names the ``layernormnorm`` op
    ``while_loop``, so its EXX2 exemption misses it and ``F.layer_norm`` is refused.
    """
    from spyre_inference.custom_ops.layer_norm import _layer_norm_kernel

    return _layer_norm_kernel(x, weight, bias, eps)


def _layer(x, params, num_heads, layer_norm=F.layer_norm, mask=None):
    """One post-norm BERT layer over ``x`` of ``[..., L, H]``, ``mask`` the additive key pad."""
    wq, bq, wk, bk, wv, bv, wo, bo, ln1_w, ln1_b, w1, b1, w2, b2, ln2_w, ln2_b = params
    *lead, length, hidden = x.shape
    head_size = hidden // num_heads

    def heads(t):
        return t.reshape(*lead, length, num_heads, head_size).transpose(-3, -2)

    q, k, v = heads(F.linear(x, wq, bq)), heads(F.linear(x, wk, bk)), heads(F.linear(x, wv, bv))
    # SDPA, as `_encoder_sdpa_kernel` does. A hand-written q @ k^T over these head
    # views has no layout inside a for_each_tile body ("no mechanism to resolve stick
    # incompatibility").
    context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=head_size**-0.5)
    context = context.transpose(-3, -2).reshape(*lead, length, hidden)
    h = layer_norm(x + F.linear(context, wo, bo), (hidden,), ln1_w, ln1_b)
    ff = F.linear(F.gelu(F.linear(h, w1, b1)), w2, b2)
    return layer_norm(h + ff, (hidden,), ln2_w, ln2_b)


def _tiled_layer(length, max_width, num_heads, layer_norm):
    from torch_spyre._inductor.wsr import for_each_tile

    def fn(rows, *params):
        n = rows.shape[0]
        torch._check(n >= length)
        torch._check(n <= max_width * length)
        torch._check(n % length == 0)

        def body(_carry, tiles):
            tile, *weights = tiles
            return None, _layer(tile, weights, num_heads, layer_norm)

        _carry, out = for_each_tile(
            body,
            (rows, *params),
            dims=(0,) + (None,) * len(params),
            tile_size=length,
            out_dim=0,
        )
        return out

    return fn


def _static_layer(length, num_heads, layer_norm):
    def fn(rows, *params):
        grid = rows.reshape(rows.shape[0] // length, length, rows.shape[1])
        return _layer(grid, params, num_heads, layer_norm).reshape(rows.shape)

    return fn


def _median_ms(fn, *args, reps=50, warmup=5):
    """Steady-state milliseconds per call, the output read back to host each time."""
    for _ in range(warmup):
        fn(*args).cpu()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn(*args).cpu()
        times.append((time.perf_counter() - start) * 1e3)
    return sorted(times)[reps // 2]


def _widths(max_width):
    """The one-tile minimum first, then the maximum, then the widths between."""
    return list(dict.fromkeys([1, max_width, *range(2, max_width)]))


NORMS = {"arith": _arith_layer_norm, "aten": F.layer_norm}


@pytest.mark.parametrize(
    "norm",
    [
        "arith",
        pytest.param(
            "aten",
            marks=pytest.mark.xfail(
                strict=True,
                reason="validate_ops refuses layernormnorm's EXX2 operand inside a "
                "for_each_tile body, where the op is named while_loop",
            ),
        ),
    ],
)
@pytest.mark.parametrize("case", sorted(CASES))
def test_encoder_layer_compiles_once_for_every_batch_width(spyre_device, case, norm):
    from torch._dynamo.utils import counters

    layer_norm = NORMS[norm]
    length, max_width, hidden, num_heads, intermediate = CASES[case]
    dtype = torch.float16
    params_cpu = _params(hidden, intermediate, dtype)
    params = tuple(p.to(spyre_device) for p in params_cpu)
    x = torch.rand(max_width * length, hidden, dtype=dtype) - 0.5

    def reference(width):
        n = width * length
        return _layer(
            x[:n].float().reshape(width, length, hidden),
            [p.float() for p in params_cpu],
            num_heads,
        ).reshape(n, hidden)

    # The static layer first: if it fails too, the op mix is unsupported on Spyre
    # rather than inside the loop.
    torch._dynamo.reset()
    static = torch.compile(
        _static_layer(length, num_heads, layer_norm), dynamic=False, fullgraph=True
    )
    x_device = x.to(spyre_device)
    start = time.perf_counter()
    out = static(x_device, *params).cpu().float()
    static_s = time.perf_counter() - start
    torch.testing.assert_close(out, reference(max_width), atol=ATOL, rtol=RTOL, msg="static")
    static_ms = _median_ms(static, x_device, *params)
    tag = f"[{case}, {norm} norm]"
    print(
        f"\n{tag} static ({length}, {max_width}) compiled in {static_s:.1f}s, "
        f"runs in {static_ms:.1f}ms"
    )

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(_tiled_layer(length, max_width, num_heads, layer_norm), fullgraph=True)
    spec = {"min": length, "max": max_width * length, "granularity": length}
    rows = x.to(spyre_device, dynamic={0: spec})

    compile_s = None
    for width in _widths(max_width):
        n = width * length
        rows.resize_((n, hidden))
        start = time.perf_counter()
        out = compiled(rows, *params).cpu().float()
        if compile_s is None:
            compile_s = time.perf_counter() - start
            print(f"{tag} declared widths 1..{max_width} compiled in {compile_s:.1f}s")
        torch.testing.assert_close(
            out, reference(width), atol=ATOL, rtol=RTOL, msg=f"at width {width}"
        )
        declared_ms = _median_ms(compiled, rows, *params)
        print(
            f"{tag} width {width}: declared {declared_ms:.1f}ms, "
            f"static padded to {max_width} {static_ms:.1f}ms"
        )

    assert counters["stats"]["unique_graphs"] == 1, "a width outside the first compile recompiled"


# The encoder the runner would run: an embedding loop over the declared token ids,
# NUM_LAYERS layer loops chained through their declared outputs, each sequence
# masked to its own length, and a loop picking every sequence's CLS row for the pooler.
VOCAB = 1024
PAD_ID = 1
NUM_LAYERS = 2
STACK_ATOL = 1e-1
STACK_RTOL = 1e-1
# The CLS loop steps its input a whole sequence (``length`` rows) per trip, and the
# device loop needs every operand's per-trip step to be a multiple of that. One
# output row is one 128-byte stick apart; eight are 1024 bytes, which divides
# every length here. The host keeps the first row of each eight.
CLS_ROWS = 8


def _embed(ids, word, pos, type_row, ln_w, ln_b, layer_norm):
    """``[..., L]`` token ids to ``[..., L, H]``: word + position + segment 0, normed."""
    hidden = word.shape[1]
    x = word.index_select(0, ids.reshape(-1)).reshape(*ids.shape, hidden) + pos + type_row
    return layer_norm(x, (hidden,), ln_w, ln_b)


def _tiled_stack(length, max_width, num_heads):
    """``(embed, layer, cls)``, each one ``for_each_tile`` loop over sequences."""
    from torch_spyre._inductor.wsr import Slice, for_each_tile

    def check(n, granule):
        torch._check(n >= granule)
        torch._check(n <= max_width * granule)
        torch._check(n % granule == 0)

    def embed(ids, *table):
        check(ids.shape[0], 1)

        def body(_carry, tiles):
            ids_tile, *rest = tiles
            return None, _embed(ids_tile.reshape(length), *rest, _arith_layer_norm)

        dims = (0,) + (None,) * len(table)
        return for_each_tile(body, (ids, *table), dims=dims, tile_size=1, out_dim=0)[1]

    def layer(rows, mask, *params):
        """``mask`` is ``[B, L]``; each trip broadcasts its row over heads and queries."""
        check(rows.shape[0], length)
        hidden = rows.shape[1]

        def body(_carry, tiles):
            tile, mask_tile, *weights = tiles
            x = _layer(
                tile.reshape(1, length, hidden),
                weights,
                num_heads,
                _arith_layer_norm,
                mask_tile.reshape(1, 1, 1, length),
            )
            return None, x.reshape(length, hidden)

        dims = (0, Slice(0, 1)) + (None,) * len(params)
        return for_each_tile(body, (rows, mask, *params), dims=dims, tile_size=length, out_dim=0)[1]

    def cls(rows):
        """``[B * CLS_ROWS, H]``: each sequence's first ``CLS_ROWS`` rows, CLS first."""
        check(rows.shape[0], length)

        def body(_carry, tiles):
            (tile,) = tiles
            return None, tile[0:CLS_ROWS].clone()

        return for_each_tile(body, (rows,), dims=(0,), tile_size=length, out_dim=0)[1]

    return embed, layer, cls


def _static_stack(length, num_heads):
    """The same three steps over the whole ``[B, L]`` grid, as the static path runs them."""

    def embed(ids, *table):
        return _embed(ids, *table, _arith_layer_norm).reshape(-1, table[0].shape[1])

    def layer(rows, mask, *params):
        grid = rows.reshape(-1, length, rows.shape[1])
        return _layer(grid, params, num_heads, _arith_layer_norm, mask).reshape(rows.shape)

    def cls(rows):
        return rows.reshape(-1, length, rows.shape[1])[:, 0].clone()

    return embed, layer, cls


@pytest.mark.parametrize("case", sorted(CASES))
def test_encoder_stack_compiles_once_for_every_batch_width(spyre_device, case):
    from torch._dynamo.utils import counters
    from torch.fx.experimental import _config as fx_config
    from torch_spyre._inductor import config as ts_config

    from spyre_inference.custom_ops.utils import row_outermost_layout
    from spyre_inference.v1.attention.backends.spyre_encoder_attn import encoder_key_pad_mask

    length, max_width, hidden, num_heads, intermediate = CASES[case]
    dtype = torch.float16
    gen = torch.Generator().manual_seed(1)

    # The first sequence fills the length; the others are shorter and padded.
    lens = [length, *torch.randint(1, length + 1, (max_width - 1,), generator=gen).tolist()]
    ids_cpu = torch.randint(2, VOCAB, (max_width, length), generator=gen, dtype=torch.int32)
    for s, n in enumerate(lens):
        ids_cpu[s, n:] = PAD_ID
    mask_cpu = encoder_key_pad_mask(length, lens, dtype)

    def w(*shape):
        return (torch.randn(*shape, generator=gen) * 0.05).to(dtype)

    table_cpu = (
        w(VOCAB, hidden),
        w(length, hidden),
        w(hidden),
        torch.ones(hidden, dtype=dtype),
        torch.zeros(hidden, dtype=dtype),
    )
    layers_cpu = [_params(hidden, intermediate, dtype, seed=i) for i in range(NUM_LAYERS)]
    table = tuple(t.to(spyre_device) for t in table_cpu)
    layers = [tuple(t.to(spyre_device) for t in p) for p in layers_cpu]

    def reference(width):
        x = _embed(ids_cpu[:width].long(), *(t.float() for t in table_cpu), F.layer_norm)
        for p in layers_cpu:
            x = _layer(x, [t.float() for t in p], num_heads, F.layer_norm, mask_cpu[:width].float())
        return x[:, 0]

    def run(stack, ids, mask):
        embed, layer, cls = stack
        x = embed(ids, *table)
        for p in layers:
            x = layer(x, mask, *p)
        return cls(x)

    tag = f"[{case} stack]"
    torch._dynamo.reset()
    static = tuple(
        torch.compile(f, dynamic=False, fullgraph=True) for f in _static_stack(length, num_heads)
    )
    static_args = (ids_cpu.to(spyre_device), mask_cpu.to(spyre_device))
    start = time.perf_counter()
    out = run(static, *static_args).cpu().float()
    static_s = time.perf_counter() - start
    torch.testing.assert_close(
        out, reference(max_width), atol=STACK_ATOL, rtol=STACK_RTOL, msg="static"
    )
    static_ms = _median_ms(run, static, *static_args)
    print(
        f"\n{tag} static ({length}, {max_width}) compiled in {static_s:.1f}s, "
        f"runs in {static_ms:.1f}ms"
    )

    torch._dynamo.reset()
    counters.clear()
    tiled = tuple(
        torch.compile(f, fullgraph=True) for f in _tiled_stack(length, max_width, num_heads)
    )
    spec = {"min": 1, "max": max_width, "granularity": 1}

    compile_s = None
    # The ids and mask are declared per sequence, so their range starts at one. A
    # one-sequence trace would otherwise specialize the count, as for decode's pages.
    with (
        ts_config.patch(min_reserved_dim_size=1),
        fx_config.patch(backed_size_oblivious=True),
    ):
        # Both loops step one sequence per trip, which needs the sequence axis
        # outermost: the default layout puts it between the sticks, one stick apart.
        ids = ids_cpu.to(
            spyre_device,
            dynamic={0: spec},
            device_layout=row_outermost_layout((max_width, length), torch.int32),
        )
        mask = mask_cpu.reshape(max_width, length).to(
            spyre_device,
            dynamic={0: spec},
            device_layout=row_outermost_layout((max_width, length), dtype),
        )
        for width in _widths(max_width):
            ids.resize_((width, length))
            mask.resize_((width, length))
            start = time.perf_counter()
            out = run(tiled, ids, mask).cpu().float().reshape(width, CLS_ROWS, hidden)[:, 0]
            if compile_s is None:
                compile_s = time.perf_counter() - start
                print(f"{tag} declared widths 1..{max_width} compiled in {compile_s:.1f}s")
            torch.testing.assert_close(
                out, reference(width), atol=STACK_ATOL, rtol=STACK_RTOL, msg=f"at width {width}"
            )
            declared_ms = _median_ms(run, tiled, ids, mask)
            print(
                f"{tag} width {width}: declared {declared_ms:.1f}ms, "
                f"static padded to {max_width} {static_ms:.1f}ms"
            )

    assert counters["stats"]["unique_graphs"] == 3, "embed, layer and cls should compile once each"
