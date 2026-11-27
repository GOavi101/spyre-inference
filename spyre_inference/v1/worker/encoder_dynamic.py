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

"""Encoder rectangles over a declared batch width (``SPYRE_ENCODER_DYNAMIC_WIDTH``).

A rectangle's rows are ``B * L``. With ``L`` fixed per graph, the rows only change in
whole sequences, so they are declared to torch-spyre on dim 0 (``granularity = S * L``,
``S`` sequences per loop trip) and one block graph per length covers every width.

torch-spyre compiles a declared dim only as a ``for_each_tile`` loop over it, so every op
on the rows runs inside one (``for_each_row_tile``), and a compiled output carries the
range but not Dynamo's dynamic mark, which ``mark_reserved_rows`` restores between graphs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch
from vllm.config import CompilationMode, VllmConfig
from vllm.logger import init_logger

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.worker.spyre_shape_bucketer import _encoder_shape_tables, _floor_pow2

logger = init_logger(__name__)


@dataclass(frozen=True)
class RectRange:
    """One length's declared widths: ``seqs_per_trip`` to ``max_width`` sequences of ``extent``.

    Instances are cached, so a compiled graph that reads one off the published grid
    guards on a stable object, one per length.
    """

    extent: int
    seqs_per_trip: int
    max_width: int

    @property
    def granule(self) -> int:
        """Rows per loop trip, and the declared granularity."""
        return self.extent * self.seqs_per_trip

    @property
    def max_rows(self) -> int:
        return self.extent * self.max_width

    def width_for(self, num_seqs: int) -> int:
        """``num_seqs`` rounded up to whole trips."""
        trips = -(-max(1, num_seqs) // self.seqs_per_trip)
        return trips * self.seqs_per_trip

    def widths(self) -> list[int]:
        return list(range(self.seqs_per_trip, self.max_width + 1, self.seqs_per_trip))

    def declaration(self) -> dict[str, int]:
        """The ``dynamic={0: ...}`` entry for a ``[B * L, N]`` tensor."""
        return {"min": self.granule, "max": self.max_rows, "granularity": self.granule}


def dynamic_width_requested(vllm_config: VllmConfig) -> bool:
    """Whether the config asks for declared widths and could compile them."""
    if not envs.SPYRE_ENCODER_DYNAMIC_WIDTH:
        return False
    model_config = vllm_config.model_config
    return (
        getattr(model_config, "runner_type", None) == "pooling"
        and not model_config.enforce_eager
        and vllm_config.compilation_config.mode is CompilationMode.STOCK_TORCH_COMPILE
        and vllm_config.parallel_config.tensor_parallel_size == 1
    )


def encoder_rect_ranges(vllm_config: VllmConfig) -> tuple[RectRange, ...]:
    """Declared-width ranges for this config; empty when not requested."""
    if not dynamic_width_requested(vllm_config):
        return ()
    ranges = _rect_ranges(
        int(vllm_config.model_config.max_model_len),
        int(vllm_config.scheduler_config.max_num_batched_tokens),
        max(1, int(vllm_config.scheduler_config.max_num_seqs)),
        envs.SPYRE_ATTN_QUERY_BUCKETS,
        envs.SPYRE_ENCODER_SEQS_PER_TRIP,
    )
    return tuple(r for r in ranges if _reservation_ok(r))


@lru_cache(maxsize=8)
def _rect_ranges(
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    query_buckets_override: str | None,
    seqs_per_trip: int,
) -> tuple[RectRange, ...]:
    """One range per ladder length whose widest batch spans more than one trip.

    The widest batch is ``min(max_num_seqs, budget // L)``: the scheduler cap, and the
    rows of the largest body. A length whose range would be a single size keeps a
    static rectangle, since a declared range with ``min == max`` does not trace.
    """
    tables = _encoder_shape_tables(
        max_model_len, max_num_batched_tokens, max_num_seqs, query_buckets_override
    )
    ranges = []
    for extent in tables.lengths:
        widest = min(max_num_seqs, tables.budget // extent)
        per_trip = min(_floor_pow2(max(1, seqs_per_trip)), _floor_pow2(widest))
        max_width = widest // per_trip * per_trip
        if max_width > per_trip:
            ranges.append(RectRange(extent, per_trip, max_width))
    return tuple(ranges)


def _reservation_ok(rect: RectRange) -> bool:
    """torch-spyre's rules for a declaration; kept when torch-spyre is absent (CPU tests)."""
    try:
        from torch_spyre._inductor import config as ts_config
    except ImportError:
        return True
    decl = rect.declaration()
    violations = ts_config.reservation_violations(decl["min"], decl["max"], decl["granularity"])
    if violations:
        logger.warning_once(
            "Encoder length %d keeps a static graph per width: %s.", rect.extent, violations[0]
        )
    return not violations


def range_for_batch(
    num_seqs: int, max_len: int, lengths: Sequence[int], ranges: Sequence[RectRange]
) -> tuple[RectRange, int] | None:
    """The range and padded width this batch runs in, or None.

    Takes the shortest ladder length covering ``max_len``, as the static rectangles do.
    None when that length has no range or the batch is wider than it.
    """
    if num_seqs < 1 or max_len < 1:
        return None
    extent = next((length for length in lengths if length >= max_len), None)
    rect = next((r for r in ranges if r.extent == extent), None)
    if rect is None or num_seqs > rect.max_width:
        return None
    return rect, rect.width_for(num_seqs)


def upload_declared(
    host: torch.Tensor, rect: RectRange, device: torch.device, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Copy ``host`` (``[B * L]`` or ``[B * L, N]``) to ``device`` with its rows declared
    as ``rect``. A 1-D ``host`` comes back as ``[B * L, 1]``: torch-spyre cannot reserve
    the innermost dim.

    Copied whole at the reserved size, then shrunk: copying part of a reservation is
    several times slower (torch-spyre#6026). A plain copy off Spyre.
    """
    if host.ndim == 1:
        host = host.unsqueeze(1)
    if device.type != "spyre":
        return convert(host, device=device, dtype=dtype)
    padded = host.new_zeros((rect.max_rows, *host.shape[1:]))
    padded[: host.shape[0]] = host
    return convert(padded, device=device, dtype=dtype, dynamic={0: rect.declaration()}).resize_(
        host.shape
    )


def is_declared(x: object) -> bool:
    """Whether ``x`` is a Spyre tensor with a dim-0 reservation."""
    if not isinstance(x, torch.Tensor) or x.device.type != "spyre":
        return False
    from torch_spyre._C import get_reserved_dims

    dims = get_reserved_dims(x)
    return bool(dims) and 0 in dims


def mark_reserved_rows(x: torch.Tensor) -> None:
    """Make a declared tensor's rows symbolic for the next compiled graph.

    A compiled output keeps its reservation but not ``mark_dynamic``, and blocks compile
    with ``dynamic=False``, so without this each width would compile its own graph.
    """
    if not torch.compiler.is_compiling() and is_declared(x):
        torch._dynamo.mark_dynamic(x, 0)


def check_rows(rows: int, rect: RectRange) -> None:
    """Restate the declared range inside the trace, as torch-spyre's recipe requires."""
    torch._check(rows >= rect.granule)
    torch._check(rows <= rect.max_rows)
    torch._check(rows % rect.granule == 0)


def for_each_row_tile(
    body: Callable[..., torch.Tensor],
    operands: Sequence[torch.Tensor],
    dims: Sequence[int | None],
    tile_size: int,
) -> torch.Tensor:
    """``body(*tiles)`` per ``tile_size`` rows of the dim-0 operands, stacked on dim 0.

    ``torch_spyre``'s ``for_each_tile`` on Spyre; a host loop elsewhere, the reference
    CPU tests compare it against.
    """
    if any(op.device.type == "spyre" for op in operands):
        from torch_spyre._inductor.wsr import for_each_tile

        _, out = for_each_tile(
            lambda _carry, tiles: (None, body(*tiles)),
            tuple(operands),
            dims=tuple(dims),
            tile_size=tile_size,
            out_dim=0,
        )
        return out
    rows = next(op.shape[0] for op, dim in zip(operands, dims) if dim == 0)
    outs = []
    for start in range(0, rows, tile_size):
        tiles = [
            op.narrow(0, start, tile_size) if dim == 0 else op for op, dim in zip(operands, dims)
        ]
        outs.append(body(*tiles))
    return torch.cat(outs)
