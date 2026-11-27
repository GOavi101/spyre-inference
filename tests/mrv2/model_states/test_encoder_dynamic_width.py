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

"""Declared-width encoder rectangles (SPYRE_ENCODER_DYNAMIC_WIDTH): the ranges, the
plan a batch gets, and the grid the model state lays out. CPU-only."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spyre_inference import envs
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    EncoderRectPlan,
    encoder_grid,
    encoder_key_pad_mask,
)
from spyre_inference.v1.worker.encoder_dynamic import (
    RectRange,
    _rect_ranges,
    encoder_rect_ranges,
    for_each_row_tile,
    range_for_batch,
)
from spyre_inference.v1.worker.spyre.model_states.encoder_only import (
    SpyreEncoderOnlyModelState,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import _encoder_shape_tables

from .state_fakes import make_config, make_input_batch
from .test_encoder_only import _EncoderBuilder, _step

# bge-reranker-v2-m3 in #1432: max_model_len 8192, max_num_seqs 4.
_BGE = (8192, 8192, 4, None)
_LENGTHS = _encoder_shape_tables(*_BGE).lengths


@pytest.fixture(autouse=True)
def _flag(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_DYNAMIC_WIDTH", "1")
    envs.clear_env_cache()
    encoder_grid().publish(None)
    yield
    encoder_grid().publish(None)
    envs.clear_env_cache()


def test_one_graph_per_length_for_bge_reranker():
    ranges = _rect_ranges(*_BGE, 1)
    assert [(r.extent, r.max_width) for r in ranges] == [
        (64, 4),
        (128, 4),
        (256, 4),
        (512, 4),
        (1024, 4),
        (2048, 4),
        (4096, 2),
    ]
    # 8192 holds one sequence, so it keeps its static rectangle: 7 ranges + 1 = 8 graphs,
    # against 12 static rectangles today.
    static = {extent for extent, _ in _encoder_shape_tables(*_BGE).rectangles}
    assert len(static - {r.extent for r in ranges}) == 1
    assert len(_encoder_shape_tables(*_BGE).rectangles) == 12


def test_two_sequences_per_trip():
    ranges = _rect_ranges(*_BGE, 3)  # rounded down to 2
    assert {r.seqs_per_trip for r in ranges} == {2}
    # 4096 is at most 2 wide, a single trip, so it goes static too.
    assert [r.extent for r in ranges] == [64, 128, 256, 512, 1024, 2048]
    assert ranges[0].widths() == [2, 4]
    assert ranges[0].width_for(3) == 4
    assert ranges[0].declaration() == {"min": 128, "max": 256, "granularity": 128}


def test_range_declaration_at_one_sequence_per_trip():
    rect = RectRange(1024, 1, 4)
    assert rect.widths() == [1, 2, 3, 4]
    assert rect.width_for(3) == 3
    assert rect.declaration() == {"min": 1024, "max": 4096, "granularity": 1024}


@pytest.mark.parametrize(
    ("num_seqs", "max_len", "want"),
    [
        (2, 1100, (2048, 2)),  # static today: the 2048 body's 2048 rectangle is 1 wide
        (3, 100, (128, 3)),
        (4, 64, (64, 4)),
        (2, 3000, (4096, 2)),
        (3, 3000, None),  # wider than 4096's range
        (1, 5000, None),  # 8192 has no range
        (0, 10, None),
    ],
)
def test_range_for_batch(num_seqs, max_len, want):
    picked = range_for_batch(num_seqs, max_len, _LENGTHS, _rect_ranges(*_BGE, 1))
    got = None if picked is None else (picked[0].extent, picked[1])
    assert got == want


def test_for_each_row_tile_reference_loop():
    x = torch.arange(12.0).view(6, 2)
    w = torch.tensor([10.0, 100.0])
    out = for_each_row_tile(lambda t, s: t.sum(0, keepdim=True) * s, (x, w), (0, None), 2)
    want = torch.stack([x[i : i + 2].sum(0) * w for i in range(0, 6, 2)])
    torch.testing.assert_close(out, want)


def _bge_config(**overrides):
    config = make_config(max_model_len=8192, max_num_seqs=4, max_num_batched_tokens=8192)
    config.model_config.runner_type = "pooling"
    config.model_config.enforce_eager = False
    config.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    for key, value in overrides.items():
        setattr(config.model_config, key, value)
    return config


def _state(tiled: bool = True, causal: bool = False, **overrides) -> SpyreEncoderOnlyModelState:
    model = nn.Module()
    if tiled:
        model.spyre_dynamic_width = True
    state = SpyreEncoderOnlyModelState(
        _bge_config(**overrides),  # ty: ignore[invalid-argument-type]
        model,
        None,
        torch.device("cpu"),
        torch.device("cpu"),
    )
    builder = _EncoderBuilder(causal)
    group = SimpleNamespace(layer_names=["enc.0", "enc.1"], get_metadata_builder=lambda _: builder)
    state.encoder_attn_groups = [group]  # ty: ignore[invalid-assignment]
    return state


def test_ranges_need_the_flag_and_a_pooling_compile(monkeypatch):
    assert encoder_rect_ranges(_bge_config())  # ty: ignore[invalid-argument-type]
    eager = _bge_config(enforce_eager=True)
    assert not encoder_rect_ranges(eager)  # ty: ignore[invalid-argument-type]
    monkeypatch.setenv("SPYRE_ENCODER_DYNAMIC_WIDTH", "0")
    envs.clear_env_cache()
    assert not encoder_rect_ranges(_bge_config())  # ty: ignore[invalid-argument-type]


def test_a_batch_static_rectangles_miss_gets_a_declared_width():
    state = _state()
    _, inputs = _step(state, make_input_batch([1100, 10], num_tokens_after_padding=2048))
    plan = state.encoder_plan.attn_plan
    assert isinstance(plan, EncoderRectPlan)
    assert (plan.extent, plan.width, plan.mask) == (2048, 2, None)
    assert plan.rect_range is not None and plan.rect_range.extent == 2048
    want_mask = encoder_key_pad_mask(2048, [1100, 10], torch.float16).reshape(-1)
    torch.testing.assert_close(plan.row_mask, want_mask)
    # The grid is the declared width, not the 2048-row body.
    assert inputs["input_ids"].shape == (4096,)
    assert inputs["positions"].shape == (4096,)
    grid = encoder_grid()
    assert grid.mask is None and grid.rect_range is plan.rect_range
    assert grid.row_mask is plan.row_mask


def test_width_rounds_to_the_batch_not_the_body():
    state = _state()
    _step(state, make_input_batch([100, 90, 80], num_tokens_after_padding=2048))
    assert state.encoder_plan.grid == (128, 3, [100, 90, 80])
    # Batch-pad lanes only up to the width: none here.
    assert state.encoder_plan.attn_plan.row_mask.shape == (3 * 128,)


def test_a_model_without_the_tiled_encoder_keeps_static_rectangles():
    state = _state(tiled=False)
    _step(state, make_input_batch([1100, 10], num_tokens_after_padding=2048))
    assert not state.encoder_plan.is_rectangular
    _step(state, make_input_batch([100], num_tokens_after_padding=2048))
    assert state.encoder_plan.grid == (128, 16, [100])
    assert state.encoder_plan.rect_range is None


def test_causal_groups_keep_static_rectangles():
    state = _state(causal=True)
    _step(state, make_input_batch([100], num_tokens_after_padding=2048))
    assert state.encoder_plan.grid == (128, 16, [100])


def test_a_length_without_a_range_keeps_its_static_rectangle():
    state = _state()
    _step(state, make_input_batch([5000], num_tokens_after_padding=8192))
    assert state.encoder_plan.grid == (8192, 1, [5000])
    assert state.encoder_plan.rect_range is None


def test_forced_declared_width():
    state = _state()
    with state.forced_plan(rect=(128, 3)):
        _step(state, make_input_batch([128] * 3, num_tokens_after_padding=384))
    assert state.encoder_plan.grid == (128, 3, [128] * 3)
    assert state.encoder_plan.rect_range.extent == 128
    with pytest.raises(ValueError, match="not a declared"), state.forced_plan(rect=(128, 5)):
        pass


def test_forced_ragged_ignores_the_ranges():
    state = _state()
    with state.forced_plan(ragged=True):
        _step(state, make_input_batch([100, 90], num_tokens_after_padding=2048))
    assert not state.encoder_plan.is_rectangular
