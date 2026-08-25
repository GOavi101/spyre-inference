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

"""Unit tests for SpyreShapeBucketer."""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from spyre_inference.v1.worker.spyre_shape_bucketer import (
    EncoderBucketDescriptor,
    SpyreShapeBucketer,
)


@pytest.fixture()
def mock_vllm_config():
    """Create a minimal VllmConfig mock with compile_sizes."""
    config = MagicMock()
    config.compilation_config.compile_sizes = [1, 2, 4, 8, 16]
    return config


@pytest.fixture()
def bucketer(mock_vllm_config):
    return SpyreShapeBucketer(mock_vllm_config)


class TestFindBucket:
    def test_exact_match(self, bucketer):
        assert bucketer.find_bucket(8) == 8

    def test_rounds_up_to_next_bucket(self, bucketer):
        assert bucketer.find_bucket(3) == 4
        assert bucketer.find_bucket(5) == 8
        assert bucketer.find_bucket(9) == 16

    def test_smallest_token_count(self, bucketer):
        assert bucketer.find_bucket(1) == 1

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.find_bucket(17) is None
        assert bucketer.find_bucket(100) is None

    def test_zero_tokens(self, bucketer):
        assert bucketer.find_bucket(0) == 1


class TestDispatch:
    def test_returns_descriptor_with_padding(self, bucketer):
        desc = bucketer.dispatch(5)
        assert desc is not None
        assert desc.actual_num_tokens == 5
        assert desc.padded_num_tokens == 8

    def test_exact_match_no_padding(self, bucketer):
        desc = bucketer.dispatch(4)
        assert desc is not None
        assert desc.actual_num_tokens == 4
        assert desc.padded_num_tokens == 4

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.dispatch(20) is None

    def test_descriptor_is_frozen(self, bucketer):
        desc = bucketer.dispatch(3)
        with pytest.raises(FrozenInstanceError):
            desc.actual_num_tokens = 10


class TestBucketerState:
    def test_initial_state_not_warmed_up(self, bucketer):
        assert not bucketer.is_warmed_up

    def test_mark_warmed_up(self, bucketer):
        bucketer.mark_warmed_up()
        assert bucketer.is_warmed_up

    def test_bucket_sizes_sorted(self, bucketer):
        assert bucketer.bucket_sizes == [1, 2, 4, 8, 16]

    def test_max_bucket_size(self, bucketer):
        assert bucketer.max_bucket_size == 16


class TestEdgeCases:
    def test_empty_compile_sizes(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = []
        b = SpyreShapeBucketer(config)
        assert b.bucket_sizes == []
        assert b.max_bucket_size == 0
        assert b.find_bucket(1) is None
        assert b.dispatch(1) is None

    def test_single_bucket(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = [8]
        b = SpyreShapeBucketer(config)
        assert b.find_bucket(1) == 8
        assert b.find_bucket(8) == 8
        assert b.find_bucket(9) is None

    def test_unsorted_input_gets_sorted(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = [16, 2, 8, 1, 4]
        b = SpyreShapeBucketer(config)
        assert b.bucket_sizes == [1, 2, 4, 8, 16]


def _pooling_vllm_config(
    *,
    max_num_seqs: int = 4,
    max_model_len: int = 128,
    max_num_batched_tokens: int = 512,
    runner_type: str = "pooling",
) -> MagicMock:
    config = MagicMock()
    config.model_config.runner_type = runner_type
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.compilation_config.compile_sizes = []
    return config


class TestEncoderDispatch:
    def test_for_pooling_loads_warmup_shapes(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "64,128")
        monkeypatch.setenv("SPYRE_ENCODER_BUCKET_BATCH_SIZES", "1,4")
        b = SpyreShapeBucketer.for_pooling(_pooling_vllm_config())
        assert b is not None
        assert b.encoder_shapes == [(1, 64), (1, 128), (4, 64), (4, 128)]
        assert b.bucket_sizes == [64, 128, 256, 512]

    def test_for_pooling_skips_non_pooling(self):
        assert SpyreShapeBucketer.for_pooling(_pooling_vllm_config(runner_type="generate")) is None

    def test_for_pooling_none_when_no_shapes(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "256")
        monkeypatch.setenv("SPYRE_ENCODER_BUCKET_BATCH_SIZES", "4")
        b = SpyreShapeBucketer.for_pooling(
            _pooling_vllm_config(max_model_len=64, max_num_batched_tokens=128)
        )
        assert b is None

    def test_dispatch_encoder_pads_to_warmed_cell(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "64,128")
        monkeypatch.setenv("SPYRE_ENCODER_BUCKET_BATCH_SIZES", "1,4")
        b = SpyreShapeBucketer.for_pooling(_pooling_vllm_config())
        assert b is not None
        desc = b.dispatch_encoder(
            num_seqs=3,
            max_query_len=30,
            max_num_seqs=4,
            max_model_len=128,
            max_num_batched_tokens=512,
        )
        assert desc is not None
        assert (desc.batch_bucket, desc.len_bucket) == (4, 64)
        assert desc.padded_num_tokens == 256
        assert desc.actual_num_seqs == 3
        assert desc.actual_max_len == 30

    def test_dispatch_encoder_stays_on_warmed_shapes(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = []
        b = SpyreShapeBucketer(config, encoder_shapes=[(4, 64)])
        desc = b.dispatch_encoder(
            num_seqs=1,
            max_query_len=30,
            max_num_seqs=4,
            max_model_len=128,
            max_num_batched_tokens=512,
        )
        assert desc is not None
        assert (desc.batch_bucket, desc.len_bucket) == (4, 64)

    def test_dispatch_encoder_none_when_over_token_budget(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = []
        b = SpyreShapeBucketer(config, encoder_shapes=[(4, 64)])
        assert (
            b.dispatch_encoder(
                num_seqs=3,
                max_query_len=30,
                max_num_seqs=4,
                max_model_len=2048,
                max_num_batched_tokens=200,
            )
            is None
        )

    def test_dispatch_encoder_none_on_1d_bucketer(self, bucketer):
        assert (
            bucketer.dispatch_encoder(
                num_seqs=1,
                max_query_len=8,
                max_num_seqs=4,
                max_model_len=128,
                max_num_batched_tokens=512,
            )
            is None
        )

    def test_encoder_descriptor_is_frozen(self):
        desc = EncoderBucketDescriptor(
            batch_bucket=4, len_bucket=64, actual_num_seqs=3, actual_max_len=30
        )
        with pytest.raises(FrozenInstanceError):
            desc.batch_bucket = 1
