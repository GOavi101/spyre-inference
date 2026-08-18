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

"""CPU tests for encoder compile-shape buckets. No Spyre device required."""

from spyre_inference.v1.encoder_buckets import (
    batch_buckets,
    encoder_batch_bucket,
    encoder_len_bucket,
    len_buckets,
    next_bucket,
    pooling_warmup_shapes,
)


def test_next_bucket_picks_smallest_fit():
    assert next_bucket(30, [64, 128, 256]) == 64
    assert next_bucket(64, [64, 128, 256]) == 64
    assert next_bucket(65, [64, 128, 256]) == 128


def test_next_bucket_overflow_stick_aligns():
    assert next_bucket(3000, [64, 128]) == 3008  # 3000 → 47*64 = 3008


def test_default_len_bucket_reuses_64_for_short_prompts():
    assert encoder_len_bucket(1) == 64
    assert encoder_len_bucket(32) == 64
    assert encoder_len_bucket(65) == 128


def test_custom_len_buckets(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "128,512")
    assert encoder_len_bucket(30) == 128
    assert encoder_len_bucket(200) == 512
    assert len_buckets() == [128, 512]


def test_len_buckets_stick_align(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "100,200")
    assert len_buckets() == [128, 256]


def test_default_batch_buckets_are_powers_of_two():
    assert batch_buckets(4) == [1, 2, 4]
    assert batch_buckets(3) == [1, 2, 3]


def test_batch_bucket_pads_to_next_power():
    assert encoder_batch_bucket(1, 4) == 1
    assert encoder_batch_bucket(3, 4) == 4
    assert encoder_batch_bucket(4, 4) == 4


def test_custom_batch_buckets(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_BATCH_SIZES", "1,4")
    assert encoder_batch_bucket(2, 4) == 4
    assert batch_buckets(4) == [1, 4]


def test_warmup_shapes_are_bucket_cartesian(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "64,128")
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_BATCH_SIZES", "1,4")
    assert pooling_warmup_shapes(
        max_num_seqs=4,
        max_model_len=128,
        max_num_batched_tokens=512,
    ) == [(1, 64), (1, 128), (4, 64), (4, 128)]


def test_warmup_shapes_skip_over_token_budget(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "64,256")
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_BATCH_SIZES", "4")
    # 4*256 = 1024 > 200 tokens
    assert pooling_warmup_shapes(
        max_num_seqs=4,
        max_model_len=2048,
        max_num_batched_tokens=200,
    ) == [(4, 64)]
