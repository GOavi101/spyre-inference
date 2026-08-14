# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for spyre_inference.envs."""

from __future__ import annotations

import os

import pytest

import spyre_inference.envs as envs


@pytest.fixture(autouse=True)
def _clear_envs_cache_and_noise_vars():
    envs.disable_envs_cache()
    saved = {
        k: os.environ.pop(k, None)
        for k in (
            "SPYRE_USE_NOISE_POOL",
            "SPYRE_NOISE_POOL_MULTIPLIER",
            "SPYRE_NOISE_POOL_DTYPE",
        )
    }
    try:
        yield
    finally:
        envs.disable_envs_cache()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_noise_pool_defaults_off():
    assert envs.SPYRE_USE_NOISE_POOL is False
    assert envs.SPYRE_NOISE_POOL_MULTIPLIER == 32
    assert envs.SPYRE_NOISE_POOL_DTYPE is None


def test_noise_pool_enable_and_multiplier(monkeypatch):
    monkeypatch.setenv("SPYRE_USE_NOISE_POOL", "1")
    monkeypatch.setenv("SPYRE_NOISE_POOL_MULTIPLIER", "8")
    assert envs.SPYRE_USE_NOISE_POOL is True
    assert envs.SPYRE_NOISE_POOL_MULTIPLIER == 8


def test_noise_pool_dtype_choices(monkeypatch):
    monkeypatch.setenv("SPYRE_NOISE_POOL_DTYPE", "FP32")
    assert envs.SPYRE_NOISE_POOL_DTYPE == "fp32"
    monkeypatch.setenv("SPYRE_NOISE_POOL_DTYPE", "float16")
    assert envs.SPYRE_NOISE_POOL_DTYPE == "float16"


def test_noise_pool_dtype_invalid(monkeypatch):
    monkeypatch.setenv("SPYRE_NOISE_POOL_DTYPE", "bfloat16")
    with pytest.raises(ValueError, match="SPYRE_NOISE_POOL_DTYPE"):
        _ = envs.SPYRE_NOISE_POOL_DTYPE


def test_enable_envs_cache_freezes_value(monkeypatch):
    monkeypatch.setenv("SPYRE_USE_NOISE_POOL", "0")
    envs.enable_envs_cache()
    assert envs.SPYRE_USE_NOISE_POOL is False
    monkeypatch.setenv("SPYRE_USE_NOISE_POOL", "1")
    # Cached: still False until cache disabled.
    assert envs.SPYRE_USE_NOISE_POOL is False
    envs.disable_envs_cache()
    assert envs.SPYRE_USE_NOISE_POOL is True
