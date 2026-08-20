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

"""Spyre-inference environment variables.

Central place for config levers: documentation, defaults, lazy evaluation,
and optional caching after service init. Prefer::

    import spyre_inference.envs as envs

    scale = envs.SPYRE_ASYNC_NOISE_SCALE

over scattered ``os.environ.get`` calls.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Help type checkers resolve lazy attributes from environment_variables.
    SPYRE_ASYNC_NOISE_SCALE: int = 4

# Populated by ``enable_envs_cache()``; ``None`` means uncached / lazy.
_env_cache: dict[str, Any] | None = None


def _async_noise_scale() -> int:
    raw = os.getenv("SPYRE_ASYNC_NOISE_SCALE", "4")
    scale = int(raw)
    if scale < 2:
        raise ValueError(
            f"SPYRE_ASYNC_NOISE_SCALE must be >= 2 (got {scale}); "
            "the async ring buffer needs at least one full batch ahead of the consumer."
        )
    return scale


environment_variables: dict[str, Callable[[], Any]] = {
    # Depth of the host-side async Exp(1) log-noise ring buffer:
    # rows = scale * max_num_seqs. Must be >= 2.
    "SPYRE_ASYNC_NOISE_SCALE": _async_noise_scale,
}


def __getattr__(name: str) -> Any:
    """Lazy attribute access into ``environment_variables``.

    After ``enable_envs_cache()``, values are served from ``_env_cache`` (do
    not change env after service init if cache is enabled).
    """
    if name not in environment_variables:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _env_cache is not None:
        return _env_cache[name]
    return environment_variables[name]()


def enable_envs_cache() -> None:
    """Cache env lookups after service initialization."""
    global _env_cache
    if _env_cache is not None:
        return
    _env_cache = {key: getter() for key, getter in environment_variables.items()}


def disable_envs_cache() -> None:
    """Clear the env cache (for tests that mutate ``os.environ``)."""
    global _env_cache
    _env_cache = None


def is_set(name: str) -> bool:
    """Return True if ``name`` is present in the process environment."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())
