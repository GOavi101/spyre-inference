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

    if envs.SPYRE_USE_NOISE_POOL:
        ...

over scattered ``os.environ.get`` calls.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Help type checkers resolve lazy attributes from environment_variables.
    SPYRE_USE_NOISE_POOL: bool = False
    SPYRE_NOISE_POOL_MULTIPLIER: int = 32
    SPYRE_NOISE_POOL_DTYPE: str | None = None


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str],
    case_sensitive: bool = False,
) -> Callable[[], str | None]:
    """Return a getter that validates ``env_name`` against ``choices``."""

    def _get_validated_env() -> str | None:
        value = os.getenv(env_name)
        if value is None or value.strip() == "":
            return default

        check_value = value if case_sensitive else value.lower()
        check_choices = choices if case_sensitive else [c.lower() for c in choices]
        if check_value not in check_choices:
            raise ValueError(f"Invalid value '{value}' for {env_name}. Valid options: {choices}.")
        return check_value if not case_sensitive else value

    return _get_validated_env


environment_variables: dict[str, Callable[[], Any]] = {
    # Opt-in host-side Exp(1) noise pool for temperature / top-k / top-p
    # sampling. Sampling still runs on CPU; this only replaces per-step
    # ``exponential_()`` with slices from a pre-filled buffer (useful on
    # s390x). Off by default -> same as upstream vLLM. This is a host
    # stopgap, not on-device / torch.compile sampling.
    "SPYRE_USE_NOISE_POOL": lambda: os.getenv("SPYRE_USE_NOISE_POOL", "0") == "1",
    # Pool size = multiplier * max_num_seqs * vocab_size.
    "SPYRE_NOISE_POOL_MULTIPLIER": lambda: int(os.getenv("SPYRE_NOISE_POOL_MULTIPLIER", "32")),
    # Explicit pool dtype: "float16" / "float32". None -> auto
    # (float32 on s390x/ppc64le, else float16).
    "SPYRE_NOISE_POOL_DTYPE": env_with_choices(
        "SPYRE_NOISE_POOL_DTYPE",
        None,
        ["float16", "fp16", "half", "float32", "fp32", "float"],
        case_sensitive=False,
    ),
}


def __getattr__(name: str):
    """Lazy attribute access into ``environment_variables``.

    After ``enable_envs_cache()``, values are cached (do not change env after
    service init if cache is enabled).
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _is_envs_cache_enabled() -> bool:
    global __getattr__
    return hasattr(__getattr__, "cache_clear")


def enable_envs_cache() -> None:
    """Cache env lookups after service initialization."""
    if _is_envs_cache_enabled():
        return
    global __getattr__
    __getattr__ = functools.cache(__getattr__)
    for key in environment_variables:
        __getattr__(key)


def disable_envs_cache() -> None:
    """Clear the env cache (for tests that mutate ``os.environ``)."""
    global __getattr__
    if _is_envs_cache_enabled():
        assert hasattr(__getattr__, "__wrapped__")
        __getattr__ = __getattr__.__wrapped__


def is_set(name: str) -> bool:
    """Return True if ``name`` is present in the process environment."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())
