#!/usr/bin/env python3
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

"""Time Gemma 4 prefill MoE: every expert, then only the selected pairs.

The packed path is what ``SPYRE_MOE_PACKED_PREFILL=1`` runs. The default stays off until a
card run of this script shows its median wall time, including the host slot-table sync, is
lower than the all-expert loop. Attention is not in this process; time it with
``scripts/microbench/spyre_attn_microbench.py``. The dense MLP timed here has intermediate
size ``8 * 704``, the width of the eight selected experts, and is one layer of the 30.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("SPYRE_NUM_CPUS", "8")
os.environ.setdefault("VLLM_PLUGINS", "spyre_inference")
os.environ.setdefault("SPYRE_MOE_PACKED_PREFILL", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.autograd.profiler import DeviceType  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

EXPERTS, HIDDEN, TOP_K = 128, 2816, 8
DTYPE = torch.float16
SPAN = "gemma4_prefill_moe"
# Eight selected experts, as a single dense MLP. Not a checkpoint intermediate size.
DENSE_INTERMEDIATE = 8 * 704
LAYERS = 30


def _device_layer(intermediate: int, seed: int) -> tuple[SimpleNamespace, dict[str, torch.Tensor]]:
    from torch_spyre._C import get_elem_in_stick

    from spyre_inference.moe import (
        SpyreMoERecipe,
        _route_reduce_dtype,
        _to_spyre_expert_weight,
    )

    stick = get_elem_in_stick(DTYPE)
    pad = -intermediate % stick
    generator = torch.Generator().manual_seed(seed)
    gate = torch.randn(EXPERTS, HIDDEN, intermediate, dtype=DTYPE, generator=generator).mul_(0.01)
    up = torch.randn(EXPERTS, HIDDEN, intermediate, dtype=DTYPE, generator=generator).mul_(0.01)
    down = torch.randn(EXPERTS, intermediate, HIDDEN, dtype=DTYPE, generator=generator).mul_(0.01)
    down.mul_(torch.rand(EXPERTS, dtype=DTYPE, generator=generator)[:, None, None] + 0.5)
    layer = SimpleNamespace(
        spyre_moe_recipe=SpyreMoERecipe("gelu_tanh", "full_softmax"),
        spyre_moe_gate=_to_spyre_expert_weight(gate, (0, pad)),
        spyre_moe_up=_to_spyre_expert_weight(up, (0, pad)),
        spyre_moe_down=_to_spyre_expert_weight(down, (0, 0, 0, pad), kernel_order=True),
        spyre_moe_biases=None,
        spyre_moe_stick=stick,
        spyre_moe_route_dtype=_route_reduce_dtype(EXPERTS, DTYPE),
        spyre_moe_route_identity=torch.eye(stick, dtype=DTYPE).to("spyre"),
        spyre_moe_regions={},
        top_k=TOP_K,
    )
    return layer, {"gate": gate, "up": up, "down": down}


def _dense_reference(
    x: torch.Tensor, router_logits: torch.Tensor, weights: dict[str, torch.Tensor]
) -> torch.Tensor:
    probs = torch.softmax(router_logits.float(), dim=-1).to(router_logits.dtype)
    selected, indices = torch.topk(probs.float(), TOP_K, dim=-1)
    route_weights = selected / selected.sum(-1, keepdim=True)
    output = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        row = x[token : token + 1].float()
        for slot in range(TOP_K):
            expert = int(indices[token, slot])
            gate = F.gelu(row @ weights["gate"][expert].float(), approximate="tanh")
            up = row @ weights["up"][expert].float()
            output[token] += ((gate * up) @ weights["down"][expert].float()).squeeze(0) * float(
                route_weights[token, slot]
            )
    return output


def _profile_once(fn: Any, args: tuple[Any, ...]) -> tuple[float, list[dict[str, float | str]]]:
    span_us: float | None = None
    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            record_shapes=False,
            acc_events=True,
        ) as prof,
        torch.profiler.record_function(SPAN),
    ):
        fn(*args)
        torch.spyre.synchronize()

    for event in prof.events():
        if event.name == SPAN:
            span_us = event.time_range.elapsed_us()
            break
    if span_us is None:
        raise RuntimeError(f"profiler did not record {SPAN!r}")
    result = prof.profiler.kineto_results
    if result is None:
        raise RuntimeError("profiler did not produce Kineto results")
    device_events: list[dict[str, float | str]] = []
    for event in result.events():
        try:
            name = event.name()
        except UnicodeDecodeError:
            name = "<undecodable>"
        if event.device_type() == DeviceType.CPU:
            continue
        duration_us = event.duration_ns() / 1000
        if duration_us > 0:
            device_events.append({"name": name, "duration_us": duration_us})
    if not device_events:
        raise RuntimeError("no Spyre device events; check that torch-spyre includes AIUPTI")
    return span_us / 1000, device_events


def _stats(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def _time_call(fn: Any, args: tuple[Any, ...], iterations: int, warmup: int, profile_iters: int):
    for _ in range(warmup):
        fn(*args)
        torch.spyre.synchronize()
    wall_ms = []
    for _ in range(iterations):
        torch.spyre.synchronize()
        start = time.perf_counter_ns()
        fn(*args)
        torch.spyre.synchronize()
        wall_ms.append((time.perf_counter_ns() - start) / 1_000_000)
    profile_span_ms = []
    compute_ms = []
    memory_ms = []
    counts = []
    events_by_iter = []
    profile_error = None
    for profile_iteration in range(profile_iters):
        try:
            span_ms, events = _profile_once(fn, args)
        except RuntimeError as exc:
            if "no Spyre device events" not in str(exc):
                raise
            profile_error = str(exc)
            break
        memory_us = sum(
            float(event["duration_us"])
            for event in events
            if any(marker in str(event["name"]).lower() for marker in ("memcpy", "memset"))
        )
        all_us = sum(float(event["duration_us"]) for event in events)
        profile_span_ms.append(span_ms)
        compute_ms.append((all_us - memory_us) / 1000)
        memory_ms.append(memory_us / 1000)
        counts.append(len(events))
        events_by_iter.append({"iteration": profile_iteration + 1, "events": events})
    timed = {"latency_ms": _stats(wall_ms)}
    if profile_error is None:
        timed.update(
            {
                "profiled_cpu_span_ms": _stats(profile_span_ms),
                "compute_device_event_sum_ms": _stats(compute_ms),
                "memcpy_memset_event_sum_ms": _stats(memory_ms),
                "device_event_counts": counts,
                "profiled_device_events": events_by_iter,
            }
        )
    else:
        timed["device_profile"] = profile_error
    return timed


def _apply(layer: SimpleNamespace, x: torch.Tensor, logits: torch.Tensor, packed: bool):
    from spyre_inference import envs
    from spyre_inference.moe import SpyreUnquantizedFusedMoEMethod

    os.environ["SPYRE_MOE_PACKED_PREFILL"] = "1" if packed else "0"
    envs.clear_env_cache()
    method = object.__new__(SpyreUnquantizedFusedMoEMethod)
    return method.apply_monolithic(layer, x, logits)


def _run_case(
    tokens: int,
    intermediate: int,
    layer: SimpleNamespace,
    weights: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from torch_spyre.ops.fallbacks import FallbackWarning

    generator = torch.Generator().manual_seed(args.seed + tokens + intermediate)
    x = torch.randn(tokens, HIDDEN, dtype=DTYPE, generator=generator).mul_(0.05)
    router_logits = torch.randn(tokens, EXPERTS, dtype=DTYPE, generator=generator)
    x_device, logits_device = x.to("spyre"), router_logits.to("spyre")
    expected = _dense_reference(x, router_logits, weights)
    paths: dict[str, Any] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        for name, packed in (("persistent", False), ("packed", True)):
            compile_start = time.perf_counter()
            actual = _apply(layer, x_device, logits_device, packed)
            torch.spyre.synchronize()
            compile_seconds = time.perf_counter() - compile_start
            torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=2e-2)
            timed = _time_call(
                lambda packed=packed: _apply(layer, x_device, logits_device, packed),
                (),
                args.iterations,
                args.warmup_iters,
                args.profile_iters,
            )
            timed["compile_and_first_run_s"] = compile_seconds
            timed["correctness"] = "matches dense CPU reference"
            paths[name] = timed
    fallbacks = [str(item.message) for item in caught if issubclass(item.category, FallbackWarning)]
    if fallbacks:
        raise RuntimeError(f"{tokens} tokens, intermediate {intermediate} fell back: {fallbacks}")
    persistent_ms = paths["persistent"]["latency_ms"]["median_ms"]
    packed_ms = paths["packed"]["latency_ms"]["median_ms"]
    return {
        "tokens": tokens,
        "intermediate": intermediate,
        "experts": EXPERTS,
        "hidden": HIDDEN,
        "top_k": TOP_K,
        "dtype": str(DTYPE),
        "persistent": paths["persistent"],
        "packed": paths["packed"],
        "packed_over_persistent": packed_ms / persistent_ms if persistent_ms else None,
        "fallback_clean": True,
        "layers": LAYERS,
        "stack_estimate_ms": {
            "persistent": persistent_ms * LAYERS,
            "packed": packed_ms * LAYERS,
            "note": "30 MoE layers only; attention and the dense MLP are not in this product",
        },
    }


def _run_dense(tokens: int, args: argparse.Namespace) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(args.seed + 17)
    gate = torch.randn(HIDDEN, DENSE_INTERMEDIATE, dtype=DTYPE, generator=generator).mul_(0.01)
    up = torch.randn(HIDDEN, DENSE_INTERMEDIATE, dtype=DTYPE, generator=generator).mul_(0.01)
    down = torch.randn(DENSE_INTERMEDIATE, HIDDEN, dtype=DTYPE, generator=generator).mul_(0.01)
    x = torch.randn(tokens, HIDDEN, dtype=DTYPE, generator=generator).mul_(0.05)
    gate_d, up_d, down_d, x_d = (t.to("spyre") for t in (gate, up, down, x))

    def dense() -> torch.Tensor:
        return (F.gelu(x_d @ gate_d, approximate="tanh") * (x_d @ up_d)) @ down_d

    from torch_spyre.ops.fallbacks import FallbackWarning

    compiled = torch.compile(dense, backend="inductor", fullgraph=True, dynamic=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        compile_start = time.perf_counter()
        compiled()
        torch.spyre.synchronize()
        compile_seconds = time.perf_counter() - compile_start
        timed = _time_call(compiled, (), args.iterations, args.warmup_iters, args.profile_iters)
    fallbacks = [str(item.message) for item in caught if issubclass(item.category, FallbackWarning)]
    if fallbacks:
        raise RuntimeError(f"dense MLP fell back to CPU: {fallbacks}")
    return {
        "tokens": tokens,
        "hidden": HIDDEN,
        "intermediate": DENSE_INTERMEDIATE,
        "label": "dense MLP the width of the 8 selected experts (8 * 704), not a checkpoint size",
        "compile_and_first_run_s": compile_seconds,
        **timed,
    }


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--intermediates", nargs="+", type=int, default=[704, 176])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--output", type=Path, help="incremental JSON results file")
    args = parser.parse_args()
    if not args.tokens or any(n < 1 for n in args.tokens):
        parser.error("--tokens must be positive")
    if not args.intermediates or any(n < 1 for n in args.intermediates):
        parser.error("--intermediates must be positive")
    if args.iterations < 1 or args.profile_iters < 1 or args.warmup_iters < 0:
        parser.error("iterations/profile-iters must be positive and warmup-iters non-negative")

    torch.set_num_threads(int(os.environ["SPYRE_NUM_CPUS"]))
    payload: dict[str, Any] = {
        "model": "Gemma 4 MoE prefill layer",
        "default_decision": (
            "SPYRE_MOE_PACKED_PREFILL stays 0 until packed median wall time on this "
            "card is lower than persistent. The host slot sync is inside the packed number."
        ),
        "torch_spyre_followups": [
            "fp16 and bias in torch._grouped_mm",
            "routing tables built on device, without a host sync",
            "grouped_mm overflow that does not raise (torch-spyre#5534)",
        ],
        "attention": "not measured; use scripts/microbench/spyre_attn_microbench.py",
        "measurements": [],
    }
    for intermediate in args.intermediates:
        layer, weights = _device_layer(intermediate, args.seed + intermediate)
        for tokens in args.tokens:
            result = _run_case(tokens, intermediate, layer, weights, args)
            payload["measurements"].append(result)
            if args.output:
                _write_output(args.output, payload)
            persistent_ms = result["persistent"]["latency_ms"]["median_ms"]
            packed_ms = result["packed"]["latency_ms"]["median_ms"]
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "intermediate": intermediate,
                        "persistent_median_ms": persistent_ms,
                        "packed_median_ms": packed_ms,
                        "packed_over_persistent": result["packed_over_persistent"],
                        "fallback_clean": True,
                        "device_profile": result["packed"].get("device_profile", "recorded"),
                    }
                ),
                flush=True,
            )
        del layer, weights
    if not args.skip_dense:
        dense = _run_dense(args.tokens[0], args)
        payload["dense_mlp"] = dense
        if args.output:
            _write_output(args.output, payload)
        moe = next(
            (
                item
                for item in payload["measurements"]
                if item["tokens"] == args.tokens[0] and item["intermediate"] == 704
            ),
            None,
        )
        share = None
        if moe is not None:
            moe_ms = moe["persistent"]["latency_ms"]["median_ms"]
            dense_ms = dense["latency_ms"]["median_ms"]
            share = moe_ms / (moe_ms + dense_ms) if moe_ms + dense_ms else None
        payload["moe_fraction_of_moe_plus_dense"] = share
        if args.output:
            _write_output(args.output, payload)
        print(
            json.dumps(
                {
                    "dense_mlp_tokens": dense["tokens"],
                    "dense_intermediate": DENSE_INTERMEDIATE,
                    "dense_median_ms": dense["latency_ms"]["median_ms"],
                    "moe_fraction_of_moe_plus_dense": share,
                    "layers": LAYERS,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
