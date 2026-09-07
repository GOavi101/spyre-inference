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

"""Spyre FP8 linear: keep checkpoint FP8 weights, run compiled ``aten._scaled_mm``.

Forward (same graph as torch-spyre ``test_fp8_scaled_mm_cpu``):

    scale_a = amax(x) / 448                         # eager, outside compile
    y = _scaled_mm(qfp8ch(x), qfp8wt(W), scale_a, scale_b)   # FP16 out

Granite fused N (QKV 6144, gate_up 25600) and wide M compile as one GEMM
once torch-spyre includes the #4179 fix (PR #4235). Host M/N tiling is gone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import register_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
)
from vllm.platforms import PlatformEnum

logger = init_logger(__name__)

_REGISTERED = False
FP8_E4M3FN_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _activation_scale(x: torch.Tensor, per_token: bool) -> torch.Tensor:
    if per_token:
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        return (amax / FP8_E4M3FN_MAX).to(dtype=torch.float16)
    amax = x.abs().amax().clamp(min=1e-12)
    return (amax / FP8_E4M3FN_MAX).to(dtype=torch.float16).reshape(1)


@torch.compile(backend="inductor", dynamic=False)
def _compiled_fp8_scaled_mm(
    x: torch.Tensor,
    scale_a: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # qfp8wt layout is assigned in this graph; do not pre-quantize weights.
    x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(
        x,  # ty: ignore[invalid-argument-type]
        scale_a,  # ty: ignore[invalid-argument-type]
    )
    w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(
        weight,  # ty: ignore[invalid-argument-type]
        weight_scale,  # ty: ignore[invalid-argument-type]
    )
    return torch.ops.aten._scaled_mm(
        x_fp8,  # ty: ignore[invalid-argument-type]
        w_fp8,  # ty: ignore[invalid-argument-type]
        scale_a=scale_a,  # ty: ignore[invalid-argument-type]
        scale_b=weight_scale,  # ty: ignore[invalid-argument-type]
        bias=bias,  # ty: ignore[invalid-argument-type]
        out_dtype=torch.float16,  # ty: ignore[invalid-argument-type]
    )


def _fp8_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    per_token: bool,
) -> torch.Tensor:
    return _compiled_fp8_scaled_mm(x, _activation_scale(x, per_token), weight, weight_scale, bias)


def _fp16_weight_for_qfp8wt(
    weight: torch.Tensor, weight_scale: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """CPU float8 is not qfp8wt. Dequant once; the compiled graph re-quantizes."""
    if weight.dtype != torch.float8_e4m3fn:
        return weight
    w = weight.detach().cpu().to(torch.float16)
    s = weight_scale.detach().cpu()
    return (w * s).contiguous().to(device)


def _normalize_weight_scale(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    scale = weight_scale.detach().to(torch.float16)
    n_out = weight.shape[-1]
    if scale.numel() == 1:
        return scale.reshape(1)
    if scale.numel() != n_out:
        raise NotImplementedError(
            "SpyreFp8LinearKernel expects per-tensor [1] or per-channel "
            f"[N]={n_out} weight_scale, got shape {tuple(weight_scale.shape)}"
        )
    return scale.reshape(1, n_out)


class SpyreFp8LinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        gs = c.weight_quant_key.scale.group_shape
        if gs.is_per_tensor() or gs.is_per_channel():
            return True, None
        return False, "requires per-tensor or per-channel weight scales"

    def __init__(self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]) -> None:
        # Skip CUDA QuantFP8 in FP8ScaledMMLinearKernel.__init__.
        self._per_token_act = c.activation_quant_key.scale.group_shape.is_per_token()
        ScaledMMLinearKernel.__init__(self, c, layer_param_names)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = cast(torch.Tensor, layer.weight)
        weight_scale = cast(torch.Tensor, layer.weight_scale)
        scale = _normalize_weight_scale(weight, weight_scale)
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)

    # Not an untraceable op. The GEMM is already Dynamo/Inductor:
    # ``_compiled_fp8_scaled_mm`` (qfp8ch + qfp8wt + aten._scaled_mm).
    # ``recursive=False`` keeps that nested compile. This wrapper stays
    # eager because first-forward CPU float8→fp16 for qfp8wt is not
    # Spyre-graphable (torch-spyre #3506). Drop disable when checkpoint
    # float8 loads as qfp8wt.
    @torch._dynamo.disable(recursive=False)
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x

        w = getattr(layer, "_fp16_for_qfp8wt", None)
        if w is None or w.device != x2d.device:
            w = _fp16_weight_for_qfp8wt(
                cast(torch.Tensor, layer.weight),
                cast(torch.Tensor, layer.weight_scale),
                x2d.device,
            )
            layer._fp16_for_qfp8wt = w

        out = _fp8_mm(x2d, w, cast(torch.Tensor, layer.weight_scale), bias, self._per_token_act)
        if x.dim() > 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])
        return out

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # Required: FP8ScaledMMLinearKernel marks this abstract. Unused on Spyre.
        # Upstream Torch only overrides this hook because parent apply_weights
        # quantizes then calls it with already-FP8 A/B. We replace apply_weights
        # (in-graph qfp8ch/qfp8wt), so this is never entered. Do not wrap
        # _fp8_mm here: that helper expects FP16 x/W, not pre-quantized A/B.
        raise RuntimeError(
            "SpyreFp8LinearKernel runs only through apply_weights "
            "(qfp8ch/qfp8wt graph). apply_scaled_mm is unused."
        )


SpyreFp8DequantLinearKernel = SpyreFp8LinearKernel


def register_spyre_fp8_linear_kernel() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    register_linear_kernel(SpyreFp8LinearKernel, PlatformEnum.OOT, kernel_type="fp8")
    _REGISTERED = True
    logger.info("Registered SpyreFp8LinearKernel for PlatformEnum.OOT (aten._scaled_mm)")
    return True
