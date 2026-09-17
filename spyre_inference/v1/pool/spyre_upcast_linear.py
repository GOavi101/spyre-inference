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

"""Classifier GEMM: staggered-K on Spyre, host ``F.linear`` on CPU.

Spyre has fp32 add/mul/sum but not ``F.linear`` / batchmatmul
(torch-spyre#1794). BERT/RoBERTa sequence heads are small-N GEMMs
(``[M, K] @ [N, K]ᵀ``). Keep weights fp16 with K innermost so ``.float()``
becomes DL16_T0_FP32 on K; broadcast mul then ``sum`` over K needs no
fp32 ReStickifyOpHBM. ``.to(fp16)`` destaggers the sparse fp32 output
(torch-spyre#2971) before D2H.

Bias is added on the host: 1-D ``[N]`` cannot restick onto ``[M, N]``.
After that D2H, later Linears (Roberta ``out_proj``) see CPU activations
and use ``F.linear`` — staggered-K materializes ``[M, N, K]`` and loses
to BLAS on the host.

Do not use this for vocab-sized heads: the Spyre path materializes
``[M, N, K]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from spyre_inference.custom_ops.utils import convert

# Stick width (fp16 elements). K must be in-stick for DL16_T0_FP32 on K.
_STICK = 64
# Label-like out dim, or hidden-to-hidden on a pooled vector (BertPooler dense).
_MAX_LABEL_OUT = 256
_MAX_SQUARE_OUT = 1024

_CompiledFn = Callable[..., torch.Tensor]
_compiled_kernels: dict[Callable[..., torch.Tensor], _CompiledFn] = {}


def _compile_if_spyre(kernel: _CompiledFn, device_type: str) -> _CompiledFn:
    if device_type != "spyre":
        return kernel
    compiled = _compiled_kernels.get(kernel)
    if compiled is None:
        compiled = cast(_CompiledFn, torch.compile(kernel, dynamic=False))
        _compiled_kernels[kernel] = compiled
    return compiled


def matmul_with_upcast(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """fp32 matmul via the DL16_T0_FP32 staggered-K path.

    ``x`` is ``[M, K]`` fp16 (K innermost), ``y`` is ``[N, K]`` fp16.
    ``.float()`` yields fp32 DL16_T0_FP32 on K, so
    ``x.unsqueeze(1) * y.unsqueeze(0)`` needs no fp32 ReStickifyOpHBM.
    ``sum(dim=-1)`` reduces in-stick K to ``[M, N]``.
    """
    x_fp32 = x.float()
    y_fp32 = y.float()
    product = x_fp32.unsqueeze(1) * y_fp32.unsqueeze(0)
    return product.sum(dim=-1)


def _upcast_linear_kernel(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Staggered-K GEMM, destaggered to ``x.dtype`` (torch-spyre#2971)."""
    out = matmul_with_upcast(x.contiguous(), weight.contiguous())
    return out.to(dtype=x.dtype)


def upcast_linear_supported(linear: nn.Linear) -> bool:
    """True when ``[M, N, K]`` is a BERT/RoBERTa-scale classifier / pooler dense."""
    if linear.weight.dtype != torch.float32:
        return False
    k = int(linear.in_features)
    n = int(linear.out_features)
    if k % _STICK != 0 or k <= 0 or n <= 0:
        return False
    if n <= _MAX_LABEL_OUT:
        return True
    return n == k and n <= _MAX_SQUARE_OUT


def _tensor_on(t: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Move ``t`` to ``device`` via ``convert`` when Spyre is involved."""
    target = torch.device(device) if isinstance(device, str) else device
    if t.device.type == target.type:
        return t
    if t.device.type == "spyre" or target.type == "spyre":
        return convert(t, target)
    return t.to(target)


def _param_to_fp16(data: torch.Tensor, device: torch.device) -> nn.Parameter:
    if data.device.type == "spyre" or device.type == "spyre":
        if data.dtype != torch.float16:
            data = convert(data, "cpu", torch.float16)
        return nn.Parameter(convert(data, device, torch.float16))
    return nn.Parameter(data.to(device=device, dtype=torch.float16))


class SpyreUpcastLinear(nn.Module):
    """Spyre: staggered-K mul+sum; CPU: ``F.linear``. Destagger, D2H, host bias."""

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        # Host fp32 copy for out_proj after dense D2Hs (and CPU-only tests).
        self._host_weight: torch.Tensor | None = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, device: torch.device) -> SpyreUpcastLinear:
        weight = _param_to_fp16(linear.weight.data, device)
        # 1-D [N] cannot restick onto [M, N] on Spyre; keep bias on the host.
        bias = None
        if linear.bias is not None:
            bias = nn.Parameter(linear.bias.data.detach().cpu().to(dtype=torch.float16))
        return cls(weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x2 = x.reshape(-1, orig[-1])
        if x2.device.type == "spyre":
            weight = self.weight
            if weight.device.type != "spyre":
                weight = _tensor_on(weight, x2.device)
            out = _compile_if_spyre(_upcast_linear_kernel, "spyre")(x2, weight)
            out = convert(out, "cpu")
            if self.bias is not None:
                out = out + _tensor_on(self.bias, out.device).to(dtype=out.dtype)
            return out.reshape(*orig[:-1], self.out_features)

        # Staggered-K materializes [M, N, K]; host F.linear uses BLAS.
        if self._host_weight is None:
            w = self.weight.data
            if w.device.type == "spyre":
                w = convert(w, "cpu")
            self._host_weight = w.float()
        bias = None if self.bias is None else self.bias.float()
        out = F.linear(x2.float(), self._host_weight, bias).to(dtype=x2.dtype)
        return out.reshape(*orig[:-1], self.out_features)


def replace_linears_with_upcast(module: nn.Module, device: torch.device) -> tuple[nn.Module, int]:
    """Replace supported ``nn.Linear`` children (and ``module`` itself) in place."""
    if isinstance(module, SpyreUpcastLinear):
        return module, 0
    if isinstance(module, nn.Linear) and upcast_linear_supported(module):
        return SpyreUpcastLinear.from_linear(module, device), 1
    replaced = 0
    for name, child in list(module.named_children()):
        new_child, n = replace_linears_with_upcast(child, device)
        if new_child is not child:
            setattr(module, name, new_child)
        replaced += n
    return module, replaced
