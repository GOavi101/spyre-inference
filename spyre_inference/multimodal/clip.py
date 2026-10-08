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

"""CLIP LayerNorm workaround for Spyre.

``vision_model.pre_layrnorm``/``post_layernorm`` and
``text_model.final_layer_norm`` sit outside any compiled graph.
``vision_model``'s ``layer_norm1``/``layer_norm2`` do too: the vision tower is
excluded from per-block compile, and under ``--enforce-eager`` nothing is
compiled, so stock ``nn.LayerNorm`` takes torch-spyre's isolated decomposition
and crashes on the 768-wide activation (mixed element arrangement). All of
those are swapped to ``SpyreLayerNorm``.

The text tower's in-block norms stay stock. Its blocks contain decoder
attention, so per-block compile traces them and they never take that path.

Under ``--enforce-eager`` the vision block's residual add still fails: the
saved residual and the attention output do not share a layout, and Spyre
rejects that add. Those two operands are rebuilt in the default layout first.
That host round trip leaves an activation whose on-device ``index_select``
compiles an identity the scheduler cannot map, so eager CLS/LAST gathers
copy the rows to the host first. The compiled path is left alone.

Applied to the already-loaded model instance (weights included), so the
replacement ``SpyreLayerNorm`` here copies the original's already-loaded
weight/bias explicitly, rather than relying on a later ``load_weights()``
pass to populate them.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm

logger = init_logger(__name__)


def _to_spyre_layer_norm(ln: torch.nn.LayerNorm, device: torch.device) -> torch.nn.LayerNorm:
    new_ln = SpyreLayerNorm(
        list(ln.normalized_shape),
        eps=ln.eps,
        elementwise_affine=ln.elementwise_affine,
        bias=ln.bias is not None,
    ).to(device=device, dtype=ln.weight.dtype if ln.elementwise_affine else torch.float16)
    if ln.elementwise_affine:
        with torch.no_grad():
            new_ln.weight.copy_(ln.weight)
            if ln.bias is not None:
                new_ln.bias.copy_(ln.bias)
    return new_ln


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Swap CLIP's uncompiled LayerNorms for ``SpyreLayerNorm``, in place.

    The ``isinstance`` checks are a second line of defense on top of the
    ``model_type == "clip"`` dispatch gate in ``multimodal/__init__.py``: they
    keep this a no-op (rather than an ``AttributeError`` on ``normalized_shape``)
    for any boundary norm that isn't a plain ``nn.LayerNorm``.
    """
    text_model = getattr(model, "text_model", None)
    if text_model is not None:
        ln = getattr(text_model, "final_layer_norm", None)
        if isinstance(ln, torch.nn.LayerNorm):
            text_model.final_layer_norm = _to_spyre_layer_norm(ln, device)

    eager = _uncompiled()
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        pre_ln = getattr(vision_model, "pre_layrnorm", None)
        if isinstance(pre_ln, torch.nn.LayerNorm):
            vision_model.pre_layrnorm = _to_spyre_layer_norm(pre_ln, device)
        post_ln = getattr(vision_model, "post_layernorm", None)
        if isinstance(post_ln, torch.nn.LayerNorm):
            vision_model.post_layernorm = _to_spyre_layer_norm(post_ln, device)
        _swap_vision_block_norms(vision_model, device, eager)

    if eager:
        from spyre_inference.v1.pool.spyre_pooler import arm_eager_host_pool

        arm_eager_host_pool()
        logger.info_once("Spyre: eager CLIP CLS/LAST pooling gathers on the host.")

    logger.info_once(
        "Spyre: CLIP boundary LayerNorms use SpyreLayerNorm. "
        "Vision-block norms stay stock on a compiled server."
    )


def _uncompiled() -> bool:
    """True when this process is not tracing per-block graphs.

    ``apply()`` also runs from unit tests with no vLLM config context; treat that
    as compiled so the eager-only forward patch stays off.
    """
    try:
        from vllm.config import CompilationMode, get_cached_compilation_config

        return get_cached_compilation_config().mode is CompilationMode.NONE
    except Exception:
        return False


def _default_layout(x: torch.Tensor) -> torch.Tensor:
    """Rebuild ``x`` in the default layout. A no-op off Spyre."""
    if x.device.type != "spyre":
        return x
    from spyre_inference.custom_ops.utils import convert

    return convert(convert(x, "cpu").contiguous(), x.device)


def _patch_eager_residual(layer: torch.nn.Module) -> None:
    """Add the residual only after both operands share the default layout."""
    forward = getattr(layer, "forward", None)
    if forward is None or getattr(forward, "_spyre_residual_patched", False):
        return
    if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
        return

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = layer.layer_norm1(hidden_states)
        hidden_states, _ = layer.self_attn(hidden_states=hidden_states)
        hidden_states = _default_layout(residual) + _default_layout(hidden_states)

        residual = hidden_states
        hidden_states = layer.layer_norm2(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        return _default_layout(residual) + _default_layout(hidden_states)

    forward._spyre_residual_patched = True  # type: ignore[attr-defined]
    layer.forward = forward  # type: ignore[method-assign]


def _swap_vision_block_norms(
    vision_model: torch.nn.Module, device: torch.device, eager: bool
) -> None:
    """Replace vision-block norms only for an eager load.

    A compiled server keeps the stock norms. They already match Hugging Face
    there, and ``SpyreLayerNorm`` would install a different kernel on the
    compiled path. The vision tower is outside per-block compile, so an eager
    load takes the crashing ``aten.layer_norm`` decomposition instead.
    ``type is`` rather than ``isinstance``: ``SpyreLayerNorm`` subclasses
    ``nn.LayerNorm``.
    """
    if not eager:
        return
    encoder = getattr(vision_model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return
    for layer in layers:
        for name in ("layer_norm1", "layer_norm2"):
            ln = getattr(layer, name, None)
            if type(ln) is torch.nn.LayerNorm:
                setattr(layer, name, _to_spyre_layer_norm(ln, device))
        _patch_eager_residual(layer)
        logger.info_once(
            "Spyre: eager CLIP vision residual adds rebuild both operands in the default layout."
        )
