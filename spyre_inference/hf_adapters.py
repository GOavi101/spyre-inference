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

"""Drop-in replacements for vLLM's Transformers backend using hf-adapters.

Registers as drop-in replacements for vLLM's Transformers modeling backend
classes when the Spyre platform is active.  vLLM's stock Transformers backend
handles model creation, weight loading, attention routing, KV cache /
pooling, scheduling, and forward execution.  Spyre OOT layers
(SpyreRMSNorm, SpyreSiluAndMul, SpyreLinears, etc.) are applied automatically
at instantiation time.

Activated when ``model_impl="transformers"`` on the Spyre platform via
``register_hf_adapters()``.

- CausalLM: matmul-based RoPE (``apply_rope_matmul``) after weight load.
- Embedding / sequence-classification: hf-adapters ``prepare_for_spyre`` +
  ``prefill_encoder`` after weight load (same driver as ``st_backend``).
  vLLM's flat token run is unpacked into a right-padded ``[B, L]`` batch so
  each request keeps its own positions and cannot attend across boundaries.

Encoder arches that vLLM routes through ``as_embedding_model(CausalLM)``
(e.g. ``RobertaForMaskedLM`` on ``all-roberta-large-v1``) are not supported
here; run those with ``model_impl="vllm"``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from types import ModuleType
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    PrecomputedRotaryEmbedding,
    SpyreNoAdapterError,
    SpyreUnsupportedModelError,
    apply_rope_matmul,
    get_backbone,
    prefill_encoder,
)
from vllm.logger import init_logger
from vllm.model_executor.models.transformers import (
    TransformersEmbeddingModel,
    TransformersForCausalLM,
    TransformersForSequenceClassification,
)
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class _SpyreRotaryEmbedding(nn.Module):
    """Drop-in for HF RotaryEmbedding using the same approach followed by hf-adapters.

    Returns ``(rotation_matrices, None)`` matching HF's ``(cos, sin)`` API.
    The patched ``apply_rotary_pos_emb`` uses ``apply_rope_matmul`` with the
    rotation matrices and ignores the second element.
    """

    def __init__(self, pre):
        super().__init__()
        self._pre = pre

    def _apply(self, fn, recurse=True):
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        return self._pre(x, position_ids), None


def _qk_expand_matrix(orig_hd: int, padded_hd: int) -> torch.Tensor:
    """Interleaved expand matrix for Q/K (RoPE-compatible half-split)."""
    half, phalf = orig_hd // 2, padded_hd // 2
    m = torch.zeros(orig_hd, padded_hd)
    m[:half, :phalf] = torch.eye(half, phalf)
    m[half:, phalf:] = torch.eye(half, phalf)
    return m


def _make_spyre_apply_rotary(original_fn, qk_expand=None):
    """Replace apply_rotary_pos_emb with matmul-based RoPE.

    When *qk_expand* is provided (head_dim/2 is not stick-aligned), Q/K are
    temporarily padded into the stick-aligned dimension for the rotation,
    then contracted back to the original size.
    """
    qk_contract = qk_expand.t().contiguous() if qk_expand is not None else None
    _cached: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def wrapper(q, k, cos, sin=None, *args, **kwargs):
        if qk_expand is not None:
            assert qk_contract is not None  # set together with qk_expand above
            dev = q.device
            if dev not in _cached:
                _cached[dev] = (
                    qk_expand.to(device=dev, dtype=q.dtype),
                    qk_contract.to(device=dev, dtype=q.dtype),
                )
            exp, con = _cached[dev]
            q = torch.matmul(q, exp)
            k = torch.matmul(k, exp)

        q, k = apply_rope_matmul(q, cos), apply_rope_matmul(k, cos)

        if qk_expand is not None:
            q = torch.matmul(q, con)
            k = torch.matmul(k, con)

        return q, k

    wrapper._spyre_patched = True
    return wrapper


def _linear_from_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    weight_is_transposed: bool = False,
) -> nn.Linear:
    """Build stock ``nn.Linear`` from a vLLM projection weight.

    Required because hf-adapters ``prepare_for_spyre`` / ``make_encoder_block``
    expect Bert-style ``query`` / ``key`` / ``value`` / FFN modules as
    ``nn.Linear``, while vLLM's Transformers backend leaves ``LinearBase``
    (and sometimes a fused ``qkv_proj``) in their place. Plain
    ``nn.Linear(...)`` defaults to float32; we keep the source weight's
    dtype/device so it stays fp16 under Spyre's platform dtype and matches
    ``prefill_encoder``'s fp16 mask.
    """
    w = weight.detach()
    if weight_is_transposed:
        w = w.t().contiguous()
    out_f, in_f = w.shape
    lin = nn.Linear(in_f, out_f, bias=bias is not None, dtype=w.dtype, device=w.device)
    with torch.no_grad():
        lin.weight.copy_(w)
        if bias is not None:
            lin.bias.copy_(bias.detach().to(dtype=w.dtype, device=w.device))
    lin.weight.requires_grad_(False)
    if lin.bias is not None:
        lin.bias.requires_grad_(False)
    return lin


def _replace_linear_base_with_nn_linear(root: nn.Module) -> int:
    """Undo vLLM's ``LinearBase`` swap so hf-adapters sees stock ``nn.Linear``.

    vLLM's Transformers backend replaces HF linears with ``LinearBase`` (TP /
    quant). That is correct for the native path, but this wrapper then calls
    hf-adapters ``prepare_for_spyre``, which was written for unmodified HF
    models: ``pad_attention_heads_simple`` reads ``proj.weight`` as
    ``[out, in]`` and rebuilds ``nn.Linear``, and ``make_encoder_block`` closes
    over plain ``q_proj(x)`` / FFN modules. Converting back is the bridge that
    lets us reuse that code instead of reimplementing encoder compile against
    ``LinearBase``.
    """
    from vllm.model_executor.layers.linear import LinearBase

    n = 0
    for name, child in list(root.named_children()):
        if isinstance(child, LinearBase):
            weight = getattr(child, "weight", None)
            if weight is None or weight.dim() != 2:
                continue
            setattr(root, name, _linear_from_weight(weight, getattr(child, "bias", None)))
            n += 1
        else:
            n += _replace_linear_base_with_nn_linear(child)
    return n


def _restore_bert_qkv_as_nn_linear(hf_model: nn.Module) -> int:
    """Rematerialize Bert ``query`` / ``key`` / ``value`` after the QKV split.

    The weight split itself is ``analyze_and_unfuse`` (``torch.split`` into
    ``q_weight`` / ``k_weight`` / ``v_weight``). That still leaves a single
    ``qkv_proj`` module; hf-adapters ``prepare_for_spyre`` indexes
    ``attn.query`` / ``key`` / ``value``, so this step only wraps the already-
    split parts as stock ``nn.Linear`` under those names. If unfuse was skipped
    (e.g. quantized), fall back to the same ``torch.split`` on the fused weight.
    """
    from vllm.model_executor.layers.linear import QKVParallelLinear

    backbone = get_backbone(hf_model)
    encoder = getattr(backbone, "encoder", None)
    if encoder is None or not hasattr(encoder, "layer"):
        return 0

    restored = 0
    for layer in encoder.layer:
        attn = getattr(getattr(layer, "attention", None), "self", None)
        if attn is None or hasattr(attn, "query"):
            continue
        qkv = getattr(attn, "qkv_proj", None)
        if qkv is None:
            continue

        if isinstance(qkv, QKVParallelLinear) and getattr(qkv, "q_weight", None) is not None:
            # Post-analyze_and_unfuse: parts are already split, stored [in, out].
            attn.query = _linear_from_weight(
                qkv.q_weight, getattr(qkv, "q_bias", None), weight_is_transposed=True
            )
            attn.key = _linear_from_weight(
                qkv.k_weight, getattr(qkv, "k_bias", None), weight_is_transposed=True
            )
            attn.value = _linear_from_weight(
                qkv.v_weight, getattr(qkv, "v_bias", None), weight_is_transposed=True
            )
        elif getattr(qkv, "weight", None) is not None:
            # Unfuse skipped — same row-wise split analyze_and_unfuse uses.
            w = qkv.weight.detach()
            if isinstance(qkv, QKVParallelLinear):
                sizes = [
                    qkv.num_heads * qkv.head_size,
                    qkv.num_kv_heads * qkv.head_size,
                    qkv.num_kv_heads * qkv.v_head_size,
                ]
            else:
                sizes = [w.shape[0] // 3] * 3
            wq, wk, wv = torch.split(w, sizes, dim=0)
            bias = getattr(qkv, "bias", None)
            if bias is None:
                bq = bk = bv = None
            else:
                bq, bk, bv = torch.split(bias.detach(), sizes, dim=0)
            attn.query = _linear_from_weight(wq, bq)
            attn.key = _linear_from_weight(wk, bk)
            attn.value = _linear_from_weight(wv, bv)
        else:
            continue

        if hasattr(attn, "qkv_proj"):
            delattr(attn, "qkv_proj")
        restored += 1
    return restored


def _flat_batch_lengths(
    positions: torch.Tensor,
    num_tokens: int,
    *,
    force_single: bool = False,
) -> list[int]:
    """Split vLLM's flat pooling batch into per-request token counts.

    Pooling prefills always start at position 0, so a restart marks a new
    request — including a batch of single-token requests whose positions are
    all zeros (``[0, 0, …]`` → ``[1, 1, …]``). Warmup leaves the positions
    buffer uncleared (also all zeros) but is not a real multi-request batch;
    the runner sets ``force_single`` for that case instead of guessing from
    the zeros.
    """
    if force_single or positions.numel() != num_tokens:
        return [num_tokens]
    starts = (positions == 0).nonzero().flatten().tolist()
    if not starts or starts[0] != 0:
        return [num_tokens]
    bounds = starts + [num_tokens]
    return [end - start for start, end in zip(bounds, bounds[1:])]


def _pack_flat_batch(
    ids: torch.Tensor,
    segments: torch.Tensor | None,
    lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Right-pad a flat token run into the ``[B, L]`` batch prefill_encoder wants.

    Mirrors what a tokenizer hands ``st_backend``: real tokens first, zeros
    after, and an attention mask carrying each row's true length so
    prefill_encoder numbers positions per row and masks across rows.
    """
    batched = pad_sequence(torch.split(ids, lengths), batch_first=True)
    mask = (torch.arange(batched.shape[1])[None, :] < torch.tensor(lengths)[:, None]).long()
    batched_segments = (
        None if segments is None else pad_sequence(torch.split(segments, lengths), batch_first=True)
    )
    return batched, mask, batched_segments


class _HfAdaptersEncoderMixin:
    """Shared prepare / forward for Transformers pooling wrappers."""

    # Set by subclasses; declared for type checkers (provided by Transformers base).
    model: nn.Module
    config: PretrainedConfig

    _sequence_classification: bool = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._adapter_module: ModuleType | None = None
        self._hf_adapters_encoder_ready = False
        # Set by SpyreModelRunner.warming_up_model around _dummy_run only.
        self._spyre_pooling_warmup = False

    @staticmethod
    def _resolve_encoder_adapter(
        config: PretrainedConfig,
        *,
        sequence_classification: bool = False,
    ) -> ModuleType:
        """Map HF config to an hf-adapters encoder module (AutoSpyre maps)."""
        import hf_adapters.auto_spyre_model as asm
        from hf_adapters.hf_common import assert_spyre_dimensions

        config_map = asm.CONFIG_TO_ADAPTER_MODULE_MAPPING
        arch_map = getattr(asm, "ARCH_TO_ADAPTER_MODULE_MAPPING", {})
        seq_map = getattr(asm, "SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING", None)
        mapping = seq_map if sequence_classification and seq_map is not None else config_map
        model_name = str(getattr(config, "_name_or_path", None) or "")

        def _check(module: ModuleType) -> ModuleType:
            # Decoder adapters are rejected before assert_spyre_dimensions so a
            # CausalLM config raises SpyreNoAdapterError (recoverable) rather than
            # SpyreUnsupportedModelError from the stick-alignment check.
            if not getattr(module, "_is_encoder_only", False):
                raise SpyreNoAdapterError(
                    f"hf-adapters module {module.__name__} is not encoder-only"
                )
            assert_spyre_dimensions(config, model_name=model_name)
            return module

        for arch in getattr(config, "architectures", None) or []:
            if arch in arch_map:
                return _check(arch_map[arch])

        cfg_type = type(config)
        if cfg_type in mapping:
            return _check(mapping[cfg_type])

        # Subclass fallbacks: skip non-encoder candidates rather than let _check
        # raise on one and abandon a later match.
        for cfg_cls, module in mapping.items():
            if isinstance(config, cfg_cls) and getattr(module, "_is_encoder_only", False):
                return _check(module)

        if sequence_classification:
            for cfg_cls, module in config_map.items():
                if isinstance(config, cfg_cls) and getattr(module, "_is_encoder_only", False):
                    return _check(module)

        raise SpyreNoAdapterError(
            f"No hf-adapters encoder for config type {type(config).__name__} "
            f"(architectures={getattr(config, 'architectures', None)})"
        )

    def prepare_hf_adapters_encoder(self) -> None:
        """Restore Bert layout, then ``prepare_for_spyre`` (before ``.to("spyre")``)."""
        if self._hf_adapters_encoder_ready:
            return
        try:
            adapter = self._resolve_encoder_adapter(
                self.config,
                sequence_classification=self._sequence_classification,
            )
        except (SpyreNoAdapterError, SpyreUnsupportedModelError):
            logger.warning(
                "No usable hf-adapters encoder for %s; keeping Transformers forward",
                type(self.config).__name__,
                exc_info=True,
            )
            return

        n_qkv = _restore_bert_qkv_as_nn_linear(self.model)
        n_lin = _replace_linear_base_with_nn_linear(self.model)
        # No dtype cast: Spyre's platform forces model_config.dtype=float16, and
        # _linear_from_weight preserves it through the LinearBase→nn.Linear
        # restore. That matches prefill_encoder's fp16 additive mask.
        adapter.prepare_for_spyre(self.model)
        self._adapter_module = adapter
        self._hf_adapters_encoder_ready = True
        logger.debug(
            "HfAdapters encoder ready (%s): restored %d QKV, converted %d LinearBase, "
            "%d compiled blocks",
            adapter.__name__,
            n_qkv,
            n_lin,
            len(getattr(self.model, "_spyre_compiled_blocks", [])),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if not self._hf_adapters_encoder_ready or self._adapter_module is None:
            return super().forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        if intermediate_tensors is not None or inputs_embeds is not None:
            raise NotImplementedError(
                "hf-adapters encoder path does not support PP intermediates " "or inputs_embeds"
            )
        if input_ids is None:
            raise ValueError("hf-adapters encoder forward requires input_ids")

        # Host-side on purpose: prefill_encoder's contract is CPU [B, L] ids /
        # mask (same device split as st_backend). It builds pad, position_ids,
        # and the bidirectional mask with host ops, then moves only the
        # backbone inputs to Spyre. The flat→padded pack (split / pad_sequence /
        # nonzero→tolist lengths) also has no usable Spyre path. The model
        # wrapper leaves ints on CPU when this encoder path is active, so these
        # .cpu() calls are no-ops in the steady state — not a per-step D2H.
        ids_cpu = input_ids.detach().cpu().flatten()
        # Segment ids matter for Bert-family cross-encoders; prefill_encoder
        # falls back to all-zeros (correct for single-sentence embedding).
        segments = kwargs.get("token_type_ids")
        if segments is not None:
            segments = segments.detach().cpu().flatten()

        # vLLM concatenates every scheduled request into one flat token run, so
        # split it back apart: collapsing it into a single sequence would let one
        # request attend to another and number positions across the boundary.
        # Warmup leaves positions all-zero (same pattern as N single-token
        # requests); the runner sets _spyre_pooling_warmup so we don't mis-split.
        lengths = _flat_batch_lengths(
            positions.detach().cpu(),
            ids_cpu.numel(),
            force_single=getattr(self, "_spyre_pooling_warmup", False),
        )
        if len(lengths) == 1:
            # Every position holds a real token: the Spyre runner disables cudagraph
            # and DP padding, the two sources of trailing pad in the flat batch.
            batched = ids_cpu[None, ...]
            mask = torch.ones(batched.shape, dtype=torch.long)
            batched_segments = None if segments is None else segments[None, ...]
        else:
            batched, mask, batched_segments = _pack_flat_batch(ids_cpu, segments, lengths)

        hidden = prefill_encoder(
            self._adapter_module._run_backbone_forward,
            self.model,
            batched,
            mask,
            token_type_ids=batched_segments,
        )
        if len(lengths) == 1:
            return hidden[0, ...]

        # Drop each row's right padding and re-flatten for the pooler, which
        # indexes hidden states by cumulative prompt length. Cropping runs on
        # CPU because aten.slice does not lower on Spyre; _SpyreModelWrapper
        # moves hidden states host-side anyway.
        hidden = hidden.to("cpu")
        return torch.cat([hidden[row, :length] for row, length in enumerate(lengths)], dim=0)


class HfAdaptersForCausalLM(TransformersForCausalLM):
    """TransformersForCausalLM wrapper to use HF adapters."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self._fix_generic_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.debug("HfAdaptersForCausalLM ready: %s", type(self.model).__name__)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights and patch rope."""
        result = super().load_weights(weights)
        self._patch_rope()
        return result

    @staticmethod
    def _fix_generic_config(vllm_config: VllmConfig) -> None:
        """Re-resolve generic PretrainedConfig produced by vLLM's
        config parser for some models where both config.json and params.json exists
        and force HF-format weight loading."""
        hf_config = vllm_config.model_config.hf_config
        if type(hf_config) is not PretrainedConfig:
            return

        model_id = vllm_config.model_config.hf_config_path or vllm_config.model_config.model
        try:
            resolved = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
                revision=vllm_config.model_config.revision,
            )
        except Exception:
            logger.warning("AutoConfig re-resolve failed for %s", model_id, exc_info=True)
            return

        skip = {"model_type", "_name_or_path", "transformers_version", "auto_map", "architectures"}
        for key, val in hf_config.to_dict().items():
            if key not in skip and val is not None:
                setattr(resolved, key, val)

        vllm_config.model_config.hf_config = resolved
        vllm_config.model_config.hf_text_config = resolved.get_text_config()
        if vllm_config.load_config.load_format in ("auto", "mistral"):
            vllm_config.load_config.load_format = "hf"
        logger.debug(
            "Re-resolved config: %s (model_type=%s), load_format=hf",
            type(resolved).__name__,
            resolved.model_type,
        )

    # TODO: Add support for models with fused QKV / gate_up projections
    # (e.g. Phi-3) by splitting them into separate modules with TP-aware
    # weight redistribution and partial-rotary dimension permutation.

    def _patch_rope(self):
        """Replace RoPE with matmul-based rotation.

        When head_dim/2 is not stick-aligned (not a multiple of BLOCK_SIZE),
        an expand/contract matrix pair is built and passed to the patched
        ``apply_rotary_pos_emb`` so that Q/K are temporarily padded into
        a stick-aligned dimension for the rotation, then contracted back.
        Attention and the KV cache keep using the original head_dim.
        """

        cfg = self.model.config
        orig_head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

        stick_aligned = ((orig_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)) * (
            2 * BLOCK_SIZE
        )
        padded_head_dim = stick_aligned if stick_aligned > orig_head_dim else None

        qk_exp = (
            _qk_expand_matrix(orig_head_dim, padded_head_dim)
            if padded_head_dim is not None
            else None
        )

        backbone = get_backbone(self.model)
        spyre_rope = PrecomputedRotaryEmbedding(
            backbone.rotary_emb,
            padded_head_dim=padded_head_dim,
        )

        spyre_rope_emb = _SpyreRotaryEmbedding(spyre_rope)
        backbone.rotary_emb = spyre_rope_emb

        _own_ids = {id(m) for m in spyre_rope_emb.modules()}

        patched_mods: set[int] = set()
        for name, module in self.model.named_modules():
            if id(module) in _own_ids:
                continue

            cls_name = module.__class__.__name__

            if cls_name.endswith("RotaryEmbedding") and not isinstance(
                module, _SpyreRotaryEmbedding
            ):
                pname, _, attr = name.rpartition(".")
                parent = self.model.get_submodule(pname) if pname else self.model
                setattr(parent, attr, _SpyreRotaryEmbedding(spyre_rope))
                continue

            if "Attention" not in cls_name:
                continue

            if not hasattr(module, "rotary_emb"):
                module.rotary_emb = _SpyreRotaryEmbedding(spyre_rope)

            mod = sys.modules.get(type(module).__module__)
            if mod is None or id(mod) in patched_mods:
                continue
            orig = getattr(mod, "apply_rotary_pos_emb", None)
            if orig is None or getattr(orig, "_spyre_patched", False):
                continue
            mod.apply_rotary_pos_emb = _make_spyre_apply_rotary(orig, qk_exp)
            patched_mods.add(id(mod))


class HfAdaptersEmbeddingModel(_HfAdaptersEncoderMixin, TransformersEmbeddingModel):
    """TransformersEmbeddingModel wrapper to use HF adapters."""

    _sequence_classification = False


class HfAdaptersForSequenceClassification(
    _HfAdaptersEncoderMixin, TransformersForSequenceClassification
):
    """TransformersForSequenceClassification wrapper to use HF adapters."""

    _sequence_classification = True


# vLLM's Transformers backend test checks ModelConfig.using_transformers_backend()
# compares _ModelInfo.architecture (set to model_cls.__name__) against the
# Transformers backend class name. Without this, subclass names fail that check.
HfAdaptersForCausalLM.__name__ = "TransformersForCausalLM"
HfAdaptersEmbeddingModel.__name__ = "TransformersEmbeddingModel"
HfAdaptersForSequenceClassification.__name__ = "TransformersForSequenceClassification"
