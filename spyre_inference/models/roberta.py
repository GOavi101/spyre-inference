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

"""Spyre adaptations for vLLM RoBERTa / XLM-R pooling models.

RoBERTa re-exports BERT's ``_encode_token_type_ids`` /
``_decode_token_type_ids`` into its own module globals, so the side-buffer
adapter must be installed here as well as on ``bert``.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models.token_type_adapter import install_on

logger = init_logger(__name__)

_ROBERTA_POS_OFFSET_ATTR = "_spyre_cpu_position_offset"


def offset_roberta_position_ids(
    position_ids: torch.Tensor, padding_idx: int, device: torch.device
) -> torch.Tensor:
    """``position_ids + padding_idx + 1`` on CPU, then H2D as int64.

    Stock torch-spyre cannot schedule SDSC int32 add (warmup crash:
    ``0_add``), and int64 add CPU-falls-back through ``to_dtype``. Keep the
    offset off the device so position embedding is only a gather.
    """
    pos = convert(position_ids, device="cpu")
    pos = pos + int(padding_idx) + 1
    return convert(pos, device=device, dtype=torch.int64)


def _roberta_embedding_forward_spyre(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """vLLM ``RobertaEmbedding.forward`` with the padding offset on CPU."""
    from vllm.model_executor.models.roberta import _decode_token_type_ids

    token_type_ids = _decode_token_type_ids(input_ids)

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    position_embeddings = self.position_embeddings(
        offset_roberta_position_ids(
            position_ids, self.padding_idx, input_ids.device
        )
    )
    token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embeddings = inputs_embeds + token_type_embeddings + position_embeddings
    return self.LayerNorm(embeddings)


def _install_cpu_position_offset() -> None:
    from vllm.model_executor.models.roberta import RobertaEmbedding

    if getattr(RobertaEmbedding.forward, _ROBERTA_POS_OFFSET_ATTR, False):
        return
    RobertaEmbedding.forward = _roberta_embedding_forward_spyre
    setattr(RobertaEmbedding.forward, _ROBERTA_POS_OFFSET_ATTR, True)
    logger.info(
        "Spyre: RoBERTa position_ids + padding_idx offset runs on CPU "
        "(SDSC cannot schedule integer add)"
    )


def install_spyre_patches() -> None:
    """Install token_type adapter and CPU RoBERTa position offset."""
    from vllm.model_executor.models import roberta

    if not hasattr(roberta, "_encode_token_type_ids") or not hasattr(
        roberta, "_decode_token_type_ids"
    ):
        logger.debug("Spyre: RoBERTa module has no token_type helpers; skipping adapter")
    else:
        install_on(roberta)
        logger.info(
            "Spyre: RoBERTa token_type_ids use side-buffer adapter "
            "(skip vLLM bit-pack; torch-spyre#3509)"
        )

    _install_cpu_position_offset()
