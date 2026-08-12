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

"""Spyre product embed tests vs cached HF refs and reranker smoke tests.

Covers both modeling backends:
- ``model_impl="vllm"`` — native in-tree Bert/RoBERTa
- ``model_impl="transformers"`` — hf-adapters Transformers wrappers (#504)

Multi-prompt embed closeness is covered by upstream
``tests/models/language/pooling/test_embedding.py`` (``check_embeddings_close``).

Regenerate embed refs: ``python tests/data/generate_encoder_embed_refs.py``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from vllm import LLM

EMBEDDING_MODELS = [
    "ibm-granite/granite-embedding-125m-english",
    "ibm-granite/granite-embedding-278m-multilingual",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/all-roberta-large-v1",
]

# vLLM routes RobertaForMaskedLM through as_embedding_model(TransformersForCausalLM),
# which leaves meta StageMissingLayer heads the Spyre device cannot copy. Native
# in-tree RobertaModel handles it, so keep the model on the vllm impl only.
NO_HF_ADAPTERS_EMBEDDING_MODELS = {"sentence-transformers/all-roberta-large-v1"}

# Cross-encoder reranker smoke (classify / score path). One model is enough —
# both BGE variants share XLMRobertaForSequenceClassification.
RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
]

# Native in-tree vs hf-adapters Transformers backend (same seam as CausalLM).
MODEL_IMPLS = [
    pytest.param("vllm", id="native"),
    pytest.param("transformers", id="hf_adapters"),
]

# Match upstream check_embeddings_close(tol=1e-2).
COSINE_MIN = 0.99

_REF_PATH = Path(__file__).parent / "data" / "encoder_embed_refs.json"
_REFERENCES: dict = json.loads(_REF_PATH.read_text()) if _REF_PATH.exists() else {}


def _cosine(a: list[float], b: list[float]) -> float:
    return F.cosine_similarity(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        dim=0,
    ).item()


def _assert_backend(llm: LLM, model_impl: str) -> None:
    using_transformers = llm.llm_engine.model_config.using_transformers_backend()
    if model_impl == "transformers":
        assert using_transformers
    else:
        assert not using_transformers


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model_impl", MODEL_IMPLS)
@pytest.mark.parametrize("model", EMBEDDING_MODELS)
def test_encoder_embed_models(model: str, model_impl: str) -> None:
    """Spyre embeddings match cached HF references within cosine tolerance."""
    if model_impl == "transformers" and model in NO_HF_ADAPTERS_EMBEDDING_MODELS:
        pytest.skip(f"{model} is not supported on the hf-adapters backend")

    ref = _REFERENCES.get(model)
    if ref is None:
        pytest.skip(f"No HF ref for {model}; run tests/data/generate_encoder_embed_refs.py")

    prompts = ref["prompts"]
    llm = LLM(
        model=model,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
        model_impl=model_impl,
    )
    _assert_backend(llm, model_impl)
    outputs = llm.embed(prompts)
    assert len(outputs) == len(prompts)

    for prompt, out, ref_emb in zip(prompts, outputs, ref["embeddings"]):
        emb = out.outputs.embedding
        assert len(emb) == len(
            ref_emb
        ), f"{model}: dim mismatch {len(emb)} vs cached {len(ref_emb)}"
        assert all(math.isfinite(x) for x in emb)
        sim = _cosine(emb, ref_emb)
        assert (
            sim >= COSINE_MIN
        ), f"{model}: cosine {sim:.4f} < {COSINE_MIN} vs cached HF reference for prompt {prompt!r}"


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model_impl", MODEL_IMPLS)
@pytest.mark.parametrize("model", RERANKER_MODELS)
def test_encoder_rerank_models(model: str, model_impl: str) -> None:
    """Load reranker and return one finite score via LLM.score()."""
    llm = LLM(
        model=model,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
        model_impl=model_impl,
    )
    _assert_backend(llm, model_impl)
    scores = llm.score("What is Spyre?", "An IBM AI accelerator.")
    assert len(scores) == 1
    assert math.isfinite(scores[0].outputs.score)
