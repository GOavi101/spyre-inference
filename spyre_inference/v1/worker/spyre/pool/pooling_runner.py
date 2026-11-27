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

"""MRV2: pooling (vllm/v1/worker/gpu/pool/pooling_runner.py:PoolingRunner).

Host cursor instead of a device one, no dim-0 slice of the hidden states, and the
encoder grid handled through the shared encoder_grid helpers. The grid comes
from the encoder model state's EncoderPlan, read through ``encoder_plan_source``;
load.py sets that, ``pooling_on_spyre`` and ``spyre_device`` once both the state and
the wrapper exist.
"""

from collections.abc import Callable, Collection

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
from vllm.v1.worker.gpu.states import RequestState

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._dynamic_width_encoder import DynamicWidthPlan, cls_rows
from spyre_inference.v1.pool.encoder_grid import pool_rows_for, unpad_encoder_grid
from spyre_inference.v1.pool.warmup import dummy_pooling_metadata
from spyre_inference.v1.worker.spyre.model_states.encoder_plan import EncoderPlan
from spyre_inference.v1.worker.spyre.model_wrapper import unwrap_model

_HOST = torch.device("cpu")


class SpyrePoolingRunner(PoolingRunner):
    """MRV2: PoolingRunner, pooling synchronously with a host cursor."""

    def __init__(self, model: nn.Module, vllm_config: VllmConfig) -> None:
        super().__init__(model, vllm_config)
        self.pooling_on_spyre = False
        self.spyre_device = torch.device("spyre")
        self.encoder_plan_source: Callable[[], EncoderPlan | None] = lambda: None

    @staticmethod
    def get_supported_tasks(model: nn.Module) -> list[PoolingTask]:
        # The _SpyreModelWrapper fails is_vllm_model: its __init__ takes no vllm_config.
        return PoolingRunner.get_supported_tasks(unwrap_model(model))

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[PoolerOutput, list[bool]]:
        plan = self.encoder_plan_source()
        if plan is not None and plan.dynamic_width is not None:
            return self._pool_cls_rows(hidden_states, plan.dynamic_width, input_batch, req_states)
        grid = plan.grid if plan is not None else None
        if not self.pooling_on_spyre:
            # FP32 linear heads keep the pooler on the host.
            hidden_states = unpad_encoder_grid(convert(hidden_states, "cpu"), grid)
            return super().pool(hidden_states, input_batch, req_states)

        # A host cursor: upstream's last_token_indices is an offset-1 view that isn't
        # stick-aligned on Spyre, so the ``*_gpu`` cursor fields are host tensors here.
        # No ``[:num_tokens]`` crop either: Spyre dim-0 slices are unsafe, and each
        # pooling method gathers its own rows.
        pooling_metadata = self._get_pooling_metadata(input_batch, req_states, _HOST)
        num_reqs = input_batch.num_reqs
        seq_lens_cpu = input_batch.seq_lens_cpu_upper_bound[:num_reqs]
        pooling_metadata.build_pooling_cursor(
            input_batch.num_scheduled_tokens, seq_lens_cpu, device=_HOST
        )
        # A no-op on Spyre, and the Spyre heads need a Spyre input.
        hidden_states = convert(hidden_states, self.spyre_device)
        hidden_states = pool_rows_for(hidden_states, self.model.pooler, pooling_metadata, grid)
        pooler_output = self.model.pooler(hidden_states, pooling_metadata)
        finished_mask = pooling_metadata.get_pooling_cursor().get_finished_mask()
        pooler_output = self.late_interaction_runner.postprocess_pooler_output(
            raw_pooler_output=pooler_output,
            pooling_params=pooling_metadata.pooling_params,
            req_ids=input_batch.req_ids,
            finished_mask=finished_mask,
        )
        return pooler_output, finished_mask

    def pools_only(self, input_batch: InputBatch, tasks: Collection[PoolingTask]) -> bool:
        """Whether every request of ``input_batch`` pools one of ``tasks``."""
        params = [self.pooling_params.get(i) for i in input_batch.idx_mapping_np.tolist()]
        return all(p is not None and p.task in tasks for p in params)

    def _pool_cls_rows(
        self,
        hidden_states: torch.Tensor,
        plan: DynamicWidthPlan,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[PoolerOutput, list[bool]]:
        """Pool the declared-width loops' output, ``CLS_ROWS`` rows per request.

        The CLS rows are picked on the host and handed to the pooler as rows
        ``0..n-1``. On Spyre they are padded to a power of two, so the pooler's gather
        and head see the row counts ``warm_pooler_row_widths`` warmed.
        """
        num_reqs = input_batch.num_reqs
        assert num_reqs == len(plan.query_lens), (num_reqs, len(plan.query_lens))
        rows = cls_rows(hidden_states, num_reqs)
        pooling_metadata = self._get_pooling_metadata(input_batch, req_states, _HOST)
        seq_lens_cpu = input_batch.seq_lens_cpu_upper_bound[:num_reqs]
        pooling_metadata.build_pooling_cursor(
            input_batch.num_scheduled_tokens, seq_lens_cpu, device=_HOST
        )
        cursor = pooling_metadata.get_pooling_cursor()
        cursor.first_token_indices_gpu = torch.arange(num_reqs, dtype=torch.int64)
        if self.pooling_on_spyre:
            padded = 1 << max(0, num_reqs - 1).bit_length()
            rows = convert(F.pad(rows, (0, 0, 0, padded - num_reqs)), self.spyre_device)
        pooler_output = self.model.pooler(rows, pooling_metadata)
        finished_mask = cursor.get_finished_mask()
        pooler_output = self.late_interaction_runner.postprocess_pooler_output(
            raw_pooler_output=pooler_output,
            pooling_params=pooling_metadata.pooling_params,
            req_ids=input_batch.req_ids,
            finished_mask=finished_mask,
        )
        return pooler_output, finished_mask

    def _dummy_pooler_run_task(
        self, hidden_states: torch.Tensor, task: PoolingTask
    ) -> PoolerOutput:
        if not self.pooling_on_spyre:
            return super()._dummy_pooler_run_task(convert(hidden_states, "cpu"), task)
        pooling_metadata = dummy_pooling_metadata(
            self.model.pooler,
            self.model_config,
            task,
            hidden_states.shape[0],
            self.max_num_reqs,
            _HOST,
        )
        return self.model.pooler(hidden_states, pooling_metadata)
