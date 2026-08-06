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

"""SpyreLogitsProcessor.to_host_logits."""

import torch

from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor


def test_to_host_logits_trims_to_org_vocab():
    logits = torch.randn(2, 4096, dtype=torch.float16)
    host = SpyreLogitsProcessor.to_host_logits(logits, org_vocab_size=3000)
    assert host.shape == (2, 3000)
    assert host.device == torch.device("cpu")
    torch.testing.assert_close(host, logits[:, :3000].cpu(), atol=0, rtol=0)


def test_to_host_logits_noop_when_already_trimmed():
    logits = torch.randn(2, 3000, dtype=torch.float16)
    host = SpyreLogitsProcessor.to_host_logits(logits, org_vocab_size=3000)
    assert host.shape == (2, 3000)
    torch.testing.assert_close(host, logits.cpu(), atol=0, rtol=0)
