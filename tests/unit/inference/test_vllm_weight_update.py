from types import SimpleNamespace

import pytest
import torch

import prime_rl.inference.vllm.worker.nccl as nccl
from prime_rl.inference.vllm.worker.nccl import NCCLWeightUpdateWorker


def test_target_weight_update_guards_frozen_draft(monkeypatch):
    target = torch.nn.Linear(2, 2, bias=False)
    draft = torch.nn.Linear(2, 2, bias=False)
    runner = SimpleNamespace(
        model=target,
        model_config=object(),
        get_draft_model=lambda: draft,
    )
    worker = object.__new__(NCCLWeightUpdateWorker)
    worker.model_runner = runner
    worker.vllm_config = object()
    worker.quantize_in_weight_transfer = False
    worker.nccl_broadcast_receiver = SimpleNamespace(receive_state_dict=lambda: iter(()))

    def update_target(model, *_args):
        with torch.no_grad():
            model.weight.add_(1)

    monkeypatch.setattr(nccl, "load_weights_checkpoint_layerwise", update_target)
    worker.update_weights_from_path("")

    def update_target_and_draft(model, *_args):
        with torch.no_grad():
            model.weight.add_(1)
            draft.weight.add_(1)

    monkeypatch.setattr(nccl, "load_weights_checkpoint_layerwise", update_target_and_draft)
    with pytest.raises(AssertionError, match="Draft weights changed"):
        worker.update_weights_from_path("")
