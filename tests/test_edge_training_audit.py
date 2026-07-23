from __future__ import annotations

import json

import torch

from piper_cosmos.cosmos3.edge_training_audit import EdgeTrainingAuditCallback


class _AuditNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action2llm = torch.nn.Linear(2, 2)
        self.moe_gen = torch.nn.Linear(2, 2)
        self.reasoner = torch.nn.Linear(2, 2)
        self.reasoner.requires_grad_(False)


class _AuditModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = _AuditNet()

    def generate_samples_from_batch(self, data_batch, **kwargs):
        del data_batch, kwargs
        return {"action": [torch.ones(32, 64)]}


def _callback(tmp_path) -> EdgeTrainingAuditCallback:
    return EdgeTrainingAuditCallback(
        report_path=str(tmp_path / "audit.json"),
        selected_keys=["action2llm", "moe_gen"],
        action_head_keys=["action2llm"],
    )


def test_runtime_audit_records_real_gradient_and_frozen_boundary(tmp_path):
    model = _AuditModel()
    callback = _callback(tmp_path)
    callback.on_train_start(model)
    loss = model.net.action2llm(torch.ones(1, 2)).sum() + model.net.moe_gen(torch.ones(1, 2)).sum()
    loss.backward()
    callback.on_after_backward(model)
    callback.on_training_step_batch_end(
        model,
        {"raw_action_dim": [torch.tensor(14)]},
        {
            "flow_matching_loss_action": torch.tensor(0.5),
            "flow_matching_loss_vision": torch.tensor(1.5),
        },
        loss.detach(),
    )
    callback.on_train_end(model, iteration=1)

    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["status"] == "passed"
    assert report["finite_losses"] is True
    assert report["action_head_gradient_nonzero"] is True
    assert report["frozen_gradient_zero"] is True


def test_runtime_audit_resume_and_action_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGE_AUDIT_INFERENCE", "1")
    monkeypatch.setenv("EDGE_AUDIT_EXPECT_RESUME_ITER", "100")
    monkeypatch.setenv("EDGE_AUDIT_INFERENCE_STEPS", "2")
    model = _AuditModel()
    callback = _callback(tmp_path)
    callback.on_train_start(model)
    callback.on_load_checkpoint_end(model, iteration=100, checkpoint_path="/smoke/iter_000000100")
    loss = model.net.action2llm(torch.ones(1, 2)).sum()
    loss.backward()
    callback.on_after_backward(model)
    callback.on_training_step_batch_end(
        model,
        {"raw_action_dim": [torch.tensor(14)]},
        {
            "flow_matching_loss_action": torch.tensor(0.5),
            "flow_matching_loss_vision": torch.tensor(1.5),
        },
        loss.detach(),
    )
    callback.on_train_end(model, iteration=101)

    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["status"] == "passed"
    assert report["checkpoint_reload"] is True
    assert report["action_inference_result"]["shape"] == [32, 14]
    assert report["action_inference_result"]["finite"] is True
