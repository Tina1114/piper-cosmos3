"""Runtime audit callback for the Cosmos3-Edge Piper14 baseline."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.misc import get_local_tensor_if_DTensor


ACTION_HEAD_KEYS = ("action2llm", "llm2action", "action_modality_embed")


def _first_int(value: Any) -> int:
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("cannot extract an integer from an empty sequence")
        value = value[0]
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


class EdgeTrainingAuditCallback(Callback):
    """Fail-closed runtime evidence for loss, gradients, resume, and action inference."""

    def __init__(
        self,
        report_path: str,
        selected_keys: list[str],
        action_head_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.report_path = Path(report_path)
        self.selected_keys = tuple(selected_keys)
        self.action_head_keys = tuple(action_head_keys or ACTION_HEAD_KEYS)
        self.losses: list[float] = []
        self.action_losses: list[float] = []
        self.vision_losses: list[float] = []
        self.action_head_gradient_nonzero = False
        self.action_head_gradient_finite = True
        self.frozen_gradient_zero = True
        self.trainable_parameter_count = 0
        self.frozen_parameter_count = 0
        self.action_head_parameter_count = 0
        self.load_iteration = 0
        self.load_checkpoint_path: str | None = None
        self.last_data_batch: dict[str, Any] | None = None
        self.last_batch_size = 1

    @staticmethod
    def _net(model: torch.nn.Module) -> torch.nn.Module:
        net = getattr(model, "net", None)
        if net is None:
            raise RuntimeError("Edge audit requires model.net")
        return net

    def on_train_start(self, model: torch.nn.Module, iteration: int = 0) -> None:
        bad_trainable: list[str] = []
        bad_frozen: list[str] = []
        action_trainable: list[str] = []
        for name, parameter in self._net(model).named_parameters():
            selected = any(key in name for key in self.selected_keys)
            is_action = any(key in name for key in self.action_head_keys)
            if parameter.requires_grad:
                self.trainable_parameter_count += parameter.numel()
                if not selected:
                    bad_trainable.append(name)
                if is_action:
                    action_trainable.append(name)
                    self.action_head_parameter_count += parameter.numel()
            else:
                self.frozen_parameter_count += parameter.numel()
                if selected:
                    bad_frozen.append(name)

        if bad_trainable:
            raise RuntimeError(f"non-allowlisted parameters are trainable: {bad_trainable[:8]}")
        if bad_frozen:
            raise RuntimeError(f"allowlisted parameters are unexpectedly frozen: {bad_frozen[:8]}")
        if not action_trainable:
            raise RuntimeError("no trainable action-head parameters were found")

    def on_load_checkpoint_end(
        self,
        model: torch.nn.Module,
        iteration: int = 0,
        checkpoint_path: str | None = None,
    ) -> None:
        del model
        self.load_iteration = int(iteration)
        self.load_checkpoint_path = checkpoint_path

    def on_after_backward(self, model: torch.nn.Module, iteration: int = 0) -> None:
        # One real backward pass is enough to prove graph connectivity and the
        # optimizer freeze boundary. Loss finiteness is still checked every step.
        if self.action_head_gradient_nonzero and self.frozen_gradient_zero:
            return

        device: torch.device | None = None
        action_abs_sum: torch.Tensor | None = None
        action_finite = True
        frozen_has_grad = False
        for name, parameter in self._net(model).named_parameters():
            gradient = parameter.grad
            selected = any(key in name for key in self.selected_keys)
            if not selected and gradient is not None:
                frozen_has_grad = True
            if gradient is None or not any(key in name for key in self.action_head_keys):
                continue
            local_gradient = get_local_tensor_if_DTensor(gradient).detach()
            device = local_gradient.device
            current_sum = local_gradient.float().abs().sum()
            action_abs_sum = current_sum if action_abs_sum is None else action_abs_sum + current_sum
            action_finite = action_finite and bool(torch.isfinite(local_gradient).all().item())

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        metrics = torch.tensor(
            [
                float(action_abs_sum.item()) if action_abs_sum is not None else 0.0,
                0.0 if action_finite else 1.0,
                1.0 if frozen_has_grad else 0.0,
            ],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        self.action_head_gradient_nonzero |= bool(metrics[0].item() > 0.0)
        self.action_head_gradient_finite &= bool(metrics[1].item() == 0.0)
        self.frozen_gradient_zero &= bool(metrics[2].item() == 0.0)

    def on_training_step_batch_end(
        self,
        model: torch.nn.Module,
        data_batch: dict[str, Any],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, iteration
        loss_value = float(loss.detach().float().item())
        self.losses.append(loss_value)
        if "flow_matching_loss_action" in output_batch:
            self.action_losses.append(float(output_batch["flow_matching_loss_action"].detach().float().item()))
        if "flow_matching_loss_vision" in output_batch:
            self.vision_losses.append(float(output_batch["flow_matching_loss_vision"].detach().float().item()))
        self.last_batch_size = int(output_batch.get("batch_size", 1))
        self.last_data_batch = data_batch

    def _run_action_inference(self, model: torch.nn.Module) -> dict[str, Any] | None:
        if os.environ.get("EDGE_AUDIT_INFERENCE", "0") != "1":
            return None
        if self.last_data_batch is None:
            raise RuntimeError("cannot run audit inference without a training batch")

        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                samples = model.generate_samples_from_batch(
                    self.last_data_batch,
                    guidance=1.0,
                    seed=[17 + index for index in range(self.last_batch_size)],
                    n_sample=self.last_batch_size,
                    num_steps=int(os.environ.get("EDGE_AUDIT_INFERENCE_STEPS", "2")),
                    has_negative_prompt=False,
                )
        finally:
            model.train(was_training)

        if "action" not in samples or not samples["action"]:
            raise RuntimeError("audit inference did not return action samples")
        action = samples["action"][0].detach().float().squeeze(0)
        if action.ndim != 2:
            raise RuntimeError(f"expected a 2-D action sample, got {tuple(action.shape)}")
        raw_action_dim = _first_int(self.last_data_batch["raw_action_dim"])
        action = action[:, :raw_action_dim]
        # Whole-clip policy generation preserves the current state/action as
        # conditioning row 0. The deployable horizon is future rows 1..32.
        if action.shape[0] == 33:
            action = action[1:33]
        finite = bool(torch.isfinite(action).all().item())
        if tuple(action.shape) != (32, 14) or not finite:
            raise RuntimeError(
                f"invalid audit action output: shape={tuple(action.shape)}, finite={finite}"
            )
        return {
            "shape": list(action.shape),
            "finite": finite,
            "min": float(action.min().item()),
            "max": float(action.max().item()),
            "sampling_steps": int(os.environ.get("EDGE_AUDIT_INFERENCE_STEPS", "2")),
        }

    def on_train_end(self, model: torch.nn.Module, iteration: int = 0) -> None:
        inference = self._run_action_inference(model)
        expected_resume = int(os.environ.get("EDGE_AUDIT_EXPECT_RESUME_ITER", "0"))
        finite_losses = bool(self.losses) and all(math.isfinite(value) for value in self.losses)
        finite_modality_losses = (
            bool(self.action_losses)
            and bool(self.vision_losses)
            and all(math.isfinite(value) for value in self.action_losses + self.vision_losses)
        )
        nonzero_modality_losses = (
            any(value > 0.0 for value in self.action_losses)
            and any(value > 0.0 for value in self.vision_losses)
        )
        checkpoint_reload = self.load_iteration >= expected_resume if expected_resume > 0 else True
        status = "passed"
        failures: list[str] = []
        checks = {
            "finite_losses": finite_losses,
            "finite_modality_losses": finite_modality_losses,
            "nonzero_modality_losses": nonzero_modality_losses,
            "action_head_gradient_nonzero": self.action_head_gradient_nonzero,
            "action_head_gradient_finite": self.action_head_gradient_finite,
            "frozen_gradient_zero": self.frozen_gradient_zero,
            "checkpoint_reload": checkpoint_reload,
            "action_inference": inference is not None if os.environ.get("EDGE_AUDIT_INFERENCE", "0") == "1" else True,
        }
        for name, passed in checks.items():
            if not passed:
                failures.append(name)
        if failures:
            status = "failed"

        report = {
            "status": status,
            "failures": failures,
            "world_size": dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
            "iteration_end": int(iteration),
            "losses_observed": len(self.losses),
            "loss_min": min(self.losses) if self.losses else None,
            "loss_max": max(self.losses) if self.losses else None,
            "action_loss_min": min(self.action_losses) if self.action_losses else None,
            "action_loss_max": max(self.action_losses) if self.action_losses else None,
            "vision_loss_min": min(self.vision_losses) if self.vision_losses else None,
            "vision_loss_max": max(self.vision_losses) if self.vision_losses else None,
            **checks,
            "load_iteration": self.load_iteration,
            "load_checkpoint_path": self.load_checkpoint_path,
            "expected_resume_iteration": expected_resume,
            "action_inference_result": inference,
            "trainable_parameter_count": self.trainable_parameter_count,
            "frozen_parameter_count": self.frozen_parameter_count,
            "action_head_parameter_count": self.action_head_parameter_count,
            "selected_keys": list(self.selected_keys),
        }
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.report_path.with_suffix(self.report_path.suffix + ".tmp")
            temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            temporary_path.replace(self.report_path)
        if failures:
            raise RuntimeError(f"Edge runtime audit failed: {', '.join(failures)}")
