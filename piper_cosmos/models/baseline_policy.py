#!/usr/bin/env python3
"""Small multi-view CNN baseline for 14D Piper action chunks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
GRIPPER_DIMS = (6, 13)


class ImageEncoder(nn.Module):
    def __init__(self, input_channels: int, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleMultiViewCNNPolicy(nn.Module):
    """Encode 3 camera histories plus qpos and predict an action horizon."""

    def __init__(
        self,
        history_frames: int = 2,
        qpos_dim: int = 14,
        action_horizon: int = 16,
        action_dim: int = 14,
        image_feature_dim: int = 128,
        hidden_dim: int = 256,
        camera_names: tuple[str, ...] = CAMERA_NAMES,
    ) -> None:
        super().__init__()
        self.history_frames = int(history_frames)
        self.qpos_dim = int(qpos_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.camera_names = tuple(camera_names)
        input_channels = self.history_frames * 3

        self.encoders = nn.ModuleDict(
            {name: ImageEncoder(input_channels, image_feature_dim) for name in self.camera_names}
        )
        fused_dim = len(self.camera_names) * image_feature_dim + self.qpos_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.action_horizon * self.action_dim),
        )

    def forward(self, images: dict[str, torch.Tensor], qpos: torch.Tensor) -> torch.Tensor:
        features = []
        for name in self.camera_names:
            image = images[name]
            if image.ndim != 5:
                raise ValueError(f"{name} must have shape [B,T,3,H,W], got {tuple(image.shape)}")
            batch, history, channels, height, width = image.shape
            if history != self.history_frames or channels != 3:
                raise ValueError(
                    f"{name} expected history={self.history_frames}, channels=3, got {tuple(image.shape)}"
                )
            encoded = self.encoders[name](image.reshape(batch, history * channels, height, width))
            features.append(encoded)
        fused = torch.cat([*features, qpos], dim=-1)
        action = self.head(fused)
        return action.view(-1, self.action_horizon, self.action_dim)


def baseline_action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gripper_weight: float = 2.0,
    smoothness_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    action_mse = F.mse_loss(pred, target)
    gripper_mse = F.mse_loss(pred[..., list(GRIPPER_DIMS)], target[..., list(GRIPPER_DIMS)])
    if pred.shape[1] > 1:
        pred_delta = pred[:, 1:] - pred[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        smoothness_loss = F.mse_loss(pred_delta, target_delta)
    else:
        smoothness_loss = pred.new_tensor(0.0)
    total = action_mse + gripper_weight * gripper_mse + smoothness_weight * smoothness_loss
    metrics: dict[str, float] = {
        "loss": float(total.detach().cpu()),
        "action_mse": float(action_mse.detach().cpu()),
        "gripper_mse": float(gripper_mse.detach().cpu()),
        "smoothness_loss": float(smoothness_loss.detach().cpu()),
    }
    return total, metrics


def build_policy_from_config(config: dict[str, Any]) -> SimpleMultiViewCNNPolicy:
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    return SimpleMultiViewCNNPolicy(
        history_frames=int(model_cfg.get("history_frames", dataset_cfg.get("history_frames", 2))),
        qpos_dim=int(model_cfg.get("qpos_dim", 14)),
        action_horizon=int(model_cfg.get("action_horizon", dataset_cfg.get("action_horizon", 16))),
        action_dim=int(model_cfg.get("action_dim", 14)),
        image_feature_dim=int(model_cfg.get("image_feature_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
    )
