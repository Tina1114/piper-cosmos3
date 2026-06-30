#!/usr/bin/env python3
"""M5 dry-run adapter for Cosmos3-style Piper14 policy I/O.

This module is an engineering skeleton only. It preserves the project-level
Piper14 contract while the real Cosmos3 backbone is not wired in yet.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ACTION_SEMANTICS = "Piper14 absolute joint-position target"


class PlaceholderCosmos3Backbone(nn.Module):
    """Small feature extractor used only for M5 dry-run shape checks."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 5:
            batch, history, channels, height, width = image.shape
            if channels != 3:
                raise ValueError(f"Expected 3 image channels, got {tuple(image.shape)}")
            image = image.reshape(batch * history, channels, height, width)
            encoded = self.net(image)
            return encoded.reshape(batch, history, -1).mean(dim=1)
        if image.ndim == 4:
            if image.shape[1] != 3:
                raise ValueError(f"Expected [B,3,H,W] image input, got {tuple(image.shape)}")
            return self.net(image)
        raise ValueError(f"Image input must be [B,T,3,H,W] or [B,3,H,W], got {tuple(image.shape)}")


class Cosmos3Piper14Adapter(nn.Module):
    """Adapter enforcing Piper14 I/O around a placeholder Cosmos3-style body."""

    def __init__(
        self,
        action_horizon: int = 16,
        action_dim: int = 14,
        qpos_dim: int = 14,
        camera_names: tuple[str, ...] = CAMERA_NAMES,
        image_feature_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.qpos_dim = int(qpos_dim)
        self.camera_names = tuple(camera_names)
        self.action_semantics = ACTION_SEMANTICS
        self.real_cosmos3_backbone = False
        self.placeholder_backbone = True

        if self.action_dim != 14:
            raise ValueError(f"Piper14 adapter requires action_dim=14, got {self.action_dim}")
        if self.qpos_dim != 14:
            raise ValueError(f"Piper14 adapter requires qpos_dim=14, got {self.qpos_dim}")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

        self.backbones = nn.ModuleDict(
            {name: PlaceholderCosmos3Backbone(image_feature_dim) for name in self.camera_names}
        )
        fused_dim = len(self.camera_names) * image_feature_dim + self.qpos_dim
        self.piper14_action_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.action_horizon * self.action_dim),
        )

    def forward(self, images: dict[str, torch.Tensor], qpos: torch.Tensor) -> torch.Tensor:
        if qpos.ndim != 2 or qpos.shape[-1] != self.qpos_dim:
            raise ValueError(f"qpos must have shape [B,{self.qpos_dim}], got {tuple(qpos.shape)}")

        features = []
        for name in self.camera_names:
            if name not in images:
                raise KeyError(f"Missing camera input: {name}")
            image = images[name]
            if image.shape[0] != qpos.shape[0]:
                raise ValueError(
                    f"{name} batch size {image.shape[0]} does not match qpos batch {qpos.shape[0]}"
                )
            features.append(self.backbones[name](image))

        fused = torch.cat([*features, qpos], dim=-1)
        action = self.piper14_action_head(fused)
        return action.view(qpos.shape[0], self.action_horizon, self.action_dim)

    def contract_metadata(self) -> dict[str, Any]:
        return {
            "adapter": self.__class__.__name__,
            "real_cosmos3_backbone": self.real_cosmos3_backbone,
            "placeholder_backbone": self.placeholder_backbone,
            "action_semantics": self.action_semantics,
            "qpos_dim": self.qpos_dim,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "camera_names": list(self.camera_names),
            "output_contract": "[B,horizon,14]",
        }


def build_m5_adapter_from_config(config: dict[str, Any]) -> Cosmos3Piper14Adapter:
    training_cfg = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    horizon = int(model_cfg.get("action_horizon", training_cfg.get("action_horizon", 16)))
    return Cosmos3Piper14Adapter(
        action_horizon=horizon,
        action_dim=int(model_cfg.get("action_dim", 14)),
        qpos_dim=int(model_cfg.get("qpos_dim", 14)),
    )
