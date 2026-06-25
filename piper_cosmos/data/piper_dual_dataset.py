#!/usr/bin/env python3
"""Dataset wrapper for dual-Piper raw HDF5 episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hdf5_reader import HDF5EpisodeReader, default_instruction, find_hdf5_files, load_config


@dataclass(frozen=True)
class SampleIndex:
    episode_path: Path
    t: int


def _training_int(config: dict[str, Any], name: str, default: int) -> int:
    training = config.get("training", {})
    if isinstance(training, dict):
        value = training.get(name, default)
        if isinstance(value, int):
            return value
    return default


class PiperDualDataset:
    """Index raw HDF5 episodes into image histories and action chunks."""

    def __init__(
        self,
        data_root: Path | str,
        config_path: Path | str | None = None,
        history_frames: int | None = None,
        action_horizon: int | None = None,
        image_size: int = 224,
        stride: int | None = None,
        instruction: str | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.config = load_config(config_path)
        self.history_frames = int(
            history_frames
            if history_frames is not None
            else _training_int(self.config, "history_frames", 2)
        )
        self.action_horizon = int(
            action_horizon
            if action_horizon is not None
            else _training_int(self.config, "action_horizon", 16)
        )
        self.image_size = int(image_size)
        self.stride = int(stride if stride is not None else _training_int(self.config, "stride", 1))
        self.instruction = instruction if instruction is not None else default_instruction(self.config)

        if self.history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        self.episode_paths = find_hdf5_files(self.data_root)
        self.index = self._build_index()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_index = self.index[int(idx)]
        reader = HDF5EpisodeReader(sample_index.episode_path, self.config, self.image_size)
        t = sample_index.t
        return {
            "images": reader.read_image_history(t, self.history_frames),
            "qpos": reader.read_qpos(t),
            "action": reader.read_action_chunk(t, self.action_horizon),
            "instruction": self.instruction,
            "episode_path": str(sample_index.episode_path),
            "t": t,
        }

    def _build_index(self) -> list[SampleIndex]:
        samples: list[SampleIndex] = []
        first_t = self.history_frames - 1
        for episode_path in self.episode_paths:
            reader = HDF5EpisodeReader(episode_path, self.config, self.image_size)
            length = len(reader)
            last_t = length - self.action_horizon
            if last_t < first_t:
                continue
            for t in range(first_t, last_t + 1, self.stride):
                samples.append(SampleIndex(episode_path=episode_path, t=t))
        return samples
