"""Dry-run RTC runtime primitives for Cosmos Piper14 deployment."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from piper_cosmos.data.hdf5_reader import (
    DEFAULT_ACTION_KEY,
    DEFAULT_QPOS_KEY,
    default_instruction,
    find_hdf5_files,
    hdf5_key,
    image_keys,
    load_config,
)


class PolicyClient(Protocol):
    def infer(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Return an action chunk with shape [T, action_dim]."""


class ObservationSource(Protocol):
    def read_observation(self, t: int) -> Mapping[str, Any]:
        """Read one RTC observation for control step ``t``."""


class ActionSink(Protocol):
    def send_action(self, t: int, action: np.ndarray) -> None:
        """Consume one selected action for control step ``t``."""


class RealTimeChunkingBuffer:
    """Thread-safe buffer that fuses overlapping action chunks online."""

    def __init__(self, chunk_size: int, exp_weight_factor: float = 0.5, debug: bool = False):
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        self.exp_weight_factor = float(exp_weight_factor)
        self.debug = bool(debug)
        self.control_t = 0
        self.chunks: dict[int, np.ndarray] = {}
        self.generation = 0
        self.lock = threading.Lock()

    def clear(self) -> None:
        with self.lock:
            self.control_t = 0
            self.chunks = {}
            self.generation += 1

    def set_control_time(self, control_t: int) -> None:
        with self.lock:
            self.control_t = int(control_t)

    def get_control_time(self) -> int:
        with self.lock:
            return self.control_t

    def get_generation(self) -> int:
        with self.lock:
            return self.generation

    def has_chunk(self, cursor: int) -> bool:
        with self.lock:
            return int(cursor) in self.chunks

    def enqueue(self, chunk: np.ndarray, cursor: int, generation: int | None = None) -> bool:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"Expected action chunk shape [T,D], got {chunk.shape}")
        cursor = int(cursor)
        with self.lock:
            if generation is not None and int(generation) != self.generation:
                if self.debug:
                    print(
                        f"[rtc-buffer] drop stale chunk cursor={cursor} "
                        f"generation={generation} current={self.generation}"
                    )
                return False
            self.chunks[cursor] = np.ascontiguousarray(chunk)
            return True

    def get_action(self, current_time: int) -> np.ndarray | None:
        current_time = int(current_time)
        with self.lock:
            relevant: dict[int, np.ndarray] = {}
            expired: list[int] = []
            before = sorted(self.chunks)
            for cursor, chunk in self.chunks.items():
                end = cursor + len(chunk)
                if cursor <= current_time < end:
                    relevant[cursor] = chunk[current_time - cursor]
                elif end <= current_time:
                    expired.append(cursor)
            for cursor in expired:
                del self.chunks[cursor]
            if self.debug:
                print(
                    f"[rtc-buffer] t={current_time} before={before} "
                    f"expired={sorted(expired)} after={sorted(self.chunks)}"
                )
            if not relevant:
                return None
            items = sorted(relevant.items(), key=lambda item: item[0])
            actions = np.asarray([action for _, action in items], dtype=np.float32)

        weights = np.exp(self.exp_weight_factor * np.arange(len(actions), dtype=np.float32))
        weights = (weights / weights.sum())[:, None]
        return np.ascontiguousarray((actions * weights).sum(axis=0).astype(np.float32))


@dataclass(frozen=True)
class Piper14RTCRuntimeConfig:
    action_dim: int = 14
    chunk_size: int = 32
    control_hz: float = 30.0
    max_steps: int = 100
    replan_interval: int = 1
    exp_weight_factor: float = 0.5
    sleep: bool = False
    prompt: str | None = None
    debug: bool = False


class RecordingActionSink:
    """Action sink for dry-run RTC tests; it records selected actions only."""

    def __init__(self) -> None:
        self.records: list[tuple[int, np.ndarray]] = []

    def send_action(self, t: int, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        self.records.append((int(t), np.ascontiguousarray(action.copy())))

    @property
    def actions(self) -> np.ndarray:
        if not self.records:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([action for _, action in self.records], axis=0).astype(np.float32, copy=False)


class Piper14RTCRuntime:
    """Synchronous dry-run RTC loop around a persistent Cosmos policy server."""

    def __init__(
        self,
        *,
        policy: PolicyClient,
        observation_source: ObservationSource,
        action_sink: ActionSink,
        config: Piper14RTCRuntimeConfig | None = None,
    ) -> None:
        self.policy = policy
        self.observation_source = observation_source
        self.action_sink = action_sink
        self.config = config or Piper14RTCRuntimeConfig()
        self.buffer = RealTimeChunkingBuffer(
            chunk_size=self.config.chunk_size,
            exp_weight_factor=self.config.exp_weight_factor,
            debug=self.config.debug,
        )
        self.selected_actions: list[np.ndarray] = []
        self.inference_chunks: list[tuple[int, np.ndarray, float]] = []
        self.starved_steps = 0

    def run(self) -> dict[str, Any]:
        self.buffer.clear()
        self.selected_actions = []
        self.inference_chunks = []
        self.starved_steps = 0

        max_steps = max(int(self.config.max_steps), 0)
        period = 1.0 / max(float(self.config.control_hz), 1e-6)
        for t in range(max_steps):
            started = time.perf_counter()
            self.step(t)
            if self.config.sleep:
                elapsed = time.perf_counter() - started
                time.sleep(max(0.0, period - elapsed))
        return self.report()

    def step(self, t: int) -> np.ndarray:
        t = int(t)
        self.buffer.set_control_time(t)
        if self._should_replan(t):
            self._infer_and_enqueue(t)

        action = self.buffer.get_action(t)
        if action is None:
            self.starved_steps += 1
            raise RuntimeError(f"No RTC action available at control step {t}")
        action = self._validate_action(action, t)
        self.action_sink.send_action(t, action)
        self.selected_actions.append(action.copy())
        return action

    def report(self) -> dict[str, Any]:
        actions = np.stack(self.selected_actions, axis=0) if self.selected_actions else np.zeros((0, self.config.action_dim), dtype=np.float32)
        inference_latencies = [latency for _, _, latency in self.inference_chunks]
        return {
            "steps": len(self.selected_actions),
            "num_inferences": len(self.inference_chunks),
            "starved_steps": int(self.starved_steps),
            "control_hz": float(self.config.control_hz),
            "chunk_size": int(self.config.chunk_size),
            "replan_interval": int(self.config.replan_interval),
            "selected_actions": summarize_array(actions),
            "inference_latency_s": summarize_numbers(inference_latencies),
        }

    def _should_replan(self, t: int) -> bool:
        interval = max(int(self.config.replan_interval), 1)
        return t % interval == 0 and not self.buffer.has_chunk(t)

    def _infer_and_enqueue(self, cursor: int) -> None:
        obs = dict(self.observation_source.read_observation(cursor))
        if self.config.prompt is not None:
            obs["prompt"] = self.config.prompt

        generation = self.buffer.get_generation()
        started = time.perf_counter()
        chunk = np.asarray(self.policy.infer(obs), dtype=np.float32)
        latency = time.perf_counter() - started
        chunk = self._validate_chunk(chunk, cursor)
        chunk = chunk[: self.config.chunk_size]
        if self.buffer.enqueue(chunk, cursor=cursor, generation=generation):
            self.inference_chunks.append((int(cursor), chunk.copy(), float(latency)))

    def _validate_chunk(self, chunk: np.ndarray, cursor: int) -> np.ndarray:
        if chunk.ndim != 2 or chunk.shape[1] != int(self.config.action_dim):
            raise ValueError(f"Expected action chunk [T,{self.config.action_dim}], got {chunk.shape} at cursor {cursor}")
        if chunk.shape[0] == 0:
            raise ValueError(f"Expected non-empty action chunk at cursor {cursor}")
        if not np.isfinite(chunk).all():
            raise ValueError(f"Policy returned non-finite action chunk at cursor {cursor}")
        return np.ascontiguousarray(chunk.astype(np.float32, copy=False))

    def _validate_action(self, action: np.ndarray, t: int) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (int(self.config.action_dim),):
            raise ValueError(f"Expected selected action [{self.config.action_dim}], got {action.shape} at step {t}")
        if not np.isfinite(action).all():
            raise ValueError(f"RTC selected non-finite action at step {t}")
        return np.ascontiguousarray(action)


class HDF5ObservationSource:
    """Replay one HDF5 episode as RTC observations without robot hardware."""

    def __init__(
        self,
        *,
        episode_path: Path | str,
        config_path: Path | str,
        prompt: str | None = None,
        loop: bool = False,
    ) -> None:
        self.episode_path = Path(episode_path)
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.image_keys = image_keys(self.config)
        self.qpos_key = hdf5_key(self.config, "qpos_key", DEFAULT_QPOS_KEY)
        self.action_key = hdf5_key(self.config, "action_key", DEFAULT_ACTION_KEY)
        self.prompt = prompt if prompt is not None else default_instruction(self.config) or "Assemble the mouse's battery."
        self.loop = bool(loop)
        self.length = self._episode_length()

    @classmethod
    def from_data_root(
        cls,
        *,
        data_root: Path | str,
        config_path: Path | str,
        episode: Path | str | None = None,
        prompt: str | None = None,
        loop: bool = False,
    ) -> "HDF5ObservationSource":
        episode_path = Path(episode) if episode is not None else resolve_hdf5_episode(Path(data_root))
        return cls(episode_path=episode_path, config_path=config_path, prompt=prompt, loop=loop)

    def read_observation(self, t: int) -> Mapping[str, Any]:
        import h5py  # type: ignore

        idx = self._resolve_index(t)
        images: dict[str, np.ndarray] = {}
        with h5py.File(self.episode_path, "r") as h5:
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                key = self.image_keys[name]
                frame = np.asarray(h5[key][idx])
                if frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError(f"{self.episode_path}:{key}[{idx}] must be H,W,3 RGB, got {frame.shape}")
                images[name] = np.ascontiguousarray(frame.astype(np.uint8, copy=False))
            state = np.asarray(h5[self.qpos_key][idx], dtype=np.float32).reshape(-1)
        if state.size != 14:
            raise ValueError(f"{self.episode_path}:{self.qpos_key}[{idx}] must have 14 values, got {state.size}")
        return {"images": images, "state": np.ascontiguousarray(state), "prompt": self.prompt}

    def _episode_length(self) -> int:
        import h5py  # type: ignore

        with h5py.File(self.episode_path, "r") as h5:
            return min(int(h5[self.action_key].shape[0]), int(h5[self.qpos_key].shape[0]))

    def _resolve_index(self, t: int) -> int:
        if self.length <= 0:
            raise RuntimeError(f"Empty HDF5 episode: {self.episode_path}")
        idx = int(t)
        if self.loop:
            return idx % self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Control step {idx} is outside episode length {self.length}")
        return idx


def resolve_hdf5_episode(data_root: Path) -> Path:
    files = find_hdf5_files(data_root)
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {data_root}")
    return files[0]


def summarize_array(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array, dtype=np.float32)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
    }
    if array.size:
        summary.update(
            {
                "min": float(array.min()),
                "max": float(array.max()),
                "mean": float(array.mean()),
                "std": float(array.std()),
            }
        )
    return summary


def summarize_numbers(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def write_json_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2) + "\n", encoding="utf-8")
