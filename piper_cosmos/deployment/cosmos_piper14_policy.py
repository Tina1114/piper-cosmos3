"""RTC-facing Piper14 wrapper for Cosmos3 action policy inference."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image

from piper_cosmos.cosmos3.domain import PIPER14_DOMAIN_NAME, PIPER14_RAW_ACTION_DIM


IMAGE_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_PROMPT = "Assemble the mouse's battery."


class CosmosActionBackend(Protocol):
    def predict_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run one Cosmos policy prediction."""


@dataclass(frozen=True)
class CosmosPiper14PolicyConfig:
    """Configuration for the Piper14 deployment policy wrapper."""

    checkpoint: str = "/project/peilab/wam/cosmos3_cy/cosmos_battery/20k"
    config_file: str | None = None
    prompt: str = DEFAULT_PROMPT
    action_horizon: int = 32
    raw_action_dim: int = PIPER14_RAW_ACTION_DIM
    max_action_dim: int = 64
    camera_height: int = 480
    camera_width: int = 640
    resolution: str = "480"
    num_steps: int = 4
    guidance: float = 3.0
    shift: float = 5.0
    fps: int = 30
    seed: int = 0
    host: str = "127.0.0.1"
    port: int = 8766
    mock_backend: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CosmosPiper14PolicyConfig":
        raw = payload.get("cosmos_piper14", payload)
        if not isinstance(raw, Mapping):
            raise TypeError("Expected a mapping or top-level `cosmos_piper14` mapping.")
        values = {field: raw[field] for field in cls.__dataclass_fields__ if field in raw}
        return cls(**values)


@dataclass
class Piper14Observation:
    cam_high: np.ndarray
    cam_left_wrist: np.ndarray
    cam_right_wrist: np.ndarray
    state: np.ndarray
    prompt: str


class MockCosmosActionBackend:
    """Small deterministic backend for no-robot/no-GPU tests."""

    def __init__(self, action_horizon: int, action_dim: int) -> None:
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)

    def predict_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        state = np.asarray(request.get("state", np.zeros(self.action_dim)), dtype=np.float32).reshape(-1)
        if state.size != self.action_dim:
            state = np.zeros(self.action_dim, dtype=np.float32)
        increments = np.arange(self.action_horizon, dtype=np.float32)[:, None] * 0.001
        return {"action": (state[None, :] + increments).astype(np.float32).tolist()}


class LiberoActionServiceBackend:
    """Adapter around Cosmos Framework's official HTTP action service core.

    Piper14 training stores current qpos as the first action row, followed by
    future actions.  This backend mirrors the official LIBERO server batch
    construction but fills row 0 with the incoming Piper14 state and returns only
    the predicted future rows.
    """

    def __init__(self, config: CosmosPiper14PolicyConfig) -> None:
        from cosmos_framework.inference.common.args import CheckpointOverrides
        from cosmos_framework.scripts.action_policy_server_libero import ActionModelService, ActionServerArgs
        from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

        from piper_cosmos.cosmos3.domain import register_piper14_domain

        self.config = config
        register_piper14_domain()
        checkpoint_kwargs = {"checkpoint_path": str(config.checkpoint)}
        if config.config_file:
            checkpoint_kwargs["config_file"] = str(config.config_file)
        checkpoint_overrides = CheckpointOverrides(**checkpoint_kwargs)

        class Piper14ActionServerArgs(ActionServerArgs):
            def build_setup_overrides(self):
                setup_overrides = super().build_setup_overrides()
                setup_overrides.guardrails = False
                return setup_overrides

        args = Piper14ActionServerArgs(
            checkpoint=checkpoint_overrides,
            seed=int(config.seed),
            guidance=float(config.guidance),
            num_steps=int(config.num_steps),
            fps=int(config.fps),
            action_chunk_size=int(config.action_horizon),
            raw_action_dim=int(config.raw_action_dim),
            max_action_dim=int(config.max_action_dim),
        )
        self.service = ActionModelService(args)
        self.transform = ActionTransformPipeline(
            tokenizer_config=None,
            cfg_dropout_rate=0.0,
            max_action_dim=int(config.max_action_dim),
            append_viewpoint_info=True,
            append_duration_fps_timestamps=True,
            append_resolution_info=True,
            append_idle_frames=False,
        )

    def predict_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        import torch

        from cosmos_framework.data.vfm.action.domain_utils import get_domain_id

        concat_view = request.get("concat_view")
        if concat_view is None:
            return self.service.predict_policy(dict(request))

        image = ensure_rgb_uint8(concat_view, "concat_view")
        state = np.asarray(request.get("state"), dtype=np.float32).reshape(-1)
        if state.size != PIPER14_RAW_ACTION_DIM:
            raise ValueError(f"Expected Piper14 state dim 14, got {state.size}")

        prompt = request.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")
        domain_name = request.get("domain_name")
        if not isinstance(domain_name, str):
            raise ValueError("'domain_name' must be a string")

        future_horizon = int(self.service.cfg.action_chunk_size)
        action_rows = future_horizon + 1
        video = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous().unsqueeze(1).repeat(1, action_rows, 1, 1)
        action = torch.zeros((action_rows, PIPER14_RAW_ACTION_DIM), dtype=torch.float32)
        action[0, :PIPER14_RAW_ACTION_DIM] = torch.from_numpy(state)
        sample = {
            "ai_caption": prompt,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.service.cfg.fps, dtype=torch.long),
            "mode": "policy",
            "domain_id": torch.tensor(get_domain_id(domain_name), dtype=torch.long),
            "viewpoint": "concat_view",
        }
        batch = build_data_batch_from_sample(self.transform(sample, resolution=self.config.resolution))

        with self.service._lock:
            with torch.inference_mode():
                samples = self.service.model.generate_samples_from_batch(
                    batch,
                    guidance=float(self.config.guidance),
                    seed=[int(self.config.seed)],
                    num_steps=int(self.config.num_steps),
                    shift=float(self.config.shift),
                    has_negative_prompt=False,
                )
        pred_action = samples["action"][0].float().squeeze(0)
        future = pred_action[1 : future_horizon + 1, :PIPER14_RAW_ACTION_DIM].detach().cpu().numpy()
        return {"action": future.astype(np.float32, copy=False).tolist()}


class CosmosPiper14PolicyClient:
    """Local policy object with an RTC-compatible interface."""

    def __init__(
        self,
        config: CosmosPiper14PolicyConfig | Mapping[str, Any],
        backend: CosmosActionBackend | None = None,
    ) -> None:
        self.config = (
            config if isinstance(config, CosmosPiper14PolicyConfig) else CosmosPiper14PolicyConfig.from_mapping(config)
        )
        if self.config.raw_action_dim != PIPER14_RAW_ACTION_DIM:
            raise ValueError(f"Piper14 raw_action_dim must be 14, got {self.config.raw_action_dim}")
        if self.config.camera_height <= 0 or self.config.camera_width <= 0:
            raise ValueError("camera_height and camera_width must be positive")
        if self.config.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

        self.backend = backend or self._build_backend()
        self.observation: Piper14Observation | None = None

    def _build_backend(self) -> CosmosActionBackend:
        if self.config.mock_backend:
            return MockCosmosActionBackend(self.config.action_horizon, self.config.raw_action_dim)
        return LiberoActionServiceBackend(self.config)

    def update_observation(self, obs: Mapping[str, Any]) -> None:
        self.observation = self._coerce_observation(obs)

    def get_action(self) -> np.ndarray:
        if self.observation is None:
            raise RuntimeError("Policy observation is empty. Call update_observation(obs) before get_action().")
        return self._infer_observation(self.observation)

    def infer(self, obs: Mapping[str, Any]) -> np.ndarray:
        self.update_observation(obs)
        return self.get_action()

    def reset(self) -> None:
        self.observation = None

    def metadata(self) -> dict[str, Any]:
        return {
            "domain_name": PIPER14_DOMAIN_NAME,
            "raw_action_dim": int(self.config.raw_action_dim),
            "action_horizon": int(self.config.action_horizon),
            "image_keys": list(IMAGE_KEYS),
            "action_type": "absolute_joint_position_command",
            "checkpoint": str(self.config.checkpoint),
            "prompt": str(self.config.prompt),
            "mock_backend": bool(self.config.mock_backend),
            "camera_height": int(self.config.camera_height),
            "camera_width": int(self.config.camera_width),
            "resolution": str(self.config.resolution),
            "num_steps": int(self.config.num_steps),
            "guidance": float(self.config.guidance),
            "shift": float(self.config.shift),
        }

    def build_policy_request(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        observation = self._coerce_observation(obs)
        concat_view = compose_concat_view(
            observation.cam_high,
            observation.cam_left_wrist,
            observation.cam_right_wrist,
            camera_height=self.config.camera_height,
            camera_width=self.config.camera_width,
        )
        return {
            "image": encode_rgb_png_base64(concat_view),
            "concat_view": concat_view,
            "prompt": observation.prompt,
            "domain_name": PIPER14_DOMAIN_NAME,
            "image_size": int(concat_view.shape[0]),
            "state": observation.state.astype(np.float32, copy=True),
        }

    def _infer_observation(self, observation: Piper14Observation) -> np.ndarray:
        request = self.build_policy_request(
            {
                "images": {
                    "cam_high": observation.cam_high,
                    "cam_left_wrist": observation.cam_left_wrist,
                    "cam_right_wrist": observation.cam_right_wrist,
                },
                "state": observation.state,
                "prompt": observation.prompt,
            }
        )
        response = self.backend.predict_policy(request)
        action = np.asarray(response.get("action", []), dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.ndim != 2:
            raise ValueError(f"Backend returned action with shape {action.shape}; expected [T,D].")
        if action.shape[1] < self.config.raw_action_dim:
            raise ValueError(
                f"Backend returned action dim {action.shape[1]}; expected at least {self.config.raw_action_dim}."
            )
        action = action[:, : self.config.raw_action_dim]
        if action.shape[0] == 0:
            raise ValueError("Backend returned an empty action chunk.")
        if action.shape[0] < self.config.action_horizon:
            tail = np.repeat(action[-1:, :], self.config.action_horizon - action.shape[0], axis=0)
            action = np.concatenate([action, tail], axis=0)
        elif action.shape[0] > self.config.action_horizon:
            action = action[: self.config.action_horizon]
        if not np.isfinite(action).all():
            raise ValueError("Backend returned non-finite action values.")
        return np.ascontiguousarray(action.astype(np.float32, copy=False))

    def _coerce_observation(self, obs: Mapping[str, Any]) -> Piper14Observation:
        images = obs.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("Observation must contain an `images` mapping.")
        missing = [key for key in IMAGE_KEYS if key not in images or images[key] is None]
        if missing:
            raise ValueError(f"Observation is missing required image keys: {missing}")

        if "state" in obs:
            state_value = obs["state"]
        elif "qpos" in obs:
            state_value = obs["qpos"]
        else:
            raise ValueError("Observation must contain `state` or `qpos`.")
        state = np.asarray(state_value, dtype=np.float32).reshape(-1)
        if state.size != PIPER14_RAW_ACTION_DIM:
            raise ValueError(f"Expected state/qpos dim 14, got {state.size}")

        return Piper14Observation(
            cam_high=ensure_rgb_uint8(images["cam_high"], "images.cam_high"),
            cam_left_wrist=ensure_rgb_uint8(images["cam_left_wrist"], "images.cam_left_wrist"),
            cam_right_wrist=ensure_rgb_uint8(images["cam_right_wrist"], "images.cam_right_wrist"),
            state=np.ascontiguousarray(state),
            prompt=str(obs.get("prompt", self.config.prompt)),
        )


def ensure_rgb_uint8(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key} must have shape [H,W,3], got {image.shape}")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    image_f = image.astype(np.float32)
    if image_f.size and float(np.nanmax(image_f)) <= 1.0:
        image_f = image_f * 255.0
    return np.ascontiguousarray(np.clip(image_f, 0, 255).astype(np.uint8))


def resize_rgb_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)
    pil = Image.fromarray(image, mode="RGB")
    resized = pil.resize((int(width), int(height)), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8).copy()


def compose_concat_view(
    cam_high: Any,
    cam_left_wrist: Any,
    cam_right_wrist: Any,
    *,
    camera_height: int = 480,
    camera_width: int = 640,
) -> np.ndarray:
    high = resize_rgb_uint8(ensure_rgb_uint8(cam_high, "cam_high"), camera_height, camera_width)
    left = resize_rgb_uint8(ensure_rgb_uint8(cam_left_wrist, "cam_left_wrist"), camera_height, camera_width)
    right = resize_rgb_uint8(ensure_rgb_uint8(cam_right_wrist, "cam_right_wrist"), camera_height, camera_width)

    half_h = max(1, camera_height // 2)
    half_w = max(1, camera_width // 2)
    left = resize_rgb_uint8(left, half_h, half_w)
    right = resize_rgb_uint8(right, half_h, half_w)
    bottom = np.concatenate([left, right], axis=1)
    return np.ascontiguousarray(np.concatenate([high, bottom], axis=0))


def encode_rgb_png_base64(image: np.ndarray) -> str:
    image = ensure_rgb_uint8(image, "image")
    buf = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_data_batch_from_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    from cosmos_framework.data.vfm.joint_dataloader import IterativeJointDataLoader

    data_batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            data_batch[key] = [[value]]
        else:
            data_batch[key] = [value]
    return data_batch


def resolve_checkpoint(path: str | Path) -> str:
    return str(Path(path).expanduser())
