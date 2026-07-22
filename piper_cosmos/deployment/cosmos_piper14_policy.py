"""RTC-facing Piper14 wrapper for Cosmos3 action policy inference."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol

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
    condition_only_vae: bool = True
    instruction_cache: bool = True
    instruction_cache_dir: str | None = None
    instruction_cache_max_entries: int = 4
    host: str = "127.0.0.1"
    port: int = 8766
    mock_backend: bool = False
    timing: bool = False
    cuda_memory: bool = False
    cuda_memory_history: str | None = None
    cuda_memory_history_max_entries: int = 200_000

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


class _SegmentedTimer:
    """Collect synchronized wall-clock timings for one policy request."""

    def __init__(self, enabled: bool, cuda_memory: bool = False) -> None:
        self.enabled = bool(enabled)
        self.cuda_memory = bool(cuda_memory)
        self.started = time.perf_counter()
        self._durations_ms: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._cuda_probe: Callable[[], dict[str, int]] | None = None
        self._cuda_baseline: dict[str, int] | None = None
        self._cuda_stages: dict[str, dict[str, int]] = {}

    def attach_cuda(self, torch: Any) -> None:
        if not self.enabled or not self.cuda_memory or self._cuda_probe is not None:
            return
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return
        cuda.synchronize()
        cuda.reset_peak_memory_stats()

        def probe() -> dict[str, int]:
            free_bytes, total_bytes = cuda.mem_get_info()
            return {
                "allocated": int(cuda.memory_allocated()),
                "reserved": int(cuda.memory_reserved()),
                "peak_allocated": int(cuda.max_memory_allocated()),
                "driver_used": int(total_bytes - free_bytes),
                "total": int(total_bytes),
            }

        self._cuda_probe = probe
        self._cuda_baseline = probe()

    @contextmanager
    def measure(self, name: str, synchronize: Callable[[], None] | None = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if synchronize is not None:
            synchronize()
        memory_before = self._cuda_probe() if self._cuda_probe is not None else None
        started = time.perf_counter()
        try:
            yield
        finally:
            if synchronize is not None:
                synchronize()
            self.add(name, (time.perf_counter() - started) * 1000.0)
            if self._cuda_probe is not None and memory_before is not None:
                self._record_cuda_memory(name, memory_before, self._cuda_probe())

    def add(self, name: str, duration_ms: float) -> None:
        if not self.enabled:
            return
        self._durations_ms[name] = self._durations_ms.get(name, 0.0) + float(duration_ms)
        self._counts[name] = self._counts.get(name, 0) + 1

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        if not self.enabled:
            return {}
        result = {
            name: {"ms": round(duration_ms, 3), "count": self._counts[name]}
            for name, duration_ms in self._durations_ms.items()
        }
        result["policy.total"] = {"ms": round((time.perf_counter() - self.started) * 1000.0, 3), "count": 1}
        return result

    def cuda_snapshot(self) -> dict[str, Any]:
        if self._cuda_probe is None or self._cuda_baseline is None:
            return {}
        return {
            "baseline": dict(self._cuda_baseline),
            "final": self._cuda_probe(),
            "stages": {name: dict(values) for name, values in self._cuda_stages.items()},
        }

    def _record_cuda_memory(self, name: str, before: Mapping[str, int], after: Mapping[str, int]) -> None:
        baseline = self._cuda_baseline
        if baseline is None:
            return
        values = self._cuda_stages.setdefault(
            name,
            {
                "end_allocated": 0,
                "end_reserved": 0,
                "end_driver_used": 0,
                "stage_delta": 0,
                "delta_from_baseline": 0,
                "request_peak": 0,
                "new_peak": 0,
                "total": int(after["total"]),
            },
        )
        values["end_allocated"] = max(values["end_allocated"], int(after["allocated"]))
        values["end_reserved"] = max(values["end_reserved"], int(after["reserved"]))
        values["end_driver_used"] = max(values["end_driver_used"], int(after["driver_used"]))
        values["stage_delta"] = max(values["stage_delta"], int(after["allocated"] - before["allocated"]))
        values["delta_from_baseline"] = max(
            values["delta_from_baseline"], int(after["allocated"] - baseline["allocated"])
        )
        values["request_peak"] = max(values["request_peak"], int(after["peak_allocated"]))
        values["new_peak"] += max(0, int(after["peak_allocated"] - before["peak_allocated"]))

    def log(self, request_id: int) -> dict[str, dict[str, float | int]]:
        result = self.snapshot()
        if result:
            fields = []
            for name, value in result.items():
                count = int(value["count"])
                count_suffix = f"[{count}]" if count > 1 else ""
                fields.append(f"{name}{count_suffix}={float(value['ms']):.3f}ms")
            print(f"[cosmos-piper14-timing] request={request_id} " + " ".join(fields), flush=True)
        memory = self.cuda_snapshot()
        if memory:
            baseline = memory["baseline"]
            final = memory["final"]
            total = int(final["total"])
            mib = 1024.0 * 1024.0
            print(
                f"[cosmos-piper14-cuda-memory] request={request_id} "
                f"baseline_allocated={baseline['allocated'] / mib:.1f}MiB "
                f"baseline_reserved={baseline['reserved'] / mib:.1f}MiB "
                f"baseline_driver_used={baseline['driver_used'] / mib:.1f}MiB "
                f"request_peak={final['peak_allocated'] / mib:.1f}MiB "
                f"final_allocated={final['allocated'] / mib:.1f}MiB "
                f"final_reserved={final['reserved'] / mib:.1f}MiB "
                f"total={total / mib:.1f}MiB "
                f"peak_pct={100.0 * final['peak_allocated'] / total:.2f}%",
                flush=True,
            )
            for name, values in memory["stages"].items():
                print(
                    f"[cosmos-piper14-cuda-memory-stage] request={request_id} stage={name} "
                    f"end_allocated={values['end_allocated'] / mib:.1f}MiB "
                    f"delta_from_baseline={values['delta_from_baseline'] / mib:+.1f}MiB "
                    f"stage_delta={values['stage_delta'] / mib:+.1f}MiB "
                    f"new_peak={values['new_peak'] / mib:.1f}MiB "
                    f"request_peak={values['request_peak'] / mib:.1f}MiB "
                    f"peak_pct={100.0 * values['request_peak'] / values['total']:.2f}%",
                    flush=True,
                )
        return result


@contextmanager
def _profile_model_methods(
    model: Any,
    timer: _SegmentedTimer,
    synchronize_cuda: Callable[[], None] | None,
) -> Iterator[None]:
    """Temporarily time Cosmos model stages called inside generate_samples_from_batch."""

    if not timer.enabled:
        yield
        return

    originals: list[tuple[str, Any, bool]] = []

    def install(method_name: str, timing_name: str | Callable[[Mapping[str, Any]], str]) -> None:
        original = getattr(model, method_name, None)
        if original is None or not callable(original):
            return
        had_instance_attribute = method_name in vars(model)

        def timed_method(*args, **kwargs):
            name = timing_name(kwargs) if callable(timing_name) else timing_name
            with timer.measure(name, synchronize_cuda):
                return original(*args, **kwargs)

        try:
            setattr(model, method_name, timed_method)
        except (AttributeError, TypeError):
            return
        originals.append((method_name, original, had_instance_attribute))

    install("get_data_and_condition", "model.prepare.vision_action")
    install("_get_inference_text_tokens", "model.prepare.text")
    install("_pack_input_sequence", "model.prepare.pack")
    install("_prepare_inference_data", "model.prepare.total")
    install("_prepare_reasoner_prefill", "model.reasoner.restore")
    install("build_inference_memory_state", "model.reasoner.memory")
    install(
        "_get_velocity",
        lambda kwargs: "model.reasoner.prefill" if kwargs.get("und_only", False) else "model.denoise.velocity",
    )
    install("_maybe_finalize_reasoner_offload", "model.reasoner.offload")
    try:
        yield
    finally:
        for method_name, original, had_instance_attribute in reversed(originals):
            if had_instance_attribute:
                setattr(model, method_name, original)
            else:
                delattr(model, method_name)


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
        if config.instruction_cache:
            # Attention dispatch must be installed while the Cosmos model is
            # constructed, before the first policy request arrives.
            os.environ["COSMOS3_REASONER_KVCACHE"] = "1"

        from cosmos_framework.inference.common.args import CheckpointOverrides
        from cosmos_framework.scripts.action_policy_server_libero import ActionModelService, ActionServerArgs
        from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

        from piper_cosmos.cosmos3.domain import register_piper14_domain

        self.config = config
        self._instruction_cache_namespace = build_instruction_cache_namespace(config)
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
        self._cuda_memory_history_dumped = False
        if config.cuda_memory_history:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("--cuda-memory-history requires an available CUDA device")
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="all",
                max_entries=int(config.cuda_memory_history_max_entries),
                clear_history=True,
                global_record_annotations=True,
            )

    def predict_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        import torch

        from cosmos_framework.data.generator.action.domain_utils import get_domain_id

        timer_value = request.get("_timing")
        timer = (
            timer_value
            if isinstance(timer_value, _SegmentedTimer)
            else _SegmentedTimer(self.config.timing or self.config.cuda_memory, self.config.cuda_memory)
        )
        owns_timer = timer is not timer_value
        cuda = getattr(torch, "cuda", None)
        synchronize_cuda = cuda.synchronize if cuda is not None and cuda.is_available() else None
        timer.attach_cuda(torch)

        concat_view = request.get("concat_view")
        if concat_view is None:
            official_request = {key: value for key, value in request.items() if key != "_timing"}
            with timer.measure("backend.official_service", synchronize_cuda):
                response = self.service.predict_policy(official_request)
            if owns_timer:
                timer.log(request_id=0)
            return response

        with timer.measure("backend.validate"):
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
        with timer.measure("backend.video_repeat"):
            video = (
                torch.from_numpy(image.copy())
                .permute(2, 0, 1)
                .contiguous()
                .unsqueeze(1)
                .repeat(1, action_rows, 1, 1)
            )
        with timer.measure("backend.sample_build"):
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
        with timer.measure("backend.transform", synchronize_cuda):
            transformed = self.transform(sample, resolution=self.config.resolution)
        with timer.measure("backend.batch_build"):
            batch = build_data_batch_from_sample(transformed)
            batch["inference_condition_only_vae"] = bool(self.config.condition_only_vae)
            batch["inference_instruction_cache"] = bool(self.config.instruction_cache)
            batch["inference_instruction_cache_namespace"] = self._instruction_cache_namespace
            batch["inference_instruction_cache_max_entries"] = int(self.config.instruction_cache_max_entries)
            if self.config.instruction_cache_dir:
                batch["inference_instruction_cache_dir"] = str(
                    Path(self.config.instruction_cache_dir).expanduser().resolve()
                )

        lock_started = time.perf_counter()
        with self.service._lock:
            timer.add("model.lock_wait", (time.perf_counter() - lock_started) * 1000.0)
            with torch.inference_mode():
                with _profile_model_methods(self.service.model, timer, synchronize_cuda):
                    with timer.measure("model.generate.total", synchronize_cuda):
                        samples = self.service.model.generate_samples_from_batch(
                            batch,
                            guidance=float(self.config.guidance),
                            seed=[int(self.config.seed)],
                            num_steps=int(self.config.num_steps),
                            shift=float(self.config.shift),
                            has_negative_prompt=False,
                        )
        with timer.measure("backend.action_output", synchronize_cuda):
            pred_action = samples["action"][0].float().squeeze(0)
            future = pred_action[1 : future_horizon + 1, :PIPER14_RAW_ACTION_DIM].detach().cpu().numpy()
            response = {"action": future.astype(np.float32, copy=False).tolist()}
        self._dump_cuda_memory_history(torch)
        if owns_timer:
            timer.log(request_id=0)
        return response

    def _dump_cuda_memory_history(self, torch: Any) -> None:
        if not self.config.cuda_memory_history or self._cuda_memory_history_dumped:
            return
        output_path = Path(self.config.cuda_memory_history).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(str(output_path))
        torch.cuda.memory._record_memory_history(enabled=None)
        self._cuda_memory_history_dumped = True
        print(f"[cosmos-piper14-cuda-memory] snapshot={output_path}", flush=True)


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
        self.last_timing: dict[str, dict[str, float | int]] = {}
        self._timing_request_id = 0

    def _build_backend(self) -> CosmosActionBackend:
        if self.config.mock_backend:
            return MockCosmosActionBackend(self.config.action_horizon, self.config.raw_action_dim)
        return LiberoActionServiceBackend(self.config)

    def update_observation(self, obs: Mapping[str, Any]) -> None:
        self.observation = self._coerce_observation(obs)

    def get_action(self) -> np.ndarray:
        if self.observation is None:
            raise RuntimeError("Policy observation is empty. Call update_observation(obs) before get_action().")
        timer = _SegmentedTimer(self.config.timing or self.config.cuda_memory, self.config.cuda_memory)
        return self._run_timed_inference(self.observation, timer)

    def infer(self, obs: Mapping[str, Any]) -> np.ndarray:
        timer = _SegmentedTimer(self.config.timing or self.config.cuda_memory, self.config.cuda_memory)
        with timer.measure("observation.coerce"):
            self.update_observation(obs)
        assert self.observation is not None
        return self._run_timed_inference(self.observation, timer)

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
            "timing": bool(self.config.timing),
            "cuda_memory": bool(self.config.cuda_memory),
            "cuda_memory_history": self.config.cuda_memory_history,
            "camera_height": int(self.config.camera_height),
            "camera_width": int(self.config.camera_width),
            "resolution": str(self.config.resolution),
            "num_steps": int(self.config.num_steps),
            "guidance": float(self.config.guidance),
            "shift": float(self.config.shift),
            "condition_only_vae": bool(self.config.condition_only_vae),
            "instruction_cache": bool(self.config.instruction_cache),
            "instruction_cache_dir": self.config.instruction_cache_dir,
        }

    def build_policy_request(
        self,
        obs: Mapping[str, Any],
        *,
        _timing: _SegmentedTimer | None = None,
    ) -> dict[str, Any]:
        timer = _timing or _SegmentedTimer(False)
        with timer.measure("request.coerce"):
            observation = self._coerce_observation(obs)
        with timer.measure("request.concat_view"):
            concat_view = compose_concat_view(
                observation.cam_high,
                observation.cam_left_wrist,
                observation.cam_right_wrist,
                camera_height=self.config.camera_height,
                camera_width=self.config.camera_width,
            )
        with timer.measure("request.png_base64"):
            encoded_image = encode_rgb_png_base64(concat_view)
        return {
            "image": encoded_image,
            "concat_view": concat_view,
            "prompt": observation.prompt,
            "domain_name": PIPER14_DOMAIN_NAME,
            "image_size": int(concat_view.shape[0]),
            "state": observation.state.astype(np.float32, copy=True),
        }

    def _run_timed_inference(self, observation: Piper14Observation, timer: _SegmentedTimer) -> np.ndarray:
        self._timing_request_id += 1
        request_id = self._timing_request_id
        try:
            return self._infer_observation(observation, timer)
        finally:
            self.last_timing = timer.log(request_id)

    def _infer_observation(self, observation: Piper14Observation, timer: _SegmentedTimer) -> np.ndarray:
        request = self.build_policy_request(
            {
                "images": {
                    "cam_high": observation.cam_high,
                    "cam_left_wrist": observation.cam_left_wrist,
                    "cam_right_wrist": observation.cam_right_wrist,
                },
                "state": observation.state,
                "prompt": observation.prompt,
            },
            _timing=timer,
        )
        request["_timing"] = timer
        with timer.measure("backend.total"):
            response = self.backend.predict_policy(request)
        with timer.measure("policy.action_validate"):
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
    from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader

    data_batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            data_batch[key] = [[value]]
        else:
            data_batch[key] = [value]
    return data_batch


def build_instruction_cache_namespace(config: CosmosPiper14PolicyConfig) -> str:
    """Fingerprint model/layout inputs that must match before K/V reuse."""

    def path_identity(value: str | None) -> dict[str, Any] | None:
        if value is None:
            return None
        path = Path(value).expanduser().resolve()
        identity: dict[str, Any] = {"path": str(path)}
        try:
            stat = path.stat()
        except OSError:
            identity["missing"] = True
        else:
            identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "is_dir": path.is_dir()})
        return identity

    payload = {
        "schema_version": 1,
        "checkpoint": path_identity(config.checkpoint),
        "config_file": path_identity(config.config_file),
        "action_horizon": int(config.action_horizon),
        "raw_action_dim": int(config.raw_action_dim),
        "max_action_dim": int(config.max_action_dim),
        "resolution": str(config.resolution),
        "fps": int(config.fps),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_checkpoint(path: str | Path) -> str:
    return str(Path(path).expanduser())
