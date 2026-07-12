"""Real Piper14 ObservationSource and ActionSink for Cosmos deployment.

The hardware-facing pieces mirror the FastWAM deployment runtime, but expose
the smaller interfaces used by ``Piper14RTCRuntime``:

* ObservationSource.read_observation(t) -> images + 14-dim state + prompt
* ActionSink.send_action(t, action) -> validate and optionally command Piper
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from piper_cosmos.deployment.cosmos_piper14_remote_client import CosmosPiper14RemotePolicyClient
from piper_cosmos.deployment.piper14_rtc_runtime import Piper14RTCRuntime, Piper14RTCRuntimeConfig


PIPER14_ACTION_DIM = 14
DEFAULT_PROMPT = "Assemble the mouse's battery."


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section `{name}` must be a mapping.")
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def load_mapping_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config without requiring FastWAM imports at runtime."""

    config_path = Path(path).expanduser().resolve()
    try:
        from omegaconf import OmegaConf

        payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    except Exception:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {config_path}")
    return payload


class RealPiper14RosObservationSource:
    """Read synchronized RGB frames and Piper14 qpos for Cosmos policy input."""

    def __init__(self, config: Mapping[str, Any], robot: "Piper14RobotController", prompt: str | None = None):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.rospy = rospy
        self.Image = Image
        self.bridge = CvBridge()
        self.robot = robot
        self.prompt = prompt or str(_section(config, "fastwam").get("prompt", DEFAULT_PROMPT))
        self.cameras = _section(config, "cameras")
        self.topics = _section(config, "ros_topics_dual")
        self.deques = {
            "front": deque(maxlen=2000),
            "left_wrist": deque(maxlen=2000),
            "right_wrist": deque(maxlen=2000),
        }
        self.camera_names = {
            "front": self.cameras.get("front_camera_name", "cam_high"),
            "left_wrist": self.cameras.get("left_wrist_camera_name", "cam_left_wrist"),
            "right_wrist": self.cameras.get("right_wrist_camera_name", "cam_right_wrist"),
        }
        self._validate_camera_config()
        self._init_ros()

    def _validate_camera_config(self) -> None:
        required_flags = ["use_front", "use_wrist", "use_left_wrist", "use_right_wrist"]
        if not all(_bool(self.cameras.get(flag, True)) for flag in required_flags):
            raise ValueError("Cosmos Piper14 real runtime requires front, left wrist, and right wrist RGB cameras.")
        required_names = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
        actual_names = set(str(name) for name in self.camera_names.values())
        if actual_names != required_names:
            raise ValueError(f"Cosmos policy requires image keys {sorted(required_names)}, got {sorted(actual_names)}")

    def _init_ros(self) -> None:
        if not self.rospy.core.is_initialized():
            self.rospy.init_node("cosmos_piper14_real_runtime", anonymous=True)
        self.rospy.Subscriber(
            self.topics.get("img_front_topic", "/camera_h/color/image_raw"),
            self.Image,
            lambda msg: self.deques["front"].append(msg),
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.rospy.Subscriber(
            self.topics.get("img_left_topic", "/camera_l/color/image_raw"),
            self.Image,
            lambda msg: self.deques["left_wrist"].append(msg),
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.rospy.Subscriber(
            self.topics.get("img_right_topic", "/camera_r/color/image_raw"),
            self.Image,
            lambda msg: self.deques["right_wrist"].append(msg),
            queue_size=1000,
            tcp_nodelay=True,
        )

    def read_observation(self, t: int) -> Mapping[str, Any]:
        images = self._read_synchronized_images()
        state = self.robot.get_status_and_state()
        if state.shape != (PIPER14_ACTION_DIM,):
            raise ValueError(f"Expected Piper14 state [{PIPER14_ACTION_DIM}], got {state.shape} at step {t}")
        if not np.isfinite(state).all():
            raise ValueError(f"Non-finite Piper14 state at step {t}")
        return {"images": images, "state": np.ascontiguousarray(state), "prompt": self.prompt}

    def _read_synchronized_images(self) -> dict[str, np.ndarray]:
        if any(len(image_deque) == 0 for image_deque in self.deques.values()):
            raise RuntimeError("Waiting for synchronized ROS RGB frames.")

        frame_time = min(image_deque[-1].header.stamp.to_sec() for image_deque in self.deques.values())
        images: dict[str, np.ndarray] = {}
        for camera_key, image_deque in self.deques.items():
            while image_deque and image_deque[0].header.stamp.to_sec() < frame_time:
                image_deque.popleft()
            if not image_deque:
                raise RuntimeError("Synchronized ROS frame queue was exhausted.")
            image = self.bridge.imgmsg_to_cv2(image_deque.popleft(), "passthrough")
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{camera_key} RGB image must be [H,W,3], got {image.shape}")
            images[str(self.camera_names[camera_key])] = np.ascontiguousarray(image.astype(np.uint8, copy=False))
        return images


class Piper14RobotController:
    """Connect/read/command dual Piper arms using the FastWAM-proven layout."""

    def __init__(self, config: Mapping[str, Any], *, no_robot: bool = False):
        self.robot = _section(config, "robot")
        self.no_robot = bool(no_robot)
        self.left_arm = None
        self.right_arm = None
        self.left_gripper = None
        self.right_gripper = None
        self.left_init_position = np.asarray(
            self.robot.get("left_init_position", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]),
            dtype=np.float32,
        )
        self.right_init_position = np.asarray(
            self.robot.get("right_init_position", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]),
            dtype=np.float32,
        )
        self.initial_position = np.concatenate([self.left_init_position, self.right_init_position]).astype(np.float32)
        self.sim_state = self.initial_position.copy()
        self.action_safety_threshold = float(self.robot.get("action_safety_threshold", 1.5))
        self.state_safety_threshold = float(self.robot.get("state_safety_threshold", 0.3))
        self.max_action_delta = float(self.robot.get("max_action_delta", self.action_safety_threshold))
        self.max_state_tracking_delta = float(self.robot.get("max_state_tracking_delta", self.state_safety_threshold))
        self.action_lower_bounds = self._optional_bounds("action_lower_bounds")
        self.action_upper_bounds = self._optional_bounds("action_upper_bounds")

    def _optional_bounds(self, key: str) -> np.ndarray | None:
        value = self.robot.get(key)
        if value is None:
            return None
        bounds = np.asarray(value, dtype=np.float32).reshape(-1)
        if bounds.size != PIPER14_ACTION_DIM:
            raise ValueError(f"robot.{key} must contain {PIPER14_ACTION_DIM} values, got {bounds.size}")
        return bounds

    def connect(self) -> bool:
        if self.no_robot:
            print("[cosmos-runtime] no_robot=True: skip pyAgxArm/CAN connection.")
            return True
        try:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config

            bitrate = int(self.robot.get("bitrate", 1000000))
            for side, channel_key in (("left", "left_channel"), ("right", "right_channel")):
                channel = str(self.robot.get(channel_key, f"can_{side}"))
                print(f"[cosmos-runtime] connecting {side} arm on {channel} @ {bitrate}")
                cfg = create_agx_arm_config(robot="piper", comm="can", channel=channel, bitrate=bitrate)
                arm = AgxArmFactory.create_arm(cfg)
                gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
                arm.connect()
                if side == "left":
                    self.left_arm = arm
                    self.left_gripper = gripper
                else:
                    self.right_arm = arm
                    self.right_gripper = gripper
            time.sleep(0.5)

            for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
                if arm is None or not arm.is_ok():
                    raise RuntimeError(f"{side} arm connection status check failed.")
                arm.set_flange_vel_acc_limits(
                    max_linear_vel=float(self.robot.get("max_linear_vel", 0.5)),
                    max_angular_vel=float(self.robot.get("max_angular_vel", 0.1)),
                    max_linear_acc=float(self.robot.get("max_linear_acc", 0.1)),
                    max_angular_acc=float(self.robot.get("max_angular_acc", 0.05)),
                    timeout=1.0,
                )
                arm.set_speed_percent(int(self.robot.get("speed_pct", 15)))
                if not self._enable_arm(arm):
                    raise RuntimeError(f"{side} arm enable timeout.")
            return True
        except Exception as exc:
            print(f"[cosmos-runtime] robot connect failed: {exc}")
            return False

    def _enable_arm(self, arm: Any) -> bool:
        for _ in range(5):
            if arm.enable():
                return True
            time.sleep(0.5)
        return False

    def get_status_and_state(self) -> np.ndarray:
        if self.no_robot:
            return self.sim_state.copy()
        state = np.zeros(PIPER14_ACTION_DIM, dtype=np.float32)
        self._read_arm_into(self.left_arm, self.left_gripper, state, 0)
        self._read_arm_into(self.right_arm, self.right_gripper, state, 7)
        return np.ascontiguousarray(state)

    def _read_arm_into(self, arm: Any, gripper: Any, state: np.ndarray, base: int) -> None:
        if arm is None:
            raise RuntimeError("Piper arm is not connected.")
        ja = arm.get_joint_angles()
        if ja is not None:
            state[base : base + 6] = ja.msg
        if gripper is not None:
            gs = gripper.get_gripper_status()
            if gs is not None:
                state[base + 6] = gs.msg.value

    def move_initial(self) -> None:
        if self.no_robot:
            self.sim_state = self.initial_position.copy()
            print("[cosmos-runtime] no_robot=True: skip move_initial.")
            return
        n = 50
        left_traj = np.linspace(self._current_joints(self.left_arm, self.left_init_position[:6]), self.left_init_position[:6], n)
        right_traj = np.linspace(
            self._current_joints(self.right_arm, self.right_init_position[:6]), self.right_init_position[:6], n
        )
        left_gripper = float(np.clip(self.left_init_position[6], 0.0, 0.1))
        right_gripper = float(np.clip(self.right_init_position[6], 0.0, 0.1))
        for i in range(n):
            self.left_arm.move_js(left_traj[i].tolist())
            self.left_gripper.move_gripper_m(value=left_gripper, force=1.0)
            self.right_arm.move_js(right_traj[i].tolist())
            self.right_gripper.move_gripper_m(value=right_gripper, force=1.0)
            time.sleep(0.02)

    def _current_joints(self, arm: Any, fallback: np.ndarray) -> np.ndarray:
        ja = arm.get_joint_angles()
        return np.asarray(ja.msg, dtype=np.float32) if ja is not None else np.asarray(fallback, dtype=np.float32)

    def move(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (PIPER14_ACTION_DIM,):
            raise ValueError(f"Expected Piper14 action [{PIPER14_ACTION_DIM}], got {action.shape}")
        if self.no_robot:
            self.sim_state = action.copy()
            return
        self._move_arm(self.left_arm, self.left_gripper, action[0:6], action[6])
        self._move_arm(self.right_arm, self.right_gripper, action[7:13], action[13])
        time.sleep(0.02)

    def _move_arm(self, arm: Any, gripper: Any, joints: np.ndarray, gripper_cmd: float) -> None:
        arm.move_js(np.asarray(joints, dtype=np.float32).tolist())
        gripper.move_gripper_m(value=float(np.clip(gripper_cmd, 0.0, 0.1)), force=1.0)


class RealPiper14ActionSink:
    """Validate RTC-selected actions and optionally send them to both Pipers."""

    def __init__(self, robot: Piper14RobotController, *, execute_actions: bool = False):
        self.robot = robot
        self.execute_actions = bool(execute_actions)
        self.records: list[tuple[int, np.ndarray]] = []
        self.last_action = robot.get_status_and_state()

    def send_action(self, t: int, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (PIPER14_ACTION_DIM,):
            raise ValueError(f"Expected action [{PIPER14_ACTION_DIM}], got {action.shape} at step {t}")
        if not np.isfinite(action).all():
            raise ValueError(f"Non-finite action at step {t}")

        action_l1 = float(np.mean(np.abs(action - self.last_action)))
        if action_l1 > self.robot.action_safety_threshold:
            raise RuntimeError(f"Safety stop at step {t}: action jump too large {action_l1:.3f}")
        action_linf = float(np.max(np.abs(action - self.last_action)))
        if action_linf > self.robot.max_action_delta:
            raise RuntimeError(f"Safety stop at step {t}: per-joint action jump too large {action_linf:.3f}")
        if self.robot.action_lower_bounds is not None and np.any(action < self.robot.action_lower_bounds):
            raise RuntimeError(f"Safety stop at step {t}: action below configured lower bounds")
        if self.robot.action_upper_bounds is not None and np.any(action > self.robot.action_upper_bounds):
            raise RuntimeError(f"Safety stop at step {t}: action above configured upper bounds")

        current_state = self.robot.get_status_and_state()
        state_tracking_l1 = float(np.mean(np.abs(current_state - self.last_action)))
        if state_tracking_l1 > self.robot.state_safety_threshold:
            raise RuntimeError(f"Safety stop at step {t}: current state too far from last action {state_tracking_l1:.3f}")
        state_tracking_linf = float(np.max(np.abs(current_state - self.last_action)))
        if state_tracking_linf > self.robot.max_state_tracking_delta:
            raise RuntimeError(
                f"Safety stop at step {t}: per-joint state tracking too far {state_tracking_linf:.3f}"
            )

        target_state_l1 = float(np.mean(np.abs(current_state - action)))
        print(
            f"[cosmos-runtime] t={t} action_l1={action_l1:.3f} "
            f"action_linf={action_linf:.3f} state_tracking_l1={state_tracking_l1:.3f} "
            f"state_tracking_linf={state_tracking_linf:.3f} target_state_l1={target_state_l1:.3f} action={action}"
        )
        self.records.append((int(t), action.copy()))
        if self.execute_actions:
            self.robot.move(action)
        elif self.robot.no_robot:
            self.robot.move(action)
        self.last_action = action.copy()

    @property
    def actions(self) -> np.ndarray:
        if not self.records:
            return np.zeros((0, PIPER14_ACTION_DIM), dtype=np.float32)
        return np.stack([action for _, action in self.records], axis=0).astype(np.float32, copy=False)


def validate_cosmos_metadata(metadata: Mapping[str, Any]) -> None:
    raw_action_dim = int(metadata.get("raw_action_dim", PIPER14_ACTION_DIM))
    action_horizon = int(metadata.get("action_horizon", 0))
    image_keys = set(metadata.get("image_keys", []))
    required_image_keys = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    if raw_action_dim != PIPER14_ACTION_DIM:
        raise RuntimeError(f"Cosmos server raw_action_dim must be 14 for Piper14, got {raw_action_dim}")
    if action_horizon <= 0:
        raise RuntimeError(f"Cosmos server returned invalid action_horizon={action_horizon}")
    if image_keys != required_image_keys:
        raise RuntimeError(f"Cosmos server image_keys must be {sorted(required_image_keys)}, got {sorted(image_keys)}")


def run_real_cosmos_piper14_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run real-robot Cosmos Piper14 RTC against an already-started policy server."""

    runtime_cfg = _section(config, "runtime")
    server_cfg = _section(config, "policy_server")
    fastwam_cfg = _section(config, "fastwam")
    no_robot = _bool(runtime_cfg.get("no_robot", False))
    execute_actions = _bool(runtime_cfg.get("execute_actions", False))

    robot = Piper14RobotController(config, no_robot=no_robot)
    if not robot.connect():
        raise RuntimeError("Failed to connect Piper14 robot controller.")
    if _bool(runtime_cfg.get("move_to_initial", False)):
        robot.move_initial()

    obs_source = RealPiper14RosObservationSource(
        config,
        robot,
        prompt=str(fastwam_cfg.get("prompt", DEFAULT_PROMPT)),
    )
    sink = RealPiper14ActionSink(robot, execute_actions=execute_actions)
    rtc_config = Piper14RTCRuntimeConfig(
        action_dim=PIPER14_ACTION_DIM,
        chunk_size=int(runtime_cfg.get("action_chunk_size", fastwam_cfg.get("action_horizon", 32))),
        control_hz=float(runtime_cfg.get("rospy_rate", 30)),
        max_steps=int(runtime_cfg["max_steps"]) if runtime_cfg.get("max_steps") is not None else 2**31 - 1,
        replan_interval=int(runtime_cfg.get("replan_interval", 8)),
        exp_weight_factor=float(runtime_cfg.get("exp_weight_factor", 0.5)),
        sleep=True,
        prompt=str(fastwam_cfg.get("prompt", DEFAULT_PROMPT)),
        debug=_bool(runtime_cfg.get("rtc_debug", False)),
    )

    with CosmosPiper14RemotePolicyClient(
        host=str(server_cfg.get("host", "127.0.0.1")),
        port=int(server_cfg.get("port", 8766)),
        authkey=server_cfg.get("authkey", "cosmos-piper14"),
    ) as policy:
        metadata = policy.metadata()
        validate_cosmos_metadata(metadata)
        print(f"[cosmos-runtime] policy metadata ok: {metadata}")
        input("Press Enter to start Cosmos Piper14 real runtime...")
        runtime = Piper14RTCRuntime(policy=policy, observation_source=obs_source, action_sink=sink, config=rtc_config)
        report = runtime.run()
        policy.reset()

    output_dir = Path(str(runtime_cfg.get("output_dir", "./output_actions"))).expanduser()
    if sink.records:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "cosmos_rtc_selected_actions.npy"
        np.save(output_path, sink.actions)
        print(f"[cosmos-runtime] saved selected actions to {output_path}")
    return {"metadata": metadata, **report}
