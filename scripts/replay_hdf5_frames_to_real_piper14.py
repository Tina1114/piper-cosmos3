#!/usr/bin/env python3
"""Infer on HDF5 frames 0,32,64,...,last and optionally execute on Piper14."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piper_cosmos.data.hdf5_reader import (  # noqa: E402
    DEFAULT_ACTION_KEY,
    HDF5EpisodeReader,
    hdf5_key,
    load_config,
)
from piper_cosmos.deployment.cosmos_piper14_remote_client import (  # noqa: E402
    CosmosPiper14RemotePolicyClient,
)
from piper_cosmos.deployment.piper14_rtc_runtime import HDF5ObservationSource  # noqa: E402
from piper_cosmos.deployment.real_robot_runtime import (  # noqa: E402
    Piper14RobotController,
    RealPiper14ActionSink,
    load_mapping_config,
    validate_cosmos_metadata,
)


DEFAULT_CONFIG = ROOT / "configs" / "real_deploy_cosmos_battery_motion.yaml"
DEFAULT_DATA_CONFIG = ROOT / "configs" / "dataset_configs" / "battery_assemble_hdf5.yaml"
DEFAULT_OUTPUT = ROOT / "output_actions" / "hdf5_replay_predictions.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--authkey", default="cosmos-piper14")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--execute-actions", action="store_true")
    parser.add_argument(
        "--move-to-dataset-start",
        action="store_true",
        help="Command the recorded qpos[0] before executing predicted actions.",
    )
    parser.add_argument(
        "--reset-to-dataset-qpos-each-anchor",
        action="store_true",
        help="Before every sampled frame, safely interpolate the real robot to that frame's recorded qpos.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the interactive real-motion confirmation.")
    return parser.parse_args()


def frame_anchors(length: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError("Episode must contain at least one frame.")
    if stride <= 0:
        raise ValueError("--stride must be positive.")
    anchors = list(range(0, length, stride))
    if anchors[-1] != length - 1:
        anchors.append(length - 1)
    return anchors


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return dict(value) if isinstance(value, dict) else {}


def _read_ground_truth(reader: HDF5EpisodeReader, frame: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 14), dtype=np.float32)
    return np.asarray(reader.read_action_chunk(frame, count), dtype=np.float32)[:, :14]


def _confirm_real_motion(args: argparse.Namespace, anchors: list[int]) -> None:
    if not args.execute_actions or args.yes:
        return
    print("\nWARNING: this will command both physical Piper arms.")
    print(f"Episode: {args.episode}")
    print(f"Anchors: {anchors}")
    print("Dataset images/qpos drive inference; they are not live robot observations.")
    answer = input("Type EXECUTE_HDF5_REPLAY to continue: ").strip()
    if answer != "EXECUTE_HDF5_REPLAY":
        raise SystemExit("Real-motion confirmation rejected.")


def _move_to_dataset_qpos(
    robot: Piper14RobotController,
    sink: RealPiper14ActionSink,
    target: np.ndarray,
    *,
    control_hz: float,
    frame: int,
    max_delta_per_step: float = 0.02,
) -> None:
    """Move to a recorded qpos in bounded, safety-checked steps."""
    target = np.asarray(target, dtype=np.float32).reshape(14)
    start = robot.get_status_and_state().astype(np.float32)
    max_delta = float(np.max(np.abs(target - start)))
    steps = max(1, int(np.ceil(max_delta / max(float(max_delta_per_step), 1e-6))))
    period = 1.0 / max(float(control_hz), 1e-6)
    print(
        f"[hdf5-replay] interpolating to dataset qpos[{frame}]: max_delta={max_delta:.4f} "
        f"steps={steps} max_delta_per_step={max_delta_per_step:.4f}"
    )
    for index in range(1, steps + 1):
        step_started = time.perf_counter()
        alpha = float(index) / float(steps)
        action = start + alpha * (target - start)
        sink.send_action(-steps + index - 1, action)
        elapsed = time.perf_counter() - step_started
        time.sleep(max(0.0, period - elapsed))


def _save_results(
    output: Path,
    *,
    anchors: list[int],
    pred_chunks: list[np.ndarray],
    gt_chunks: list[np.ndarray],
    qpos_rows: list[np.ndarray],
    actual_before_rows: list[np.ndarray],
    executed_counts: list[int],
    inference_latencies: list[float],
) -> None:
    if not pred_chunks:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    completed_anchors = anchors[: len(pred_chunks)]
    np.savez_compressed(
        output,
        anchors=np.asarray(completed_anchors, dtype=np.int64),
        pred_action=np.stack(pred_chunks).astype(np.float32),
        ground_truth_action=np.stack(gt_chunks).astype(np.float32),
        dataset_qpos=np.stack(qpos_rows).astype(np.float32),
        actual_before=np.stack(actual_before_rows).astype(np.float32),
        executed_counts=np.asarray(executed_counts, dtype=np.int64),
        inference_latency_s=np.asarray(inference_latencies, dtype=np.float64),
    )


def main() -> None:
    args = parse_args()
    episode = args.episode.expanduser().resolve()
    if not episode.is_file():
        raise SystemExit(f"Episode not found: {episode}")

    mapping = load_mapping_config(args.config)
    server = _section(mapping, "policy_server")
    host = str(args.host if args.host is not None else server.get("host", "127.0.0.1"))
    port = int(args.port)
    authkey = str(args.authkey)

    data_config = load_config(args.data_config)
    reader = HDF5EpisodeReader(episode, config=data_config)
    source = HDF5ObservationSource(
        episode_path=episode,
        config_path=args.data_config,
        prompt=args.prompt,
        loop=False,
    )
    length = int(source.length)
    all_anchors = frame_anchors(length, int(args.stride))
    anchors = all_anchors
    if args.max_anchors is not None:
        anchors = anchors[: max(0, int(args.max_anchors))]
    if not anchors:
        raise SystemExit("No frame anchors selected.")

    summary = reader.summary()
    print(json.dumps({"episode": str(episode), "summary": summary.__dict__, "anchors": anchors}, indent=2))
    _confirm_real_motion(args, anchors)

    robot: Piper14RobotController | None = None
    sink: RealPiper14ActionSink | None = None
    if args.execute_actions:
        robot = Piper14RobotController(mapping, no_robot=False)
        if not robot.connect():
            raise SystemExit("Failed to connect both Piper arms.")
        if args.move_to_dataset_start:
            # Use the configured, known battery_assemble pose as a waypoint
            # before the recorded start.  This mirrors the normal real runtime
            # and avoids one large current-state -> dataset-state command.
            print("[hdf5-replay] moving through configured battery initial pose")
            robot.move_initial()
            time.sleep(2.0)
        sink = RealPiper14ActionSink(robot, execute_actions=True)
        if args.move_to_dataset_start:
            dataset_start = reader.read_qpos(0)[:14]
            print(f"[hdf5-replay] moving to recorded qpos[0]: {dataset_start}")
            _move_to_dataset_qpos(
                robot,
                sink,
                dataset_start,
                control_hz=float(args.control_hz),
                frame=0,
            )

    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    qpos_rows: list[np.ndarray] = []
    executed_counts: list[int] = []
    inference_latencies: list[float] = []
    actual_before_rows: list[np.ndarray] = []

    with CosmosPiper14RemotePolicyClient(host=host, port=port, authkey=authkey) as policy:
        metadata = policy.metadata()
        validate_cosmos_metadata(metadata)
        horizon = int(metadata["action_horizon"])
        print(f"[hdf5-replay] policy metadata ok: {metadata}")

        for anchor_index, frame in enumerate(anchors):
            observation = dict(source.read_observation(frame))
            qpos = np.asarray(observation["state"], dtype=np.float32).reshape(14)
            if sink is not None and robot is not None and args.reset_to_dataset_qpos_each_anchor and frame != 0:
                _move_to_dataset_qpos(
                    robot,
                    sink,
                    qpos,
                    control_hz=float(args.control_hz),
                    frame=frame,
                )
            actual_before = (
                robot.get_status_and_state().astype(np.float32)
                if robot is not None
                else np.full((14,), np.nan, dtype=np.float32)
            )
            actual_before_rows.append(actual_before)
            if robot is not None:
                delta = actual_before - qpos
                print(
                    f"[hdf5-replay] frame={frame} actual_vs_dataset_l1={np.abs(delta).mean():.4f} "
                    f"actual_vs_dataset_linf={np.abs(delta).max():.4f}"
                )

            started = time.perf_counter()
            pred = np.asarray(policy.infer(observation), dtype=np.float32)
            latency = time.perf_counter() - started
            if pred.shape != (horizon, 14) or not np.isfinite(pred).all():
                raise RuntimeError(f"Invalid policy output at frame {frame}: shape={pred.shape}")

            # Execute only up to the next sampled dataset frame.  The appended
            # final frame is therefore inferred and executed once, not also as
            # the tail of the preceding chunk.
            frames_until_next = (
                all_anchors[anchor_index + 1] - frame if anchor_index + 1 < len(all_anchors) else 1
            )
            execute_count = min(frames_until_next, horizon, length - frame)
            gt = _read_ground_truth(reader, frame, execute_count)
            gt_padded = np.full((horizon, 14), np.nan, dtype=np.float32)
            gt_padded[: len(gt)] = gt
            mae = float(np.abs(pred[:execute_count] - gt).mean()) if execute_count else float("nan")
            pred_first_vs_qpos_linf = float(np.max(np.abs(pred[0] - qpos)))
            pred_internal_linf = (
                float(np.max(np.abs(np.diff(pred[:execute_count], axis=0)))) if execute_count > 1 else 0.0
            )
            print(
                f"[hdf5-replay] anchor={anchor_index}/{len(anchors)-1} frame={frame} "
                f"latency_s={latency:.3f} execute_count={execute_count} pred_vs_gt_mae={mae:.6f} "
                f"pred_first_vs_qpos_linf={pred_first_vs_qpos_linf:.6f} "
                f"pred_internal_linf={pred_internal_linf:.6f}"
            )
            print(f"[hdf5-replay] predicted_action_chunk[{frame}]=\n{pred}")

            pred_chunks.append(pred.copy())
            gt_chunks.append(gt_padded)
            qpos_rows.append(qpos.copy())
            executed_counts.append(execute_count)
            inference_latencies.append(latency)

            # Persist before physical execution so a later safety stop still
            # leaves the model output and ground-truth comparison available.
            _save_results(
                args.output,
                anchors=anchors,
                pred_chunks=pred_chunks,
                gt_chunks=gt_chunks,
                qpos_rows=qpos_rows,
                actual_before_rows=actual_before_rows,
                executed_counts=executed_counts,
                inference_latencies=inference_latencies,
            )

            if sink is not None:
                period = 1.0 / max(float(args.control_hz), 1e-6)
                for offset, action in enumerate(pred[:execute_count]):
                    step_started = time.perf_counter()
                    sink.send_action(frame + offset, action)
                    elapsed = time.perf_counter() - step_started
                    time.sleep(max(0.0, period - elapsed))

    print(f"[hdf5-replay] saved predictions and ground truth: {args.output}")


if __name__ == "__main__":
    main()
