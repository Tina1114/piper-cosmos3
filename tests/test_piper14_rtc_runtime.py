import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from piper_cosmos.deployment.piper14_rtc_runtime import (
    HDF5ObservationSource,
    Piper14RTCRuntime,
    Piper14RTCRuntimeConfig,
    RealTimeChunkingBuffer,
    RecordingActionSink,
)


class FakePolicyClient:
    def __init__(self, chunks):
        self.chunks = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
        self.calls = []

    def infer(self, obs):
        self.calls.append(obs)
        if not self.chunks:
            raise RuntimeError("no fake chunks left")
        return self.chunks.pop(0)


class FakeObservationSource:
    def __init__(self):
        self.calls = []

    def read_observation(self, t):
        self.calls.append(int(t))
        return {"state": np.full(14, t, dtype=np.float32), "images": {}, "prompt": "test"}


class Piper14RTCRuntimeTest(unittest.TestCase):
    def test_buffer_fuses_overlapping_chunks_with_recency_weight(self) -> None:
        buffer = RealTimeChunkingBuffer(chunk_size=3, exp_weight_factor=1.0)
        buffer.enqueue(np.ones((3, 2), dtype=np.float32), cursor=0)
        buffer.enqueue(np.full((3, 2), 3.0, dtype=np.float32), cursor=1)

        action = buffer.get_action(1)

        weights = np.exp(np.array([0.0, 1.0], dtype=np.float32))
        expected = (1.0 * weights[0] + 3.0 * weights[1]) / weights.sum()
        self.assertTrue(np.allclose(action, np.full(2, expected, dtype=np.float32)))

    def test_runtime_requests_chunks_and_records_one_action_per_control_step(self) -> None:
        chunks = [
            np.tile(np.arange(4, dtype=np.float32)[:, None], (1, 14)),
            np.tile((10 + np.arange(4, dtype=np.float32))[:, None], (1, 14)),
            np.tile((20 + np.arange(4, dtype=np.float32))[:, None], (1, 14)),
        ]
        policy = FakePolicyClient(chunks)
        source = FakeObservationSource()
        sink = RecordingActionSink()
        runtime = Piper14RTCRuntime(
            policy=policy,
            observation_source=source,
            action_sink=sink,
            config=Piper14RTCRuntimeConfig(max_steps=3, chunk_size=4, action_dim=14, replan_interval=1),
        )

        report = runtime.run()

        self.assertEqual(report["steps"], 3)
        self.assertEqual(report["num_inferences"], 3)
        self.assertEqual(source.calls, [0, 1, 2])
        self.assertEqual(sink.actions.shape, (3, 14))
        self.assertTrue(np.isfinite(sink.actions).all())
        self.assertEqual(report["selected_actions"]["shape"], [3, 14])

    def test_runtime_rejects_bad_action_chunk_shape(self) -> None:
        policy = FakePolicyClient([np.zeros((2, 13), dtype=np.float32)])
        runtime = Piper14RTCRuntime(
            policy=policy,
            observation_source=FakeObservationSource(),
            action_sink=RecordingActionSink(),
            config=Piper14RTCRuntimeConfig(max_steps=1, chunk_size=4, action_dim=14),
        )

        with self.assertRaisesRegex(ValueError, r"Expected action chunk"):
            runtime.run()

    def test_runtime_rejects_nonfinite_action_chunk(self) -> None:
        chunk = np.zeros((2, 14), dtype=np.float32)
        chunk[0, 0] = np.nan
        runtime = Piper14RTCRuntime(
            policy=FakePolicyClient([chunk]),
            observation_source=FakeObservationSource(),
            action_sink=RecordingActionSink(),
            config=Piper14RTCRuntimeConfig(max_steps=1, chunk_size=4, action_dim=14),
        )

        with self.assertRaisesRegex(ValueError, "non-finite"):
            runtime.run()

    def test_hdf5_observation_source_reads_policy_observation_format(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
            config.write_text(
                """
hdf5:
  image_keys:
    cam_high: /obs/high
    cam_left_wrist: /obs/left
    cam_right_wrist: /obs/right
  qpos_key: /robot/qpos
  action_key: /robot/action
language:
  default_instruction: Assemble test.
""".lstrip(),
                encoding="utf-8",
            )
            episode = root / "episode.hdf5"
            with h5py.File(episode, "w") as h5:
                h5.create_dataset("/obs/high", data=np.full((3, 4, 5, 3), 10, dtype=np.uint8))
                h5.create_dataset("/obs/left", data=np.full((3, 4, 5, 3), 20, dtype=np.uint8))
                h5.create_dataset("/obs/right", data=np.full((3, 4, 5, 3), 30, dtype=np.uint8))
                h5.create_dataset("/robot/qpos", data=np.arange(42, dtype=np.float32).reshape(3, 14))
                h5.create_dataset("/robot/action", data=np.zeros((3, 14), dtype=np.float32))

            source = HDF5ObservationSource(episode_path=episode, config_path=config)
            obs = source.read_observation(2)

        self.assertEqual(obs["prompt"], "Assemble test.")
        self.assertEqual(obs["state"].shape, (14,))
        self.assertTrue(np.allclose(obs["state"], np.arange(28, 42, dtype=np.float32)))
        self.assertEqual(obs["images"]["cam_high"].shape, (4, 5, 3))
        self.assertEqual(obs["images"]["cam_high"].dtype, np.uint8)
        self.assertEqual(int(obs["images"]["cam_left_wrist"][0, 0, 0]), 20)
        self.assertEqual(int(obs["images"]["cam_right_wrist"][0, 0, 0]), 30)


if __name__ == "__main__":
    unittest.main()
