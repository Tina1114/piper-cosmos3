from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def write_episode(path: Path, length: int = 20) -> None:
    with h5py.File(path, "w") as h5:
        obs = h5.create_group("observations")
        images = obs.create_group("images")
        for offset, camera in enumerate(CAMERAS):
            data = np.full((length, 8, 8, 3), fill_value=offset * 40, dtype=np.uint8)
            data[:, :, :, 0] += np.arange(length, dtype=np.uint8)[:, None, None]
            images.create_dataset(camera, data=data)
        qpos = np.arange(length * 14, dtype=np.float32).reshape(length, 14)
        qvel = qpos + 1000.0
        action = qpos + 2000.0
        obs.create_dataset("qpos", data=qpos)
        obs.create_dataset("qvel", data=qvel)
        h5.create_dataset("action", data=action)


def write_config(path: Path, data_root: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"root": str(data_root), "default_split": "perfect"},
                "hdf5": {
                    "image_keys": {
                        "cam_high": "/observations/images/cam_high",
                        "cam_left_wrist": "/observations/images/cam_left_wrist",
                        "cam_right_wrist": "/observations/images/cam_right_wrist",
                    },
                    "action_key": "/action",
                    "qpos_key": "/observations/qpos",
                    "qvel_key": "/observations/qvel",
                },
                "training": {
                    "history_frames": 2,
                    "action_horizon": 16,
                    "stride": 1,
                },
                "language": {
                    "default_instruction": "Put the three objects on the table into the container."
                },
            }
        ),
        encoding="utf-8",
    )


def test_piper_dual_dataset_returns_expected_sample_contract(tmp_path: Path) -> None:
    from piper_cosmos.data.piper_dual_dataset import PiperDualDataset

    data_root = tmp_path / "perfect"
    data_root.mkdir()
    episode_path = data_root / "episode_0.hdf5"
    write_episode(episode_path)
    config_path = tmp_path / "config.yaml"
    write_config(config_path, data_root)

    dataset = PiperDualDataset(
        data_root=data_root,
        config_path=config_path,
        history_frames=2,
        action_horizon=16,
        image_size=224,
        stride=1,
    )
    sample = dataset[0]

    assert len(dataset) == 4
    assert set(sample) == {"images", "qpos", "action", "instruction", "episode_path", "t"}
    assert set(sample["images"]) == set(CAMERAS)
    for image_history in sample["images"].values():
        assert image_history.shape == (2, 3, 224, 224)
        assert image_history.dtype == np.float32
        assert 0.0 <= float(image_history.min()) <= float(image_history.max()) <= 1.0
    assert sample["qpos"].shape == (14,)
    assert sample["qpos"].dtype == np.float32
    np.testing.assert_array_equal(sample["qpos"], np.arange(14, 28, dtype=np.float32))
    assert sample["action"].shape == (16, 14)
    assert sample["action"].dtype == np.float32
    np.testing.assert_array_equal(
        sample["action"][0], np.arange(14, 28, dtype=np.float32) + 2000.0
    )
    assert sample["instruction"] == "Put the three objects on the table into the container."
    assert sample["episode_path"] == str(episode_path)
    assert sample["t"] == 1
