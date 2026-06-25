from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import scan_missing_metadata


def test_metadata_scan_skips_large_irrelevant_directories(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored.py").write_text("fps = 999\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "ignored.py").write_text("gripper = 1\n", encoding="utf-8")
    (tmp_path / "collector.py").write_text("fps = 30\n", encoding="utf-8")

    files = scan_missing_metadata.iter_files([tmp_path])

    assert files == [tmp_path / "collector.py"]


def test_compute_dataset_stats_reads_numeric_hdf5_without_images(tmp_path: Path) -> None:
    from scripts import compute_dataset_stats

    data_root = tmp_path / "perfect"
    data_root.mkdir()
    episode = data_root / "episode_0.hdf5"
    action = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    qpos = action + 10.0
    qvel = action + 20.0
    with h5py.File(episode, "w") as h5:
        h5.create_dataset("/action", data=action)
        h5.create_dataset("/observations/qpos", data=qpos)
        h5.create_dataset("/observations/qvel", data=qvel)
        h5.create_dataset(
            "/observations/images/cam_high",
            data=np.zeros((3, 2, 2, 3), dtype=np.uint8),
        )

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "hdf5": {
                    "action_key": "/action",
                    "qpos_key": "/observations/qpos",
                    "qvel_key": "/observations/qvel",
                },
                "action": {
                    "order": [
                        "left_waist",
                        "left_shoulder",
                        "left_elbow",
                        "left_gripper",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    stats = compute_dataset_stats.compute_stats(data_root, config)

    assert json.loads(json.dumps(stats)) == stats
    assert stats["num_files"] == 1
    assert stats["num_steps"] == 3
    assert stats["episode_length_min"] == 3
    assert stats["episode_length_max"] == 3
    assert stats["episode_length_mean"] == 3.0
    assert stats["action_mean"] == [2.0, 3.0, 4.0, 5.0]
    assert stats["action_min"] == [1.0, 2.0, 3.0, 4.0]
    assert stats["action_max"] == [3.0, 4.0, 5.0, 6.0]
    assert stats["qpos_min"] == [11.0, 12.0, 13.0, 14.0]
    assert stats["qvel_max"] == [23.0, 24.0, 25.0, 26.0]
    assert stats["left_gripper_min"] == 4.0
    assert stats["left_gripper_max"] == 6.0
    assert stats["right_gripper_min"] is None
    assert stats["right_gripper_max"] is None
