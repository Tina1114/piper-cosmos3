import unittest

import numpy as np

from piper_cosmos.deployment.cosmos_piper14_policy import (
    CosmosPiper14PolicyClient,
    CosmosPiper14PolicyConfig,
)


class CosmosPiper14PolicyPreprocessTest(unittest.TestCase):
    def test_uses_cosmos3_policy_sampling_defaults(self) -> None:
        config = CosmosPiper14PolicyConfig(mock_backend=True)

        self.assertEqual(config.num_steps, 4)
        self.assertEqual(config.guidance, 3.0)
        self.assertEqual(config.shift, 5.0)

    def test_composes_training_style_concat_view(self) -> None:
        policy = CosmosPiper14PolicyClient(
            CosmosPiper14PolicyConfig(
                mock_backend=True,
                camera_height=4,
                camera_width=6,
                action_horizon=3,
            )
        )
        obs = {
            "images": {
                "cam_high": np.full((4, 6, 3), [255, 0, 0], dtype=np.uint8),
                "cam_left_wrist": np.full((4, 6, 3), [0, 255, 0], dtype=np.uint8),
                "cam_right_wrist": np.full((4, 6, 3), [0, 0, 255], dtype=np.uint8),
            },
            "state": np.arange(14, dtype=np.float32),
        }

        request = policy.build_policy_request(obs)
        concat = request["concat_view"]

        self.assertEqual(concat.shape, (6, 6, 3))
        self.assertEqual(concat.dtype, np.uint8)
        self.assertTrue(np.all(concat[:4, :, :] == np.array([255, 0, 0], dtype=np.uint8)))
        self.assertTrue(np.all(concat[4:, :3, :] == np.array([0, 255, 0], dtype=np.uint8)))
        self.assertTrue(np.all(concat[4:, 3:, :] == np.array([0, 0, 255], dtype=np.uint8)))
        self.assertEqual(request["image_size"], 6)
        self.assertEqual(request["domain_name"], "piper14")

    def test_accepts_qpos_alias_and_returns_action_chunk(self) -> None:
        policy = CosmosPiper14PolicyClient(
            CosmosPiper14PolicyConfig(
                mock_backend=True,
                camera_height=4,
                camera_width=6,
                action_horizon=5,
            )
        )
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        obs = {
            "images": {
                "cam_high": image,
                "cam_left_wrist": image,
                "cam_right_wrist": image,
            },
            "qpos": np.linspace(-1.0, 1.0, 14, dtype=np.float32),
        }

        action = policy.infer(obs)

        self.assertEqual(action.shape, (5, 14))
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(np.isfinite(action).all())

    def test_rejects_missing_camera(self) -> None:
        policy = CosmosPiper14PolicyClient(CosmosPiper14PolicyConfig(mock_backend=True))

        with self.assertRaisesRegex(ValueError, "cam_right_wrist"):
            policy.update_observation(
                {
                    "images": {
                        "cam_high": np.zeros((4, 6, 3), dtype=np.uint8),
                        "cam_left_wrist": np.zeros((4, 6, 3), dtype=np.uint8),
                    },
                    "state": np.zeros(14, dtype=np.float32),
                }
            )

    def test_rejects_wrong_state_dim(self) -> None:
        policy = CosmosPiper14PolicyClient(CosmosPiper14PolicyConfig(mock_backend=True))
        image = np.zeros((4, 6, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "state/qpos dim 14"):
            policy.update_observation(
                {
                    "images": {
                        "cam_high": image,
                        "cam_left_wrist": image,
                        "cam_right_wrist": image,
                    },
                    "state": np.zeros(13, dtype=np.float32),
                }
            )


if __name__ == "__main__":
    unittest.main()
