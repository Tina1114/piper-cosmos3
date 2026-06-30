import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.preflight_cosmos3_piper14_sft import check_wandb_auth


class PreflightWandbAuthTest(unittest.TestCase):
    def test_online_mode_fails_without_batch_visible_credentials(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            toml = root / "config.toml"
            toml.write_text("[job]\nwandb_mode = 'online'\n", encoding="utf-8")

            report = check_wandb_auth(toml, {"HOME": str(root)})

        self.assertEqual(report["status"], "failed")
        self.assertIn("WANDB_API_KEY", report["details"]["failure_reason"])
        self.assertFalse(report["details"]["netrc_exists"])

    def test_online_mode_accepts_netrc_in_batch_home(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            toml = root / "config.toml"
            toml.write_text("[job]\nwandb_mode = 'online'\n", encoding="utf-8")
            (root / ".netrc").write_text("machine api.wandb.ai login user password key\n", encoding="utf-8")

            report = check_wandb_auth(toml, {"HOME": str(root)})

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["details"]["netrc_exists"])

    def test_extra_tail_overrides_can_force_online_mode(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            toml = root / "config.toml"
            toml.write_text("[job]\nwandb_mode = 'disabled'\n", encoding="utf-8")

            report = check_wandb_auth(
                toml,
                {
                    "HOME": str(root),
                    "EXTRA_TAIL_OVERRIDES": "job.wandb_mode=online trainer.max_iter=2",
                },
            )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["details"]["wandb_mode"], "online")
        self.assertEqual(report["details"]["wandb_mode_source"], "EXTRA_TAIL_OVERRIDES")


if __name__ == "__main__":
    unittest.main()
