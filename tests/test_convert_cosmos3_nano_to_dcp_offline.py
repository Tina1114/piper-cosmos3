import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from piper_cosmos.cosmos3.local_hf_assets import ENV_QWEN_SNAPSHOT
from piper_cosmos.cosmos3.local_hf_assets import ENV_WAN_VAE_PATH
from piper_cosmos.cosmos3.local_hf_assets import QWEN_REPOSITORY
from piper_cosmos.cosmos3.local_hf_assets import WAN_VAE_REPOSITORY
from piper_cosmos.cosmos3.local_hf_assets import bootstrap_local_hf_assets
from piper_cosmos.cosmos3.local_hf_assets import resolve_local_hf_assets
from piper_cosmos.cosmos3.local_hf_assets import seed_registry_paths
from types import SimpleNamespace


class LocalHfAssetsTest(unittest.TestCase):
    def test_seeds_only_requested_repositories_to_local_paths(self) -> None:
        qwen_path = Path("/tmp/qwen-snapshot")
        wan_path = Path("/tmp/Wan2.2_VAE.pth")
        untouched_hf = SimpleNamespace(repository="nvidia/Cosmos3-Nano", _path=None)
        qwen_hf = SimpleNamespace(repository=QWEN_REPOSITORY, _path=None)
        wan_hf = SimpleNamespace(repository=WAN_VAE_REPOSITORY, _path=None)
        checkpoints = [
            SimpleNamespace(hf=untouched_hf),
            SimpleNamespace(hf=qwen_hf),
            SimpleNamespace(hf=wan_hf),
        ]

        seeded = seed_registry_paths(
            checkpoints,
            {
                QWEN_REPOSITORY: qwen_path,
                WAN_VAE_REPOSITORY: wan_path,
            },
        )

        self.assertEqual(seeded[QWEN_REPOSITORY], 1)
        self.assertEqual(seeded[WAN_VAE_REPOSITORY], 1)
        self.assertEqual(qwen_hf._path, str(qwen_path))
        self.assertEqual(wan_hf._path, str(wan_path))
        self.assertIsNone(untouched_hf._path)

    def test_bootstrap_patches_qwen_tokenizer_download_to_local_snapshot(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            qwen_snapshot = root / "qwen"
            qwen_snapshot.mkdir()
            for name, contents in {
                "vocab.json": "{}",
                "merges.txt": "#version: 0.2\n",
                "tokenizer_config.json": '{\"chat_template\":\"{{ messages[0][\\\"content\\\"] }}\"}',
            }.items():
                (qwen_snapshot / name).write_text(contents, encoding="utf-8")
            wan_vae = root / "Wan2.2_VAE.pth"
            wan_vae.write_bytes(b"fake-vae")

            bootstrap_local_hf_assets(qwen_snapshot=qwen_snapshot, wan_vae_path=wan_vae)

            try:
                from cosmos_framework.configs.base.defaults import vlm
            except ImportError:
                from cosmos_framework.configs.base.defaults import reasoner as vlm

            self.assertEqual(vlm.download_tokenizer_files(QWEN_REPOSITORY, "hf"), str(qwen_snapshot))
            processor = vlm.create_qwen2_tokenizer_with_download(QWEN_REPOSITORY, "hf")
            self.assertEqual(processor.processor.tokenizer.chat_template, '{{ messages[0]["content"] }}')
            self.assertEqual(Path(self.assert_env(ENV_QWEN_SNAPSHOT)), qwen_snapshot)
            self.assertEqual(Path(self.assert_env(ENV_WAN_VAE_PATH)), wan_vae)

    def test_resolve_local_hf_assets_reports_missing_asset_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            qwen_snapshot = root / "qwen"
            qwen_snapshot.mkdir()
            (qwen_snapshot / "vocab.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_local_hf_assets(
                    qwen_snapshot=qwen_snapshot,
                    wan_vae_path=root / "missing" / "Wan2.2_VAE.pth",
                )

        message = str(ctx.exception)
        self.assertIn("missing Qwen tokenizer assets", message)
        self.assertIn("merges.txt", message)
        self.assertIn("tokenizer_config.json", message)
        self.assertIn("missing Wan VAE asset", message)

    def assert_env(self, name: str) -> str:
        value = __import__("os").environ.get(name)
        self.assertIsNotNone(value)
        return value


if __name__ == "__main__":
    unittest.main()
