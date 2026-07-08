import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from piper_cosmos.deployment.cosmos_piper14_policy import CosmosPiper14PolicyConfig, LiberoActionServiceBackend


class CosmosPiper14BackendImportsTest(unittest.TestCase):
    def test_libero_backend_uses_common_checkpoint_overrides_import(self) -> None:
        calls = {}

        class FakeCheckpointOverrides:
            def __init__(self, **kwargs):
                calls["checkpoint_kwargs"] = kwargs

        class FakeActionServerArgs:
            def __init__(self, **kwargs):
                calls["server_args"] = kwargs

            def build_setup_overrides(self):
                setup = types.SimpleNamespace(guardrails=True)
                calls["setup_before_service"] = setup
                return setup

        class FakeActionModelService:
            def __init__(self, args):
                calls["service_args"] = args
                calls["guardrails"] = args.build_setup_overrides().guardrails

        common_args = types.ModuleType("cosmos_framework.inference.common.args")
        common_args.CheckpointOverrides = FakeCheckpointOverrides
        libero = types.ModuleType("cosmos_framework.scripts.action_policy_server_libero")
        libero.ActionServerArgs = FakeActionServerArgs
        libero.ActionModelService = FakeActionModelService
        transforms = types.ModuleType("cosmos_framework.data.vfm.action.transforms")
        transforms.ActionTransformPipeline = lambda **_: object()
        domain_utils = types.ModuleType("cosmos_framework.data.vfm.action.domain_utils")
        domain_utils.EMBODIMENT_TO_DOMAIN_ID = {}
        domain_utils.EMBODIMENT_TO_RAW_ACTION_DIM = {}

        fake_modules = {
            "cosmos_framework": types.ModuleType("cosmos_framework"),
            "cosmos_framework.inference": types.ModuleType("cosmos_framework.inference"),
            "cosmos_framework.inference.common": types.ModuleType("cosmos_framework.inference.common"),
            "cosmos_framework.inference.common.args": common_args,
            "cosmos_framework.scripts": types.ModuleType("cosmos_framework.scripts"),
            "cosmos_framework.scripts.action_policy_server_libero": libero,
            "cosmos_framework.data": types.ModuleType("cosmos_framework.data"),
            "cosmos_framework.data.vfm": types.ModuleType("cosmos_framework.data.vfm"),
            "cosmos_framework.data.vfm.action": types.ModuleType("cosmos_framework.data.vfm.action"),
            "cosmos_framework.data.vfm.action.transforms": transforms,
            "cosmos_framework.data.vfm.action.domain_utils": domain_utils,
        }

        with patch.dict(sys.modules, fake_modules):
            LiberoActionServiceBackend(CosmosPiper14PolicyConfig(checkpoint="/tmp/fake-ckpt"))

        self.assertEqual(calls["checkpoint_kwargs"], {"checkpoint_path": "/tmp/fake-ckpt"})
        self.assertIsInstance(calls["server_args"]["checkpoint"], FakeCheckpointOverrides)
        self.assertIsInstance(calls["service_args"], FakeActionServerArgs)
        self.assertFalse(calls["guardrails"])

    def test_predict_policy_uses_action_transform_pipeline_record(self) -> None:
        calls = {}

        class FakeCheckpointOverrides:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeActionServerArgs:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def build_setup_overrides(self):
                return types.SimpleNamespace(guardrails=False)

        class FakeModel:
            input_video_key = "video"

            def generate_samples_from_batch(self, batch, **kwargs):
                calls["batch"] = batch
                calls["generate_kwargs"] = kwargs
                action_rows = batch["action"][0][0].shape[0]
                return {"action": [FakeTensor((action_rows, 14))]}

        class FakeActionModelService:
            def __init__(self, args):
                self.cfg = types.SimpleNamespace(
                    action_chunk_size=args.kwargs["action_chunk_size"],
                    max_action_dim=args.kwargs["max_action_dim"],
                    fps=args.kwargs["fps"],
                    guidance=args.kwargs["guidance"],
                    seed=args.kwargs["seed"],
                    num_steps=args.kwargs["num_steps"],
                )
                self.model = FakeModel()
                self._lock = FakeLock()

            def _denormalize_action(self, action):
                return action

        class FakeActionTransformPipeline:
            def __init__(self, **kwargs):
                calls["transform_kwargs"] = kwargs

            def __call__(self, sample, resolution):
                calls["sample"] = sample
                sample = dict(sample)
                sample["action_processing_record"] = "record"
                sample["raw_action_dim"] = FakeTensor(())
                sample["sequence_plan"] = "sequence-plan"
                sample["image_size"] = FakeTensor((4,))
                sample["action"] = FakeTensor(sample["action"].shape)
                return sample

        class FakeLock:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape

            def __len__(self):
                return self.shape[0]

            def __getitem__(self, key):
                if isinstance(key, tuple):
                    first = key[0]
                    if isinstance(first, slice):
                        start, stop, step = first.indices(self.shape[0])
                        return FakeTensor((max(0, stop - start), *self.shape[1:]))
                    return self
                if isinstance(key, slice):
                    start, stop, step = key.indices(self.shape[0])
                    return FakeTensor((max(0, stop - start), *self.shape[1:]))
                return self

            def __setitem__(self, key, value):
                return None

            def permute(self, *dims):
                return FakeTensor(tuple(self.shape[dim] for dim in dims))

            def contiguous(self):
                return self

            def unsqueeze(self, dim):
                shape = list(self.shape)
                shape.insert(dim, 1)
                return FakeTensor(tuple(shape))

            def repeat(self, *repeats):
                return FakeTensor(tuple(size * repeats[idx] for idx, size in enumerate(self.shape)))

            def float(self):
                return self

            def squeeze(self, dim=0):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.ones(self.shape, dtype=np.float32)

        class FakeTorch(types.ModuleType):
            float32 = "float32"
            long = "long"

            def zeros(self, shape, dtype=None):
                return FakeTensor(tuple(shape))

            def from_numpy(self, value):
                return FakeTensor(tuple(np.asarray(value).shape))

            def tensor(self, value, dtype=None):
                return FakeTensor(())

            class inference_mode:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return None

        common_args = types.ModuleType("cosmos_framework.inference.common.args")
        common_args.CheckpointOverrides = FakeCheckpointOverrides
        libero = types.ModuleType("cosmos_framework.scripts.action_policy_server_libero")
        libero.ActionServerArgs = FakeActionServerArgs
        libero.ActionModelService = FakeActionModelService
        transforms = types.ModuleType("cosmos_framework.data.vfm.action.transforms")
        transforms.ActionTransformPipeline = FakeActionTransformPipeline
        joint_dataloader = types.ModuleType("cosmos_framework.data.vfm.joint_dataloader")
        joint_dataloader.IterativeJointDataLoader = types.SimpleNamespace(_MULTI_ITEM_KEYS={"video", "action"})
        domain_utils = types.ModuleType("cosmos_framework.data.vfm.action.domain_utils")
        domain_utils.EMBODIMENT_TO_DOMAIN_ID = {}
        domain_utils.EMBODIMENT_TO_RAW_ACTION_DIM = {}
        domain_utils.get_domain_id = lambda name: 21

        fake_modules = {
            "torch": FakeTorch("torch"),
            "cosmos_framework": types.ModuleType("cosmos_framework"),
            "cosmos_framework.inference": types.ModuleType("cosmos_framework.inference"),
            "cosmos_framework.inference.common": types.ModuleType("cosmos_framework.inference.common"),
            "cosmos_framework.inference.common.args": common_args,
            "cosmos_framework.scripts": types.ModuleType("cosmos_framework.scripts"),
            "cosmos_framework.scripts.action_policy_server_libero": libero,
            "cosmos_framework.data": types.ModuleType("cosmos_framework.data"),
            "cosmos_framework.data.vfm": types.ModuleType("cosmos_framework.data.vfm"),
            "cosmos_framework.data.vfm.action": types.ModuleType("cosmos_framework.data.vfm.action"),
            "cosmos_framework.data.vfm.action.transforms": transforms,
            "cosmos_framework.data.vfm.action.domain_utils": domain_utils,
            "cosmos_framework.data.vfm.joint_dataloader": joint_dataloader,
        }

        with patch.dict(sys.modules, fake_modules):
            backend = LiberoActionServiceBackend(
                CosmosPiper14PolicyConfig(checkpoint="/tmp/fake-ckpt", action_horizon=4, max_action_dim=64)
            )
            out = backend.predict_policy(
                {
                    "concat_view": np.zeros((6, 6, 3), dtype=np.uint8),
                    "state": np.zeros(14, dtype=np.float32),
                    "prompt": "Assemble the mouse's battery.",
                    "domain_name": "piper14",
                }
            )

        self.assertEqual(calls["transform_kwargs"]["max_action_dim"], 64)
        self.assertEqual(calls["sample"]["action"].shape, (5, 14))
        self.assertIn("action_processing_record", calls["batch"])
        self.assertEqual(calls["batch"]["action_processing_record"], ["record"])
        self.assertEqual(np.asarray(out["action"], dtype=np.float32).shape, (4, 14))


if __name__ == "__main__":
    unittest.main()
