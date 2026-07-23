import sys
import tempfile
import types
import unittest
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from piper_cosmos.deployment.cosmos_piper14_policy import (
    CosmosPiper14PolicyConfig,
    LiberoActionServiceBackend,
    _ExplicitCudaGraphCallable,
    _GenHiddenStateCollector,
    _profile_model_methods,
    _save_gen_hidden_state_profile,
    _save_vision_experiment,
    _SegmentedTimer,
)


class CosmosPiper14BackendImportsTest(unittest.TestCase):
    def test_explicit_cuda_graph_replays_with_new_tensor_values(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        @dataclass
        class Conditioning:
            bias: object
            frame_idx: int = 1

        def generator_core(value, *, conditioning):
            return {"value": value.square() + conditioning.bias + conditioning.frame_idx}

        graphed = _ExplicitCudaGraphCallable(
            generator_core,
            torch=torch,
            name="test_generator_core",
            warmup_iterations=1,
        )
        first_input = torch.tensor([2.0, 3.0], device="cuda")
        first_bias = torch.tensor([1.0, 1.0], device="cuda")
        first = graphed(first_input, conditioning=Conditioning(first_bias))["value"].clone()
        second_input = torch.tensor([4.0, 5.0], device="cuda")
        second_bias = torch.tensor([2.0, 3.0], device="cuda")
        second = graphed(
            second_input,
            conditioning=Conditioning(second_bias, frame_idx=2),
        )["value"].clone()
        third_input = torch.tensor([6.0, 7.0], device="cuda")
        third_bias = torch.tensor([0.0, 0.0], device="cuda")
        third = graphed(third_input, conditioning=Conditioning(third_bias))["value"].clone()
        torch.cuda.synchronize()

        torch.testing.assert_close(first, torch.tensor([6.0, 11.0], device="cuda"))
        torch.testing.assert_close(second, torch.tensor([20.0, 30.0], device="cuda"))
        torch.testing.assert_close(third, torch.tensor([37.0, 50.0], device="cuda"))

    def test_vision_experiment_saves_raw_future_latents_and_decoded_frames(self) -> None:
        import torch

        latent = torch.arange(1 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
        pred_video = torch.linspace(-1.0, 1.0, 3 * 5 * 2 * 2).reshape(3, 5, 2, 2)
        conditioning = np.zeros((4, 6, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            metadata = _save_vision_experiment(
                torch=torch,
                output_root=Path(tmp),
                artifact_id="request-1",
                vision_latent=latent,
                pred_video=pred_video,
                conditioning_image=conditioning,
                fps=30,
                observation_time_s=123.0,
                camera_timestamps_s={"cam_high": 123.0},
            )
            output_dir = Path(tmp) / "request-1"
            raw = torch.load(output_dir / "denoised_vision_latent.pt", weights_only=True)
            future = torch.load(output_dir / "future_vision_latent.pt", weights_only=True)

            self.assertTrue(torch.equal(raw, latent))
            self.assertTrue(torch.equal(future, latent[:, :, 1:]))
            self.assertEqual(len(list((output_dir / "predicted_frames").glob("frame_*.png"))), 5)
            self.assertTrue((output_dir / "pred_video.pt").is_file())
            self.assertEqual(metadata["predicted_frame_count"], 5)
            self.assertEqual(metadata["observation_time_s"], 123.0)

    def test_gen_hidden_state_profile_captures_adjacent_latent_frames(self) -> None:
        import torch

        class FakeLayer(torch.nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = float(scale)

            def forward(self, pack, **kwargs):
                del kwargs
                output = dict(pack)
                output["full_only_seq"] = pack["full_only_seq"] * self.scale
                return output, {}, None

        class FakeNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                layers = torch.nn.ModuleList([FakeLayer(1.0), FakeLayer(2.0)])
                self.language_model = types.SimpleNamespace(
                    model=types.SimpleNamespace(layers=layers)
                )
                self.config = types.SimpleNamespace(temporal_compression_factor_vision=4)

            def forward(self, packed_sequence, memory=None, und_only=False):
                del memory
                pack = {
                    "causal_seq": torch.zeros((1, 2)),
                    "full_only_seq": packed_sequence.full_hidden.clone(),
                    "_num_causal_tokens": 1,
                    "_num_full_tokens": packed_sequence.full_hidden.shape[0],
                }
                if und_only:
                    return pack
                for layer in self.language_model.model.layers:
                    pack, _, _ = layer(pack, gen_only=True, und_only=False)
                return pack

        vision_hidden = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],
                [4.0, 0.0],
                [4.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        packed = types.SimpleNamespace(
            vision=types.SimpleNamespace(
                token_shapes=[(3, 1, 2)],
                sequence_indexes=torch.arange(1, 7),
                timesteps=torch.full((4,), 0.5),
            ),
            attn_modes=["causal", "full"],
            split_lens=[1, 8],
            full_hidden=vision_hidden,
        )
        net = FakeNet()
        collector = _GenHiddenStateCollector(torch=torch, net=net, guidance=3.0)
        with collector:
            net(packed)
            net(packed)
        profile = collector.to_cpu_profile()

        self.assertEqual(profile["num_forward_calls"], 2)
        self.assertEqual(profile["num_layers"], 2)
        self.assertEqual(profile["latent_shape_thw"], [3, 1, 2])
        self.assertEqual(
            tuple(profile["frame_mean_hidden"].shape),
            (2, 2, 3, 2),
        )
        torch.testing.assert_close(
            profile["adjacent_mse"][0, 0],
            torch.tensor([0.5, 2.0]),
        )
        torch.testing.assert_close(
            profile["adjacent_mse"][0, 1],
            torch.tensor([2.0, 8.0]),
        )
        torch.testing.assert_close(
            profile["adjacent_cosine_similarity"][0, 0],
            torch.ones(2),
        )
        torch.testing.assert_close(
            profile["adjacent_relative_l2"][0, 0],
            torch.ones(2),
        )
        self.assertEqual(
            profile["cfg_branch_by_call"],
            ["conditional", "unconditional"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            summary = _save_gen_hidden_state_profile(
                torch=torch,
                output_dir=Path(tmp),
                profile=profile,
                predicted_frame_count=9,
            )
            self.assertTrue((Path(tmp) / "gen_hidden_state_profile.pt").is_file())
            self.assertTrue((Path(tmp) / "gen_hidden_state_adjacent.csv").is_file())
            self.assertTrue((Path(tmp) / "gen_hidden_state_profile.json").is_file())
            loaded = torch.load(
                Path(tmp) / "gen_hidden_state_profile.pt",
                weights_only=True,
            )
            self.assertEqual(
                tuple(loaded["frame_mean_hidden"].shape),
                (2, 2, 3, 2),
            )
            self.assertEqual(
                summary["decoded_frame_ranges_by_latent_frame"],
                [[0, 0], [1, 4], [5, 8]],
            )

    def test_gen_hidden_state_profile_rejects_acceleration_and_missing_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --vision-experiment-dir"):
            LiberoActionServiceBackend(
                CosmosPiper14PolicyConfig(gen_hidden_state_profile=True)
            )
        with self.assertRaisesRegex(ValueError, "requires eager GEN execution"):
            LiberoActionServiceBackend(
                CosmosPiper14PolicyConfig(
                    gen_hidden_state_profile=True,
                    vision_experiment_dir="/tmp/profile",
                    gen_torch_compile=True,
                )
            )

    def test_gen_acceleration_compiles_only_gen_path_and_wraps_cuda_graph_core(self) -> None:
        compile_calls = []

        class FakeLayer:
            def __init__(self) -> None:
                self.calls = []

            def forward(self, value, **kwargs):
                self.calls.append(kwargs)
                return value

        layer = FakeLayer()
        decoder = types.SimpleNamespace(
            layers=[layer],
            gen_only_forward=lambda value: value,
        )
        net = types.SimpleNamespace(
            language_model=types.SimpleNamespace(model=decoder),
            pad_for_cuda_graphs=False,
        )
        backend = LiberoActionServiceBackend.__new__(LiberoActionServiceBackend)
        backend.config = CosmosPiper14PolicyConfig(gen_torch_compile=True, gen_cuda_graphs=True)
        backend.service = types.SimpleNamespace(model=types.SimpleNamespace(net=net))

        def fake_compile(function, **kwargs):
            compile_calls.append(kwargs)

            def compiled(*args, **call_kwargs):
                compile_calls.append("called")
                return function(*args, **call_kwargs)

            return compiled

        with patch("torch.compile", fake_compile):
            backend._configure_gen_acceleration()
            layer.forward("prefill", und_only=True, gen_only=False)
            layer.forward("denoise", und_only=False, gen_only=True)

        self.assertEqual(
            compile_calls[0],
            {"fullgraph": True, "dynamic": False, "mode": None},
        )
        self.assertEqual(compile_calls[1], "called")
        self.assertEqual(layer.calls[0], {"und_only": True, "gen_only": False})
        self.assertEqual(layer.calls[1], {"gen_only": True, "und_only": False})
        self.assertTrue(net.pad_for_cuda_graphs)
        self.assertIsInstance(decoder.gen_only_forward, _ExplicitCudaGraphCallable)

    def test_cuda_graphs_can_be_enabled_without_torch_compile(self) -> None:
        class FakeLayer:
            def forward(self, value, **kwargs):
                return value

        layer = FakeLayer()
        decoder = types.SimpleNamespace(
            layers=[layer],
            gen_only_forward=lambda value: value,
        )
        net = types.SimpleNamespace(
            language_model=types.SimpleNamespace(model=decoder),
            pad_for_cuda_graphs=False,
        )
        backend = LiberoActionServiceBackend.__new__(LiberoActionServiceBackend)
        backend.config = CosmosPiper14PolicyConfig(
            gen_torch_compile=False,
            gen_cuda_graphs=True,
        )
        backend.service = types.SimpleNamespace(model=types.SimpleNamespace(net=net))

        with patch("torch.compile") as compile_mock:
            backend._configure_gen_acceleration()

        compile_mock.assert_not_called()
        self.assertIsInstance(decoder.gen_only_forward, _ExplicitCudaGraphCallable)
        self.assertTrue(net.pad_for_cuda_graphs)

    def test_instruction_cache_reuses_cond_and_uncond_memory_across_chunks(self) -> None:
        class FakeMemory:
            def is_gen_only(self):
                return True

        class FakeModel:
            def __init__(self) -> None:
                self.created = []

            def build_inference_memory_state(self):
                memory = FakeMemory()
                self.created.append(memory)
                return memory

        model = FakeModel()
        backend = LiberoActionServiceBackend.__new__(LiberoActionServiceBackend)
        backend.config = CosmosPiper14PolicyConfig(instruction_cache=True)
        backend.service = types.SimpleNamespace(model=model)
        backend._instruction_cache_namespace = "namespace"
        backend._instruction_memory_cache = OrderedDict()
        timer = _SegmentedTimer(True)

        with backend._instruction_cache_scope("prompt", timer):
            first = (model.build_inference_memory_state(), model.build_inference_memory_state())
        with backend._instruction_cache_scope("prompt", timer):
            second = (model.build_inference_memory_state(), model.build_inference_memory_state())

        self.assertEqual(len(model.created), 2)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertIn("model.reasoner.cache_miss", timer.snapshot())
        self.assertIn("model.reasoner.cache_hit", timer.snapshot())

    def test_model_timing_wrappers_record_and_restore_methods(self) -> None:
        class FakeModel:
            def _get_velocity(self, *, und_only=False):
                return und_only

        model = FakeModel()
        original_method = model._get_velocity.__func__
        timer = _SegmentedTimer(True)

        with _profile_model_methods(model, timer, synchronize_cuda=None):
            self.assertTrue(model._get_velocity(und_only=True))
            self.assertFalse(model._get_velocity(und_only=False))

        result = timer.snapshot()
        self.assertIn("model.reasoner.prefill", result)
        self.assertIn("model.denoise.velocity", result)
        self.assertIs(model._get_velocity.__func__, original_method)

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
        transforms = types.ModuleType("cosmos_framework.data.generator.action.transforms")
        transforms.ActionTransformPipeline = lambda **_: object()
        domain_utils = types.ModuleType("cosmos_framework.data.generator.action.domain_utils")
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
            "cosmos_framework.data.generator": types.ModuleType("cosmos_framework.data.generator"),
            "cosmos_framework.data.generator.action": types.ModuleType("cosmos_framework.data.generator.action"),
            "cosmos_framework.data.generator.action.transforms": transforms,
            "cosmos_framework.data.generator.action.domain_utils": domain_utils,
        }

        with patch.dict(sys.modules, fake_modules):
            LiberoActionServiceBackend(
                CosmosPiper14PolicyConfig(
                    checkpoint="/tmp/fake-ckpt",
                    vision_experiment_dir="/tmp/profile",
                    gen_hidden_state_profile=True,
                )
            )

        self.assertEqual(calls["checkpoint_kwargs"], {"checkpoint_path": "/tmp/fake-ckpt"})
        self.assertIsInstance(calls["server_args"]["checkpoint"], FakeCheckpointOverrides)
        self.assertIsInstance(calls["service_args"], FakeActionServerArgs)
        self.assertFalse(calls["guardrails"])
        self.assertFalse(calls["setup_before_service"].use_torch_compile)
        self.assertFalse(calls["setup_before_service"].use_cuda_graphs)

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
                calls["resolution"] = resolution
                sample = dict(sample)
                temporal_expand = sample.pop("inference_temporal_expand_after_resize", None)
                if temporal_expand is not None:
                    video = sample["video"]
                    sample["video"] = video.expand(-1, temporal_expand, -1, -1)
                calls["sample"] = sample
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

            def expand(self, *sizes):
                return FakeTensor(tuple(self.shape[idx] if size == -1 else size for idx, size in enumerate(sizes)))

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
        transforms = types.ModuleType("cosmos_framework.data.generator.action.transforms")
        transforms.ActionTransformPipeline = FakeActionTransformPipeline
        joint_dataloader = types.ModuleType("cosmos_framework.data.generator.joint_dataloader")
        joint_dataloader.IterativeJointDataLoader = types.SimpleNamespace(_MULTI_ITEM_KEYS={"video", "action"})
        domain_utils = types.ModuleType("cosmos_framework.data.generator.action.domain_utils")
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
            "cosmos_framework.data.generator": types.ModuleType("cosmos_framework.data.generator"),
            "cosmos_framework.data.generator.action": types.ModuleType("cosmos_framework.data.generator.action"),
            "cosmos_framework.data.generator.action.transforms": transforms,
            "cosmos_framework.data.generator.action.domain_utils": domain_utils,
            "cosmos_framework.data.generator.joint_dataloader": joint_dataloader,
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
        self.assertEqual(calls["resolution"], "480")
        self.assertEqual(calls["sample"]["action"].shape, (5, 14))
        self.assertEqual(calls["sample"]["video"].shape[1], 5)
        self.assertIn("action_processing_record", calls["batch"])
        self.assertEqual(calls["batch"]["action_processing_record"], ["record"])
        self.assertIs(calls["batch"]["inference_condition_only_vae"], True)
        self.assertIs(calls["batch"]["inference_instruction_cache"], True)
        self.assertEqual(len(calls["batch"]["inference_instruction_cache_namespace"]), 64)
        self.assertNotIn("inference_instruction_cache_dir", calls["batch"])
        self.assertEqual(calls["generate_kwargs"]["num_steps"], 4)
        self.assertEqual(calls["generate_kwargs"]["guidance"], 3.0)
        self.assertEqual(calls["generate_kwargs"]["shift"], 5.0)
        self.assertFalse(calls["generate_kwargs"]["has_negative_prompt"])
        self.assertEqual(np.asarray(out["action"], dtype=np.float32).shape, (4, 14))


if __name__ == "__main__":
    unittest.main()
