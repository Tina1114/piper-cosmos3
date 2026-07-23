import json
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from piper_cosmos.quantization.fakequant_linear import W4A8FakeQuantLinear
from piper_cosmos.quantization.packed_linear import (
    W4A8CutlassLinear,
    pack_int4,
    unpack_int4,
)
from piper_cosmos.quantization import (
    QuantizationSpec,
    create_quantized_checkpoint,
    load_quantization_manifest,
    plan_quantized_checkpoint,
    prepare_quantized_checkpoint,
)
from piper_cosmos.quantization.registry import get_algorithm
from piper_cosmos.deployment.cosmos_piper14_policy import (
    CosmosPiper14PolicyClient,
    CosmosPiper14PolicyConfig,
)


class CosmosPiper14QuantizationTest(unittest.TestCase):
    def test_signed_int4_pack_layout_and_roundtrip(self) -> None:
        qweight = torch.tensor(
            [[0, 7, -1, -7, -8, 3], [1, -1, 6, -6, 2, -2]],
            dtype=torch.int8,
        )

        packed = pack_int4(qweight)

        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(tuple(packed.shape), (2, 3))
        self.assertEqual(
            packed[0].tolist(),
            [0x70, 0x9F, 0x38],
        )
        self.assertTrue(torch.equal(unpack_int4(packed), qweight))

    def test_grouped_rtn_roundtrip_is_finite_and_shape_preserving(self) -> None:
        tensor = torch.tensor(
            [[-1.0, -0.33, 0.25, 0.9, 1.7], [0.0, 0.1, 0.2, 0.3, 0.4]],
            dtype=torch.float32,
        )
        spec = QuantizationSpec(bits=4, group_size=4)
        algorithm = get_algorithm("rtn")

        quantized = algorithm.quantize(tensor, spec)
        restored = algorithm.dequantize(quantized, spec)

        self.assertEqual(tuple(restored.shape), tuple(tensor.shape))
        self.assertEqual(restored.dtype, tensor.dtype)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertFalse(torch.equal(restored, tensor))
        self.assertLess(float((restored - tensor).abs().max()), 0.15)

    def test_w4a8_linear_uses_output_channel_weight_and_token_activation_scales(self) -> None:
        linear = nn.Linear(4, 3, bias=True, dtype=torch.float32)
        with torch.no_grad():
            linear.weight.copy_(
                torch.tensor(
                    [
                        [-1.0, -0.4, 0.2, 0.9],
                        [-0.3, 0.1, 0.7, 1.3],
                        [-1.7, -0.2, 0.5, 0.8],
                    ]
                )
            )
            linear.bias.copy_(torch.tensor([0.1, -0.1, 0.2]))
        spec = QuantizationSpec(
            bits=4,
            activation_bits=8,
            group_size=0,
            weight_granularity="output_channel",
            activation_granularity="token",
        )
        module = W4A8FakeQuantLinear.from_linear(
            linear,
            algorithm=get_algorithm("rtn"),
            spec=spec,
        )
        activation = torch.tensor(
            [
                [[-1.0, -0.2, 0.3, 0.8], [0.0, 0.0, 0.0, 0.0]],
                [[0.2, 0.4, 0.7, 1.1], [-0.8, -0.1, 0.6, 0.9]],
            ],
            dtype=torch.float32,
        )

        output = module(activation)

        self.assertEqual(module.qweight.shape, (3, 4))
        self.assertEqual(module.weight_scale.shape, (3, 1))
        self.assertEqual(output.shape, (2, 2, 3))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.all(module.qweight >= -7))
        self.assertTrue(torch.all(module.qweight <= 7))

        activation_scale = activation.abs().amax(dim=-1, keepdim=True) / 127.0
        activation_scale = torch.where(
            activation_scale > 0,
            activation_scale,
            torch.ones_like(activation_scale),
        )
        activation_dq = torch.round(activation / activation_scale).clamp(-127, 127)
        activation_dq = activation_dq * activation_scale
        weight_dq = module.qweight.float() * module.weight_scale
        expected = torch.nn.functional.linear(activation_dq, weight_dq, module.bias)
        torch.testing.assert_close(output, expected)

    def test_fakequant_checkpoint_is_persistent_and_standard_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "quantized"
            source.mkdir()
            weight = torch.tensor(
                [
                    [-1.0, -0.7, -0.2, 0.4],
                    [0.15, 0.31, 0.66, 0.99],
                ],
                dtype=torch.float32,
            )
            bias = torch.tensor([0.1, -0.2], dtype=torch.float32)
            save_file({"model.linear.weight": weight}, source / "model-00001-of-00002.safetensors")
            save_file(
                {"model.linear.bias": bias, "model.step": torch.tensor(3, dtype=torch.int64)},
                source / "model-00002-of-00002.safetensors",
            )
            index = {
                "metadata": {"total_size": weight.numel() * 4 + bias.numel() * 4 + 8},
                "weight_map": {
                    "model.linear.weight": "model-00001-of-00002.safetensors",
                    "model.linear.bias": "model-00002-of-00002.safetensors",
                    "model.step": "model-00002-of-00002.safetensors",
                },
            }
            (source / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            (source / "config.json").write_text('{"model_type": "test"}\n', encoding="utf-8")

            spec = QuantizationSpec(bits=4, group_size=2)
            plan = plan_quantized_checkpoint(source, spec)
            self.assertEqual(plan["selected_tensors"], 1)
            self.assertEqual(plan["skipped_tensors"], 2)

            manifest = create_quantized_checkpoint(source, output, spec)
            self.assertEqual(manifest.statistics["quantized_tensors"], 1)
            self.assertTrue((output / "config.json").is_file())

            quantized_weight = load_file(output / "model-00001-of-00002.safetensors")[
                "model.linear.weight"
            ]
            unmodified = load_file(output / "model-00002-of-00002.safetensors")
            self.assertFalse(torch.equal(quantized_weight, weight))
            self.assertTrue(torch.equal(unmodified["model.linear.bias"], bias))
            self.assertEqual(int(unmodified["model.step"]), 3)

            loaded_manifest = load_quantization_manifest(output, required=True)
            self.assertIsNotNone(loaded_manifest)
            self.assertEqual(loaded_manifest.quantization.algo, "rtn")
            self.assertEqual(loaded_manifest.quantization.backend, "fakequant")

            runtime = prepare_quantized_checkpoint(
                output,
                algo="rtn",
                backend="fakequant",
                required=True,
            )
            self.assertTrue(runtime.active)
            self.assertEqual(Path(runtime.checkpoint), output)
            self.assertEqual(runtime.metadata()["bits"], 4)

    def test_cutlass_checkpoint_persists_packed_w4_and_installs_sidecar_module(
        self,
    ) -> None:
        class ToyModel(nn.Module):
            def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
                super().__init__()
                self.selected = nn.Linear(4, 2, bias=True, dtype=torch.float32)
                with torch.no_grad():
                    self.selected.weight.copy_(weight)
                    self.selected.bias.copy_(bias)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "packed"
            source.mkdir()
            weight = torch.tensor(
                [[-1.0, -0.4, 0.2, 0.9], [-0.3, 0.1, 0.7, 1.3]],
                dtype=torch.float32,
            )
            bias = torch.tensor([0.1, -0.2], dtype=torch.float32)
            shard = "model-00001-of-00001.safetensors"
            save_file(
                {
                    "model.selected.weight": weight,
                    "model.selected.bias": bias,
                },
                source / shard,
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": weight.numel() * 4 + bias.numel() * 4},
                        "weight_map": {
                            "model.selected.weight": shard,
                            "model.selected.bias": shard,
                        },
                    }
                ),
                encoding="utf-8",
            )
            spec = QuantizationSpec(
                backend="cutlass",
                bits=4,
                activation_bits=8,
                group_size=0,
                weight_granularity="output_channel",
                activation_granularity="token",
            )

            manifest = create_quantized_checkpoint(source, output, spec)
            tensors = load_file(output / shard)
            self.assertIn("model.selected.qweight", tensors)
            self.assertIn("model.selected.weight_scale", tensors)
            self.assertNotIn("model.selected.weight", tensors)
            self.assertEqual(tensors["model.selected.qweight"].dtype, torch.uint8)
            self.assertEqual(tuple(tensors["model.selected.qweight"].shape), (2, 2))
            self.assertEqual(
                manifest.storage_format,
                "cosmos-packed-w4a8-cutlass-v1",
            )

            runtime = prepare_quantized_checkpoint(
                output,
                backend="cutlass",
                required=True,
            )
            self.assertEqual(Path(runtime.checkpoint), source)
            self.assertEqual(Path(runtime.artifact_checkpoint), output)
            model = ToyModel(weight, bias)
            original_module = weakref.ref(model.selected)
            real_cutlass_linear = W4A8CutlassLinear

            def construct_after_release(**kwargs):
                self.assertIsNone(
                    original_module(),
                    "Original BF16 Linear must be released before constructing packed W4",
                )
                return real_cutlass_linear(**kwargs)

            with patch(
                "piper_cosmos.quantization.backends.cutlass.W4A8CutlassLinear",
                side_effect=construct_after_release,
            ):
                metadata = runtime.prepare_model(model)
            self.assertIsInstance(model.selected, W4A8CutlassLinear)
            self.assertEqual(metadata["replaced_linear_modules"], 1)

            activation = torch.tensor(
                [[-0.8, -0.1, 0.6, 0.9]],
                dtype=torch.float32,
            )
            output_value = model.selected.reference_forward(activation)
            self.assertEqual(tuple(output_value.shape), (1, 2))
            self.assertTrue(torch.isfinite(output_value).all())

    def test_runtime_rejects_manifest_mismatch_and_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                prepare_quantized_checkpoint(path, required=True)

            # A complete conversion is covered above; this minimal manifest is
            # enough to verify that user-selected algo/backend are enforced.
            payload = {
                "schema_version": 1,
                "quantization": QuantizationSpec().to_dict(),
                "storage_format": "hf-safetensors-fakequant-v1",
                "source": {},
                "statistics": {},
                "created_at_utc": "2026-01-01T00:00:00+00:00",
            }
            (path / "quantization_config.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "algo mismatch"):
                prepare_quantized_checkpoint(path, algo="gptq", backend="fakequant")
            with self.assertRaisesRegex(ValueError, "backend mismatch"):
                prepare_quantized_checkpoint(path, algo="rtn", backend="torchao")

    def test_policy_metadata_exposes_active_quantized_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            payload = {
                "schema_version": 1,
                "quantization": QuantizationSpec(bits=8, group_size=128).to_dict(),
                "storage_format": "hf-safetensors-fakequant-v1",
                "source": {},
                "statistics": {},
                "created_at_utc": "2026-01-01T00:00:00+00:00",
            }
            (path / "quantization_config.json").write_text(json.dumps(payload), encoding="utf-8")

            policy = CosmosPiper14PolicyClient(
                CosmosPiper14PolicyConfig(
                    checkpoint=str(path),
                    mock_backend=True,
                    require_quantized_checkpoint=True,
                )
            )

            metadata = policy.metadata()["quantization"]
            self.assertTrue(metadata["active"])
            self.assertEqual(metadata["algo"], "rtn")
            self.assertEqual(metadata["backend"], "fakequant")

    def test_fakequant_runtime_replaces_only_manifest_selected_linear(self) -> None:
        class ToyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.selected = nn.Linear(4, 3)
                self.excluded = nn.Linear(4, 3)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            spec = QuantizationSpec(
                bits=4,
                activation_bits=8,
                group_size=0,
                weight_granularity="output_channel",
                activation_granularity="token",
            )
            payload = {
                "schema_version": 1,
                "quantization": spec.to_dict(),
                "storage_format": "hf-safetensors-fakequant-v1",
                "source": {},
                "statistics": {
                    "quantized_tensor_names": ["model.selected.weight"],
                },
                "created_at_utc": "2026-01-01T00:00:00+00:00",
            }
            (path / "quantization_config.json").write_text(json.dumps(payload), encoding="utf-8")
            runtime = prepare_quantized_checkpoint(path, required=True)
            model = ToyModel()

            metadata = runtime.prepare_model(model)

            self.assertIsInstance(model.selected, W4A8FakeQuantLinear)
            self.assertIsInstance(model.excluded, nn.Linear)
            self.assertEqual(metadata["replaced_linear_modules"], 1)
            self.assertTrue(metadata["activation_quantization"])


if __name__ == "__main__":
    unittest.main()
