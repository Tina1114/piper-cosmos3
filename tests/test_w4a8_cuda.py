"""GPU validation for the optional CUTLASS extension.

Run explicitly after building the extension.  CPU-only CI skips this module.
"""

from __future__ import annotations

import unittest

import torch

from piper_cosmos.quantization.packed_linear import (
    load_cutlass_extension,
    pack_int4,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class W4A8CudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.extension = load_cutlass_extension()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_cuda_unpack_is_bit_exact(self) -> None:
        torch.manual_seed(11)
        qweight = torch.randint(-8, 8, (64, 64), dtype=torch.int8)
        packed = pack_int4(qweight).cuda()

        actual = torch.ops.piper_w4a8.unpack_int4(packed)

        self.assertTrue(torch.equal(actual.cpu(), qweight))

    def test_w4a8_cutlass_matches_integer_reference(self) -> None:
        torch.manual_seed(13)
        rows, cols, inner = 33, 64, 64
        activation = torch.randn(
            rows,
            inner,
            device="cuda",
            dtype=torch.bfloat16,
        )
        qweight = torch.randint(
            -7,
            8,
            (cols, inner),
            device="cuda",
            dtype=torch.int8,
        )
        packed = pack_int4(qweight)
        weight_scale = torch.rand(
            cols,
            device="cuda",
            dtype=torch.float32,
        ) * 0.03 + 0.001
        bias = torch.randn(cols, device="cuda", dtype=torch.bfloat16)

        actual = torch.ops.piper_w4a8.linear(
            activation,
            packed,
            weight_scale,
            bias,
        )
        debug = torch.ops.piper_w4a8.linear_debug(
            activation,
            packed,
            weight_scale,
            bias,
        )

        work = activation.float()
        activation_scale = work.abs().amax(dim=-1, keepdim=True) / 127.0
        activation_scale = torch.where(
            activation_scale > 0,
            activation_scale,
            torch.ones_like(activation_scale),
        )
        activation_q = torch.round(work / activation_scale).clamp(-127, 127)
        expected = (
            (activation_q @ qweight.float().T)
            * activation_scale
            * weight_scale[None, :]
            + bias.float()
        ).bfloat16()

        torch.testing.assert_close(actual, debug, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0.03125)

    def test_fused_mainloop_matches_debug_on_residual_tiles(self) -> None:
        torch.manual_seed(19)
        for rows, cols, inner in (
            (1, 64, 64),
            (129, 128, 64),
            (257, 192, 128),
            (33, 64, 96),
        ):
            activation = torch.randn(
                rows,
                inner,
                device="cuda",
                dtype=torch.bfloat16,
            )
            qweight = torch.randint(
                -7,
                8,
                (cols, inner),
                device="cuda",
                dtype=torch.int8,
            )
            packed = pack_int4(qweight)
            scale = (
                torch.rand(cols, device="cuda", dtype=torch.float32)
                * 0.03
                + 0.001
            )
            fused = torch.ops.piper_w4a8.linear(
                activation,
                packed,
                scale,
                None,
            )
            debug = torch.ops.piper_w4a8.linear_debug(
                activation,
                packed,
                scale,
                None,
            )
            torch.testing.assert_close(fused, debug, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
