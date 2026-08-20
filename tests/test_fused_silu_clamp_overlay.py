import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "patches/gpt_oss_triton_kernels_moe.pack128-fused-silu-fast-routing.py"


class FusedSiluClampOverlayTests(unittest.TestCase):
    def test_fused_silu_uses_checkpoint_clamp_limit(self) -> None:
        source = OVERLAY.read_text()
        tree = ast.parse(source)
        silu_mul = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_silu_mul"
        )

        self.assertEqual([arg.arg for arg in silu_mul.args.args], ["inp", "limit"])
        self.assertIn("gate = tl.minimum(gate, limit)", source)
        self.assertIn(
            "up = tl.maximum(tl.minimum(up, limit), -limit)", source
        )
        self.assertIn("limit = quant_config.gemm1_clamp_limit", source)
        self.assertIn(
            'FnSpecs("silu_mul", _silu_mul, ("limit",), reduction_n=2)',
            source,
        )


if __name__ == "__main__":
    unittest.main()
