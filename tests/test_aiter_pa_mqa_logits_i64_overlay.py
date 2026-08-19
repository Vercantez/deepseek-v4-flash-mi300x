import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "patches" / "aiter_pa_mqa_logits.i64.py"


class AiterPagedMqaI64OverlayTests(unittest.TestCase):
    def test_production_block_256_branch_uses_64_bit_pointer_loads(self) -> None:
        source = OVERLAY.read_text()
        function_start = source.index(
            "def _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle("
        )
        branch_start = source.index(
            "    else:\n        context_idx = split_context_start", function_start
        )
        branch_end = source.index(
            "\n\n@gluon.jit\ndef _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx(",
            branch_start,
        )
        production_branch = source[branch_start:branch_end]

        self.assertNotIn("ptr=KV_buffer", production_branch)
        self.assertNotIn("ptr=scale_buffer", production_branch)
        self.assertEqual(
            production_branch.count(
                "context_kv_idx_next_0.to(tl.int64) * stride_k_seq"
            ),
            4,
        )
        self.assertEqual(
            production_branch.count(
                "context_kv_idx_next_0.to(tl.int64) * stride_scale_seq"
            ),
            4,
        )

    def test_production_configuration_selects_the_covered_branch(self) -> None:
        compose = (ROOT / "compose.yaml").read_text()
        sparse_indexer = (
            ROOT / "patches" / "rocm_aiter_mla_sparse.prefill-bh64.py"
        ).read_text()
        overlay = OVERLAY.read_text()

        self.assertIn('- --block-size\n      - "256"', compose)
        self.assertIn("ChunkK=256", sparse_indexer)
        self.assertIn("ChunkKPerStage: gl.constexpr = ChunkK // 2", overlay)
        self.assertEqual((256 // 2) % 256, 128)

    def test_shared_cache_stride_crosses_int32_boundary_at_block_2143(self) -> None:
        shared_cache_stride_bytes = 1_002_240
        self.assertLess(2_142 * shared_cache_stride_bytes, 2**31)
        self.assertGreater(2_143 * shared_cache_stride_bytes, 2**31)


if __name__ == "__main__":
    unittest.main()
