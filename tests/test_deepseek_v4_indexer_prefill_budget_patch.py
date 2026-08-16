import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "patches"
    / "apply-deepseek-v4-indexer-prefill-budget.py"
)
SPEC = importlib.util.spec_from_file_location("indexer_prefill_patch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
indexer_patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(indexer_patch)


class IndexerPrefillBudgetPatchTests(unittest.TestCase):
    def test_patches_pinned_blocks_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            package_root = Path(tempdir)
            indexer = package_root / indexer_patch.INDEXER_RELATIVE
            indexer.parent.mkdir(parents=True)
            indexer.write_text(
                indexer_patch.OLD_BUDGET + indexer_patch.OLD_COMPRESSION
            )

            indexer_patch.apply_indexer_prefill_budget_patch(package_root)
            indexer_patch.apply_indexer_prefill_budget_patch(package_root)

            source = indexer.read_text()
            self.assertIn("// self.compress_ratio", source)
            self.assertLess(
                source.index("self.compress_ratio = 1"),
                source.index("self.max_prefill_buffer_size ="),
            )
            self.assertEqual(source.count("self.compress_ratio = 1"), 1)

    def test_refuses_unknown_source(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            package_root = Path(tempdir)
            indexer = package_root / indexer_patch.INDEXER_RELATIVE
            indexer.parent.mkdir(parents=True)
            indexer.write_text("unknown indexer")

            with self.assertRaisesRegex(RuntimeError, "unknown vLLM version"):
                indexer_patch.apply_indexer_prefill_budget_patch(package_root)


if __name__ == "__main__":
    unittest.main()
