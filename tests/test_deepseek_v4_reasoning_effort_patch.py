import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "patches"
    / "apply-deepseek-v4-reasoning-effort.py"
)
SPEC = importlib.util.spec_from_file_location("reasoning_effort_patch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reasoning_patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reasoning_patch)


class ReasoningEffortPatchTests(unittest.TestCase):
    def test_patches_pinned_blocks_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            package_root = Path(tempdir)
            tokenizer = package_root / reasoning_patch.TOKENIZER_RELATIVE
            encoding = package_root / reasoning_patch.ENCODING_RELATIVE
            tokenizer.parent.mkdir(parents=True)
            tokenizer.write_text(reasoning_patch.OLD_NORMALIZATION)
            encoding.write_text(
                reasoning_patch.OLD_PROMPTS + reasoning_patch.OLD_RENDERING
            )

            reasoning_patch.apply_reasoning_effort_patch(package_root)
            reasoning_patch.apply_reasoning_effort_patch(package_root)

            tokenizer_source = tokenizer.read_text()
            encoding_source = encoding.read_text()
            self.assertIn('"minimal": "low"', tokenizer_source)
            self.assertIn('"medium": "high"', tokenizer_source)
            self.assertIn('"xhigh": "max"', tokenizer_source)
            self.assertIn('"low": ""', encoding_source)
            self.assertIn("Reasoning Effort: Absolute maximum", encoding_source)
            self.assertIn("Reasoning Effort: Beyond maximum", encoding_source)

    def test_refuses_unknown_source(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            package_root = Path(tempdir)
            tokenizer = package_root / reasoning_patch.TOKENIZER_RELATIVE
            encoding = package_root / reasoning_patch.ENCODING_RELATIVE
            tokenizer.parent.mkdir(parents=True)
            tokenizer.write_text("unknown tokenizer")
            encoding.write_text("unknown encoding")

            with self.assertRaisesRegex(RuntimeError, "unknown vLLM version"):
                reasoning_patch.apply_reasoning_effort_patch(package_root)


if __name__ == "__main__":
    unittest.main()
