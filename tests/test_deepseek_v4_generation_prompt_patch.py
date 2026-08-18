import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "patches"
    / "apply-deepseek-v4-generation-prompt.py"
)
SPEC = importlib.util.spec_from_file_location("generation_prompt_patch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
generation_patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generation_patch)


class GenerationPromptPatchTests(unittest.TestCase):
    def make_package(self, package_root: Path) -> tuple[Path, Path]:
        tokenizer = package_root / generation_patch.TOKENIZER_RELATIVE
        encoding = package_root / generation_patch.ENCODING_RELATIVE
        tokenizer.parent.mkdir(parents=True)
        tokenizer.write_text(generation_patch.OLD_ENCODE_CONFIG)
        encoding.write_text(
            generation_patch.OLD_RENDER_SIGNATURE
            + generation_patch.OLD_WO_EOS
            + generation_patch.OLD_TRANSITION
            + generation_patch.OLD_ENCODE_SIGNATURE
            + generation_patch.OLD_CONTEXT_INITIALIZATION
            + generation_patch.OLD_RENDER_CALL
        )
        return tokenizer, encoding

    def test_patches_pinned_blocks_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            package_root = Path(tempdir)
            tokenizer, encoding = self.make_package(package_root)

            generation_patch.apply_generation_prompt_patch(package_root)
            generation_patch.apply_generation_prompt_patch(package_root)

            tokenizer_source = tokenizer.read_text()
            encoding_source = encoding.read_text()
            self.assertIn(
                'add_generation_prompt=kwargs.get("add_generation_prompt", True)',
                tokenizer_source,
            )
            self.assertIn(
                'continue_final_message=kwargs.get("continue_final_message", False)',
                tokenizer_source,
            )
            self.assertIn(
                "continue_final_message and index == len(messages) - 1",
                encoding_source,
            )
            self.assertIn(
                "add_generation_prompt\n        and index + 1 == len(messages)",
                encoding_source,
            )

    def test_validates_all_blocks_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            package_root = Path(tempdir)
            tokenizer, encoding = self.make_package(package_root)
            original_tokenizer = tokenizer.read_text()
            encoding.write_text(encoding.read_text().replace("    context =", "    ctx ="))

            with self.assertRaisesRegex(RuntimeError, "unknown vLLM version"):
                generation_patch.apply_generation_prompt_patch(package_root)

            self.assertEqual(tokenizer.read_text(), original_tokenizer)

    def test_refuses_unknown_source(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            package_root = Path(tempdir)
            tokenizer = package_root / generation_patch.TOKENIZER_RELATIVE
            encoding = package_root / generation_patch.ENCODING_RELATIVE
            tokenizer.parent.mkdir(parents=True)
            tokenizer.write_text("unknown tokenizer")
            encoding.write_text("unknown encoding")

            with self.assertRaisesRegex(RuntimeError, "unknown vLLM version"):
                generation_patch.apply_generation_prompt_patch(package_root)


if __name__ == "__main__":
    unittest.main()
