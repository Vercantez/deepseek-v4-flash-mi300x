import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "patches"
    / "apply-deepseek-v4-parser-recovery.py"
)
SPEC = importlib.util.spec_from_file_location("parser_recovery_patch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
parser_recovery_patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parser_recovery_patch)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "vllm-124154a88"
PINNED_FIXTURES = {
    parser_recovery_patch.ABSTRACT_PARSER_RELATIVE: (
        "87a1b225e95fcdc1dc20a0103647c8b49cfeca0ced2a942ea41447e628aa59a1"
    ),
    parser_recovery_patch.ADAPTERS_RELATIVE: (
        "dc1c1317dbfb298e54b8d94ca0e66d2b0cb1e481c35cdcc60a815284bd8a6ef7"
    ),
}


class ParserRecoveryPatchTests(unittest.TestCase):
    def make_package(self, package_root: Path) -> tuple[Path, Path]:
        abstract = package_root / parser_recovery_patch.ABSTRACT_PARSER_RELATIVE
        adapters = package_root / parser_recovery_patch.ADAPTERS_RELATIVE
        abstract.parent.mkdir(parents=True)
        adapters.parent.mkdir(parents=True, exist_ok=True)
        abstract.write_text(
            "class Parser:\n"
            "    def parse_delta(self, state, request):\n"
            + parser_recovery_patch.OLD_STREAMING_REASONING_ENTRY
            + "                previous_text=state.previous_text,\n"
            "            )\n"
            "            reasoning_parser = self._reasoning_parser\n"
            "            if reasoning_parser is not None and "
            "reasoning_parser.engine_based_streaming:\n"
            "                return delta_message\n"
        )
        adapters.write_text(
            "from __future__ import annotations\n"
            "from contextlib import contextmanager\n"
            "class Adapter:\n"
            "    @contextmanager\n"
            "    def _skip_tool_parsing(self):\n"
            "        yield\n"
            "    def extract_reasoning(\n"
            "        self,\n"
            "        model_output: str,\n"
            "        request: ChatCompletionRequest | ResponsesRequest,\n"
            + parser_recovery_patch.OLD_NONSTREAMING_REASONING_ENTRY
            + parser_recovery_patch.OLD_REASONING_ADJUST_REQUEST
            + "        return False\n"
        )
        return abstract, adapters

    def test_patches_pinned_blocks_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            package_root = Path(tempdir)
            abstract, adapters = self.make_package(package_root)

            parser_recovery_patch.apply_parser_recovery_patch(package_root)
            parser_recovery_patch.apply_parser_recovery_patch(package_root)
            abstract_source = abstract.read_text()
            adapters_source = adapters.read_text()

            self.assertIn("reasoning_parser.adjust_request(request)", abstract_source)
            self.assertIn("self.adjust_request(request)", adapters_source)
            self.assertIn(
                "self._parser_engine._check_skip_tool_parsing(request)",
                adapters_source,
            )

    def test_applies_to_exact_pinned_sources(self) -> None:
        pinned_abstract = (
            FIXTURE_ROOT / parser_recovery_patch.ABSTRACT_PARSER_RELATIVE
        )
        pinned_adapters = FIXTURE_ROOT / parser_recovery_patch.ADAPTERS_RELATIVE

        for relative_path, expected_sha256 in PINNED_FIXTURES.items():
            source = FIXTURE_ROOT / relative_path
            self.assertTrue(source.is_file(), f"missing pinned fixture: {source}")
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                expected_sha256,
                f"pinned fixture changed unexpectedly: {source}",
            )

        with tempfile.TemporaryDirectory() as tempdir:
            package_root = Path(tempdir)
            abstract = package_root / parser_recovery_patch.ABSTRACT_PARSER_RELATIVE
            adapters = package_root / parser_recovery_patch.ADAPTERS_RELATIVE
            abstract.parent.mkdir(parents=True)
            adapters.parent.mkdir(parents=True, exist_ok=True)
            abstract.write_text(pinned_abstract.read_text())
            adapters.write_text(pinned_adapters.read_text())

            parser_recovery_patch.apply_parser_recovery_patch(package_root)
            parser_recovery_patch.apply_parser_recovery_patch(package_root)

            self.assertIn(
                "reasoning_parser.adjust_request(request)", abstract.read_text()
            )
            self.assertIn(
                "self._parser_engine._check_skip_tool_parsing(request)",
                adapters.read_text(),
            )

    def test_refuses_unknown_source_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            package_root = Path(tempdir)
            abstract = package_root / parser_recovery_patch.ABSTRACT_PARSER_RELATIVE
            adapters = package_root / parser_recovery_patch.ADAPTERS_RELATIVE
            abstract.parent.mkdir(parents=True)
            adapters.parent.mkdir(parents=True, exist_ok=True)
            abstract.write_text("unknown abstract parser")
            adapters.write_text("unknown adapters")

            with self.assertRaisesRegex(RuntimeError, "unknown vLLM version"):
                parser_recovery_patch.apply_parser_recovery_patch(package_root)

            self.assertEqual(abstract.read_text(), "unknown abstract parser")
            self.assertEqual(adapters.read_text(), "unknown adapters")


if __name__ == "__main__":
    unittest.main()
