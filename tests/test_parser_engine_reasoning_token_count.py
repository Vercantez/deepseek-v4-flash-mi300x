import ast
import unittest
from collections.abc import Sequence
from pathlib import Path


OVERLAY_PATH = (
    Path(__file__).parents[1] / "patches" / "parser-engine.dsml-orphan.py"
)


def load_count_reasoning_tokens():
    tree = ast.parse(OVERLAY_PATH.read_text())
    parser_engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ParserEngine"
    )
    method = next(
        node
        for node in parser_engine.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "count_reasoning_tokens"
    )
    namespace = {"Sequence": Sequence}
    exec(compile(ast.Module(body=[method], type_ignores=[]), OVERLAY_PATH, "exec"), namespace)
    return namespace["count_reasoning_tokens"]


class ParserEngineReasoningTokenCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.count = load_count_reasoning_tokens()
        self.parser = type(
            "Parser",
            (),
            {
                "_reasoning_start_token_id": 100,
                "_reasoning_end_token_id": 101,
            },
        )()

    def test_counts_prompt_opened_reasoning(self) -> None:
        self.assertEqual(self.count(self.parser, [11, 12, 101, 99]), 2)

    def test_preserves_explicit_and_nested_spans(self) -> None:
        token_ids = [99, 100, 11, 100, 12, 101, 13, 101, 98]
        self.assertEqual(self.count(self.parser, token_ids), 3)

    def test_does_not_count_content_without_reasoning_boundaries(self) -> None:
        self.assertEqual(self.count(self.parser, [11, 12, 99]), 0)


if __name__ == "__main__":
    unittest.main()
