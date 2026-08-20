import ast
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


OVERLAY_PATH = (
    Path(__file__).parents[1] / "patches" / "parser-engine.dsml-orphan.py"
)


class EventType(Enum):
    TEXT_CHUNK = auto()
    REASONING_START = auto()
    REASONING_CHUNK = auto()
    REASONING_END = auto()
    TOOL_CALL_START = auto()
    TOOL_NAME = auto()
    ARG_VALUE_CHUNK = auto()
    TOOL_CALL_END = auto()


@dataclass
class SemanticEvent:
    type: EventType
    value: str = ""
    tool_index: int = -1


@dataclass
class DeltaMessage:
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[object] | None = None


def load_events_to_delta():
    tree = ast.parse(OVERLAY_PATH.read_text())
    parser_engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ParserEngine"
    )
    method = next(
        node
        for node in parser_engine.body
        if isinstance(node, ast.FunctionDef) and node.name == "_events_to_delta"
    )
    namespace = {
        "DeltaMessage": DeltaMessage,
        "EventType": EventType,
        "SemanticEvent": SemanticEvent,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), OVERLAY_PATH, "exec"),
        namespace,
    )
    return namespace["_events_to_delta"]


class FakeParser:
    def __init__(self) -> None:
        self._deferred_content = ""
        self._suppress_tool_calls = False
        self._content_has_nonws = False
        self._tool_slots: list[object] = []
        self._drop_ws_only_content_before_tools = True
        self._reasoning_ended = False

    def _ensure_slot(self, index: int) -> None:
        while len(self._tool_slots) <= index:
            self._tool_slots.append(object())

    def _handle_tool_name(self, event: SemanticEvent) -> None:
        return

    def _handle_arg_chunk(
        self, event: SemanticEvent, deltas: list[object]
    ) -> None:
        deltas.append({"index": event.tool_index, "arguments": event.value})

    def _handle_tool_end(
        self, event: SemanticEvent, deltas: list[object]
    ) -> None:
        return

    def _coalesce_tool_call_deltas(self, deltas: list[object]) -> list[object]:
        return deltas


class ParserEngineDeferredContentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.convert = load_events_to_delta()
        self.parser = FakeParser()

    @staticmethod
    def tool_then_suffix() -> list[SemanticEvent]:
        return [
            SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            SemanticEvent(EventType.ARG_VALUE_CHUNK, "{}", tool_index=0),
            SemanticEvent(EventType.TEXT_CHUNK, " VISIBLE SUFFIX", tool_index=0),
        ]

    def test_finished_batch_returns_content_after_tool_delta(self) -> None:
        delta = self.convert(self.parser, self.tool_then_suffix(), finished=True)

        self.assertIsNotNone(delta)
        self.assertEqual(delta.content, " VISIBLE SUFFIX")
        self.assertTrue(delta.tool_calls)
        self.assertEqual(self.parser._deferred_content, "")

    def test_finished_batch_preserves_content_order_around_tool_delta(self) -> None:
        events = [
            SemanticEvent(EventType.TEXT_CHUNK, "VISIBLE PREFIX "),
            *self.tool_then_suffix(),
        ]

        delta = self.convert(self.parser, events, finished=True)

        self.assertIsNotNone(delta)
        self.assertEqual(delta.content, "VISIBLE PREFIX  VISIBLE SUFFIX")
        self.assertTrue(delta.tool_calls)
        self.assertEqual(self.parser._deferred_content, "")

    def test_unfinished_batch_defers_suffix_until_later_chunk(self) -> None:
        first = self.convert(self.parser, self.tool_then_suffix(), finished=False)

        self.assertIsNotNone(first)
        self.assertIsNone(first.content)
        self.assertEqual(self.parser._deferred_content, " VISIBLE SUFFIX")

        final = self.convert(self.parser, [], finished=True)
        self.assertIsNotNone(final)
        self.assertEqual(final.content, " VISIBLE SUFFIX")
        self.assertEqual(self.parser._deferred_content, "")


if __name__ == "__main__":
    unittest.main()
