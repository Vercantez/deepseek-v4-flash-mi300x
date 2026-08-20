"""Parser-level coverage for DeepSeek V4 orphan DSML recovery.

These tests load the production overlays with a text-only lexer so they
run without vLLM or a GPU. They close vLLM #49117 limitation 2: a
thinking request that never emits ``</think>`` or the ``tool_calls``
wrapper still recovers a declared invoke from ``ParserState.REASONING``.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "patches"


# ── Minimal vLLM parser engine stubs (text-only, no tokenizer) ───────


class EventType(Enum):
    TEXT_CHUNK = auto()
    REASONING_START = auto()
    REASONING_CHUNK = auto()
    REASONING_END = auto()
    TOOL_CALL_START = auto()
    TOOL_NAME = auto()
    ARG_VALUE_CHUNK = auto()
    TOOL_CALL_END = auto()


@dataclass(slots=True)
class SemanticEvent:
    type: EventType
    value: str = ""
    tool_index: int = -1


CONTENT_TERMINAL = "__CONTENT__"
DROP_TERMINAL = "__DROP__"


@dataclass(slots=True)
class TerminalDef:
    name: str
    pattern: re.Pattern[str]
    is_literal: bool = False
    literal: str = ""


@dataclass(slots=True)
class LexToken:
    terminal: str
    value: str


class LexerShape:
    def __init__(self, terminals: list[TerminalDef]) -> None:
        self.terminals = sorted(
            terminals,
            key=lambda t: (not t.is_literal, -len(t.literal or t.pattern.pattern)),
        )
        literal_strings = [
            (t.literal, t.name) for t in self.terminals if t.is_literal
        ]
        self.literal_strings = literal_strings
        self.max_literal_len = max((len(lit) for lit, _ in literal_strings), default=0)
        self.literal_first_chars = frozenset(lit[0] for lit, _ in literal_strings if lit)
        self.has_only_literals = all(t.is_literal for t in terminals)
        prefix_set: set[str] = set()
        for lit, _ in literal_strings:
            for i in range(1, len(lit)):
                prefix_set.add(lit[:i])
        self.prefix_set = frozenset(prefix_set)
        by_first: dict[str, list[tuple[str, str]]] = {}
        for lit, name in literal_strings:
            if lit:
                by_first.setdefault(lit[0], []).append((lit, name))
        self.literals_by_first = by_first


class IncrementalLexer:
    def __init__(
        self,
        terminals: list[TerminalDef] | LexerShape,
        content_terminal: str = CONTENT_TERMINAL,
    ) -> None:
        shape = terminals if isinstance(terminals, LexerShape) else LexerShape(terminals)
        self._shape = shape
        self.terminals = shape.terminals
        self.content_terminal = content_terminal
        self.buffer = ""
        self._literal_strings = shape.literal_strings
        self._max_literal_len = shape.max_literal_len
        self._literal_first_chars = shape.literal_first_chars
        self._has_only_literals = shape.has_only_literals
        self._prefix_set = shape.prefix_set
        self._literals_by_first = shape.literals_by_first

    def reset(self) -> None:
        self.buffer = ""

    def feed(self, text: str) -> list[LexToken]:
        if not self.buffer and self._has_only_literals and self._literal_first_chars:
            if all(ch not in self._literal_first_chars for ch in text):
                return [LexToken(self.content_terminal, text)]
        self.buffer += text
        return self._drain()

    def flush(self) -> list[LexToken]:
        tokens: list[LexToken] = []
        if self.buffer:
            tokens.extend(self._drain(final=True))
        if self.buffer:
            tokens.append(LexToken(self.content_terminal, self.buffer))
            self.buffer = ""
        return tokens

    def _drain(self, *, final: bool = False) -> list[LexToken]:
        tokens: list[LexToken] = []
        while self.buffer:
            if self._has_only_literals and self._literal_first_chars:
                if all(ch not in self._literal_first_chars for ch in self.buffer):
                    tokens.append(LexToken(self.content_terminal, self.buffer))
                    self.buffer = ""
                    break
            best_match: tuple[str, str, int] | None = None
            first = self.buffer[0]
            for lit, name in self._literals_by_first.get(first, ()):
                if self.buffer.startswith(lit) and (
                    best_match is None or len(lit) > best_match[2]
                ):
                    best_match = (name, lit, len(lit))
            if self.buffer in self._prefix_set and not final:
                if best_match is not None:
                    longer = any(
                        len(lit) > best_match[2] and lit.startswith(self.buffer)
                        for lit, _ in self._literals_by_first.get(first, ())
                    )
                    if not longer:
                        tokens.append(LexToken(best_match[0], best_match[1]))
                        self.buffer = self.buffer[best_match[2] :]
                        continue
                    break
                break
            if best_match is not None:
                tokens.append(LexToken(best_match[0], best_match[1]))
                self.buffer = self.buffer[best_match[2] :]
            else:
                content_end = self._find_content_boundary()
                if content_end > 0:
                    tokens.append(LexToken(self.content_terminal, self.buffer[:content_end]))
                    self.buffer = self.buffer[content_end:]
                else:
                    tokens.append(LexToken(self.content_terminal, self.buffer[0]))
                    self.buffer = self.buffer[1:]
        return tokens

    def _find_content_boundary(self) -> int:
        buf = self.buffer
        n = len(buf)
        for i in range(1, n):
            if buf[i] not in self._literal_first_chars:
                continue
            remaining = n - i
            for lit, _ in self._literal_strings:
                check_len = min(remaining, len(lit))
                if buf[i : i + check_len] == lit[:check_len]:
                    return i
        return n


def terminals_from_literals(literals: dict[str, str]) -> list[TerminalDef]:
    return [
        TerminalDef(
            name=name,
            pattern=re.compile(re.escape(lit)),
            is_literal=True,
            literal=lit,
        )
        for name, lit in literals.items()
    ]


@dataclass(slots=True)
class TextChunk:
    text: str


@dataclass(slots=True)
class PreLexedTerminal:
    terminal: str
    token_id: int
    text: str


LexerInput = TextChunk | PreLexedTerminal


class TokenIDScanner:
    """Text-only scanner: token IDs are unused in these parser tests."""

    def __init__(self, token_id_to_terminal, tokenizer) -> None:
        self.token_id_to_terminal = token_id_to_terminal
        self.tokenizer = tokenizer
        self._deferred_terminals: list[PreLexedTerminal] = []

    def reset(self) -> None:
        self._deferred_terminals.clear()

    def scan(self, delta_text: str, delta_token_ids) -> list[LexerInput]:
        return [TextChunk(delta_text)] if delta_text else []

    def flush_pending(self) -> list[LexerInput]:
        return []


def _install_parser_stubs() -> None:
    """Register overlay-compatible ``vllm.parser`` modules once."""
    if "vllm.parser.engine.streaming_parser_engine" in sys.modules:
        return

    events = types.ModuleType("vllm.parser.engine.events")
    events.EventType = EventType
    events.SemanticEvent = SemanticEvent

    lexer = types.ModuleType("vllm.parser.engine.incremental_lexer")
    lexer.CONTENT_TERMINAL = CONTENT_TERMINAL
    lexer.IncrementalLexer = IncrementalLexer
    lexer.LexerShape = LexerShape
    lexer.LexToken = LexToken
    lexer.TerminalDef = TerminalDef
    lexer.terminals_from_literals = terminals_from_literals

    scanner = types.ModuleType("vllm.parser.engine.token_id_scanner")
    scanner.DROP_TERMINAL = DROP_TERMINAL
    scanner.LexerInput = LexerInput
    scanner.PreLexedTerminal = PreLexedTerminal
    scanner.TextChunk = TextChunk
    scanner.TokenIDScanner = TokenIDScanner

    for name in (
        "vllm",
        "vllm.parser",
        "vllm.parser.engine",
        "vllm.parser.deepseek_v4",
        "vllm.tool_parsers",
        "vllm.tokenizers",
        "vllm.tool_parsers.abstract_tool_parser",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    sys.modules["vllm.parser.engine.events"] = events
    sys.modules["vllm.parser.engine.incremental_lexer"] = lexer
    sys.modules["vllm.parser.engine.token_id_scanner"] = scanner
    sys.modules.setdefault("regex", re)

    config_mod = _load_overlay(
        "vllm.parser.engine.parser_engine_config",
        PATCHES / "parser-engine-config.dsml-orphan.py",
    )
    sys.modules["vllm.parser.engine.parser_engine_config"] = config_mod

    engine_mod = _load_overlay(
        "vllm.parser.engine.streaming_parser_engine",
        PATCHES / "streaming-parser-engine.dsml-orphan.py",
    )
    sys.modules["vllm.parser.engine.streaming_parser_engine"] = engine_mod

    parser_engine = types.ModuleType("vllm.parser.engine.parser_engine")
    parser_engine.ParserEngine = object
    sys.modules["vllm.parser.engine.parser_engine"] = parser_engine

    utils = types.ModuleType("vllm.tool_parsers.utils")
    utils.find_tool_properties = lambda *args, **kwargs: {}
    sys.modules["vllm.tool_parsers.utils"] = utils

    v4 = _load_overlay(
        "vllm.parser.deepseek_v4",
        PATCHES / "parser-deepseek-v4.dsml-orphan.py",
    )
    sys.modules["vllm.parser.deepseek_v4"] = v4
    v32 = _load_overlay(
        "vllm.parser.deepseek_v32",
        PATCHES / "parser-deepseek-v32.dsml-orphan.py",
    )
    sys.modules["vllm.parser.deepseek_v32"] = v32


def _load_overlay(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_install_parser_stubs()

from vllm.parser.deepseek_v4 import (  # noqa: E402
    DSML_INVOKE_END,
    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
    DSML_THINK_END,
    DSML_TOOL_END,
    DSML_TOOL_START,
    deepseek_v4_config,
)
from vllm.parser.deepseek_v32 import deepseek_v32_config  # noqa: E402
from vllm.parser.engine.parser_engine_config import ParserState  # noqa: E402
from vllm.parser.engine.streaming_parser_engine import (  # noqa: E402
    StreamingParserEngine,
)


_DSML = "｜DSML｜"
_PARAM_CLOSE = f"</{_DSML}parameter>"


def _param(name: str, is_str: str, value: str) -> str:
    return (
        f'<{_DSML}parameter name="{name}" string="{is_str}">'
        f"{value}{_PARAM_CLOSE}"
    )


def _invoke(name: str, *params: tuple[str, str, str]) -> str:
    body = "\n".join(_param(n, s, v) for n, s, v in params)
    return (
        f"{DSML_INVOKE_PREFIX}{name}{DSML_INVOKE_NAME_END}\n"
        f"{body}\n"
        f"{DSML_INVOKE_END}"
    )


def _wrapped(*invokes: str) -> str:
    return DSML_TOOL_START + "\n".join(invokes) + DSML_TOOL_END


def _make_engine(
    *,
    thinking: bool = True,
    allowed: frozenset[str] | None = frozenset({"apply_patch"}),
    suppress: bool = False,
) -> StreamingParserEngine:
    engine = StreamingParserEngine(deepseek_v4_config(thinking=thinking), tokenizer=None)
    engine.allowed_tool_names = allowed
    engine.suppress_tool_calls = suppress
    return engine


def _event_types(events: list[SemanticEvent]) -> list[EventType]:
    return [event.type for event in events]


def _joined(events: list[SemanticEvent], event_type: EventType) -> str:
    return "".join(event.value for event in events if event.type == event_type)


def _tool_names(events: list[SemanticEvent]) -> list[str]:
    names: dict[int, list[str]] = {}
    for event in events:
        if event.type == EventType.TOOL_NAME:
            names.setdefault(event.tool_index, []).append(event.value)
    return ["".join(parts) for _, parts in sorted(names.items())]


class ReasoningOrphanInvokeConfigTests(unittest.TestCase):
    def test_reasoning_invoke_is_provisional_until_invoke_end(self) -> None:
        config = deepseek_v4_config(thinking=True)
        content = config.transitions[(ParserState.CONTENT, "INVOKE_PREFIX")]
        reasoning = config.transitions[(ParserState.REASONING, "INVOKE_PREFIX")]
        invoke_end = config.transitions[(ParserState.TOOL_ARGS, "INVOKE_END")]
        self.assertTrue(content.provisional_tool_call)
        self.assertEqual(content.events, (EventType.TOOL_CALL_START,))
        self.assertTrue(reasoning.provisional_tool_call)
        self.assertEqual(
            reasoning.events,
            (EventType.REASONING_END, EventType.TOOL_CALL_START),
        )
        self.assertEqual(reasoning.next_state, ParserState.TOOL_NAME)
        self.assertTrue(invoke_end.commit_provisional_tool_call)


class ReasoningOrphanInvokeTests(unittest.TestCase):
    def test_declared_orphan_invoke_in_reasoning_emits_tool_call(self) -> None:
        engine = _make_engine()
        text = (
            "I will apply the edit now.\n"
            + _invoke("apply_patch", ("input", "true", "*** Begin Patch\n*** End Patch"))
        )
        events = engine.parse_complete(text)
        types = _event_types(events)
        self.assertIn(EventType.REASONING_END, types)
        self.assertIn(EventType.TOOL_CALL_START, types)
        self.assertLess(
            types.index(EventType.REASONING_END),
            types.index(EventType.TOOL_CALL_START),
        )
        self.assertEqual(_tool_names(events), ["apply_patch"])
        self.assertIn("I will apply the edit now.", _joined(events, EventType.REASONING_CHUNK))
        self.assertNotIn(DSML_INVOKE_PREFIX, _joined(events, EventType.REASONING_CHUNK))

    def test_undeclared_orphan_invoke_in_reasoning_stays_reasoning(self) -> None:
        engine = _make_engine()
        text = (
            "Still thinking about the change.\n"
            + _invoke("not_a_real_tool", ("input", "true", "nope"))
        )
        events = engine.parse_complete(text)
        self.assertNotIn(EventType.TOOL_CALL_START, _event_types(events))
        self.assertEqual(_tool_names(events), [])
        reasoning = _joined(events, EventType.REASONING_CHUNK)
        self.assertIn("Still thinking about the change.", reasoning)
        self.assertIn(DSML_INVOKE_PREFIX, reasoning)
        self.assertIn("not_a_real_tool", reasoning)

    def test_wrapper_after_long_reasoning_still_uses_tool_start(self) -> None:
        engine = _make_engine()
        reasoning = "Let me inspect the code path. " * 80
        text = reasoning + _wrapped(
            _invoke("apply_patch", ("input", "true", "*** Begin Patch\n*** End Patch"))
        )
        events = engine.parse_complete(text)
        types = _event_types(events)
        self.assertIn(EventType.REASONING_END, types)
        self.assertEqual(_tool_names(events), ["apply_patch"])
        self.assertIn(reasoning.strip(), _joined(events, EventType.REASONING_CHUNK).strip())
        self.assertNotIn(DSML_TOOL_START, _joined(events, EventType.REASONING_CHUNK))

    def test_think_end_then_orphan_invoke_uses_content_recovery(self) -> None:
        engine = _make_engine()
        text = (
            "Done reasoning."
            + DSML_THINK_END
            + "\n"
            + _invoke("apply_patch", ("input", "true", "*** Begin Patch\n*** End Patch"))
        )
        events = engine.parse_complete(text)
        types = _event_types(events)
        self.assertIn(EventType.REASONING_END, types)
        self.assertIn(EventType.TOOL_CALL_START, types)
        self.assertEqual(_tool_names(events), ["apply_patch"])
        self.assertIn("Done reasoning.", _joined(events, EventType.REASONING_CHUNK))
        self.assertNotIn(DSML_INVOKE_PREFIX, _joined(events, EventType.REASONING_CHUNK))

    def test_tool_choice_none_keeps_reasoning_orphan_as_reasoning(self) -> None:
        engine = _make_engine(suppress=True)
        text = (
            "I would edit, but tools are disabled.\n"
            + _invoke("apply_patch", ("input", "true", "*** Begin Patch\n*** End Patch"))
        )
        events = engine.parse_complete(text)
        self.assertNotIn(EventType.TOOL_CALL_START, _event_types(events))
        self.assertEqual(_tool_names(events), [])
        reasoning = _joined(events, EventType.REASONING_CHUNK)
        self.assertIn("I would edit, but tools are disabled.", reasoning)
        self.assertIn(DSML_INVOKE_PREFIX, reasoning)
        self.assertIn("apply_patch", reasoning)

    def test_aborted_hold_does_not_emit_tool_call_and_later_wrapper_parses(self) -> None:
        engine = _make_engine()
        events: list[SemanticEvent] = []
        events.extend(engine.feed("I will edit after checking.\n", []))
        events.extend(engine.feed(DSML_INVOKE_PREFIX, []))
        self.assertFalse(any(event.type == EventType.TOOL_CALL_START for event in events))
        self.assertTrue(engine._recovery_hold_active)

        events.extend(engine.feed("zzz", []))
        self.assertFalse(engine._recovery_hold_active)
        self.assertNotIn(EventType.TOOL_CALL_START, _event_types(events))
        self.assertEqual(engine.state, ParserState.REASONING)
        self.assertIn(
            DSML_INVOKE_PREFIX + "zzz",
            _joined(events, EventType.REASONING_CHUNK),
        )

        events.extend(
            engine.feed(
                _wrapped(
                    _invoke(
                        "apply_patch",
                        ("input", "true", "*** Begin Patch\n*** End Patch"),
                    )
                ),
                [],
            )
        )
        events.extend(engine.finish())
        self.assertIn(EventType.TOOL_CALL_START, _event_types(events))
        self.assertEqual(_tool_names(events), ["apply_patch"])

    def test_name_only_candidate_rolls_back_instead_of_calling_empty_tool(self) -> None:
        engine = _make_engine()
        raw = DSML_INVOKE_PREFIX + "apply_patch" + DSML_INVOKE_NAME_END

        events = engine.parse_complete("Checking first.\n" + raw)

        self.assertNotIn(EventType.TOOL_CALL_START, _event_types(events))
        self.assertEqual(_tool_names(events), [])
        reasoning = _joined(events, EventType.REASONING_CHUNK)
        self.assertIn("Checking first.", reasoning)
        self.assertIn(raw, reasoning)

    def test_wrong_closer_rolls_back_complete_candidate(self) -> None:
        engine = _make_engine()
        raw = (
            DSML_INVOKE_PREFIX
            + "apply_patch"
            + DSML_INVOKE_NAME_END
            + _param("input", "true", "patch")
            + DSML_TOOL_END
        )

        events = engine.parse_complete(raw)

        self.assertNotIn(EventType.TOOL_CALL_START, _event_types(events))
        self.assertIn(raw, _joined(events, EventType.REASONING_CHUNK))

    def test_each_bare_invoke_is_independently_validated(self) -> None:
        engine = _make_engine()
        good = _invoke("apply_patch", ("input", "true", "patch"))
        bad = _invoke("dangerous_tool", ("input", "true", "nope"))

        events = engine.parse_complete(good + "\n" + bad)

        self.assertEqual(_tool_names(events), ["apply_patch"])
        self.assertEqual(_event_types(events).count(EventType.TOOL_CALL_START), 1)
        self.assertIn(bad, _joined(events, EventType.TEXT_CHUNK))

    def test_two_declared_bare_invokes_both_recover(self) -> None:
        engine = _make_engine(
            allowed=frozenset({"apply_patch", "read_file"})
        )
        first = _invoke("apply_patch", ("input", "true", "patch"))
        second = _invoke("read_file", ("path", "true", "README.md"))

        events = engine.parse_complete(first + "\n" + second)

        self.assertEqual(_tool_names(events), ["apply_patch", "read_file"])
        self.assertEqual(_event_types(events).count(EventType.TOOL_CALL_START), 2)

    def test_suffix_after_recovered_invoke_is_content(self) -> None:
        engine = _make_engine(thinking=False)
        invoke = _invoke("apply_patch", ("input", "true", "patch"))

        events = engine.parse_complete(invoke + "\nDone.")

        self.assertEqual(_tool_names(events), ["apply_patch"])
        self.assertIn("Done.", _joined(events, EventType.TEXT_CHUNK))

    def test_optional_outer_closer_after_recovered_invoke_is_absorbed(self) -> None:
        engine = _make_engine(thinking=False)
        invoke = _invoke("apply_patch", ("input", "true", "patch"))

        events = engine.parse_complete(invoke + DSML_TOOL_END)

        self.assertEqual(_tool_names(events), ["apply_patch"])
        self.assertNotIn(DSML_TOOL_END, _joined(events, EventType.TEXT_CHUNK))

    def test_reasoning_adapter_promotes_only_a_complete_invoke(self) -> None:
        engine = _make_engine()
        engine.skip_tool_parsing = True
        invoke = _invoke("apply_patch", ("input", "true", "patch"))

        events = engine.parse_complete("Reasoning.\n" + invoke)

        self.assertNotIn(EventType.TOOL_CALL_START, _event_types(events))
        self.assertIn(EventType.REASONING_END, _event_types(events))
        self.assertEqual(_joined(events, EventType.TEXT_CHUNK), invoke)
        self.assertIn("Reasoning.", _joined(events, EventType.REASONING_CHUNK))

    def test_reasoning_adapter_keeps_truncated_invoke_in_reasoning(self) -> None:
        engine = _make_engine()
        engine.skip_tool_parsing = True
        raw = DSML_INVOKE_PREFIX + "apply_patch" + DSML_INVOKE_NAME_END

        events = engine.parse_complete(raw)

        self.assertNotIn(EventType.TOOL_CALL_START, _event_types(events))
        self.assertEqual(_joined(events, EventType.TEXT_CHUNK), "")
        self.assertIn(raw, _joined(events, EventType.REASONING_CHUNK))

    def test_dropped_special_token_never_enters_recovery_buffer(self) -> None:
        engine = _make_engine(thinking=False)
        prefix = DSML_INVOKE_PREFIX + "apply_patch" + DSML_INVOKE_NAME_END
        events = engine.feed(prefix + _param("input", "true", "patch"), [])
        self.assertEqual(events, [])
        self.assertTrue(engine._recovery_hold_active)

        engine._has_drops = True
        self.assertEqual(engine._on_terminal(DROP_TERMINAL, "<unused-special>"), [])
        events.extend(engine.feed(DSML_INVOKE_END, []))
        events.extend(engine.finish())

        self.assertEqual(_tool_names(events), ["apply_patch"])
        self.assertNotIn("<unused-special>", _joined(events, EventType.ARG_VALUE_CHUNK))


class OverlayKeepsRecoveryProvisionalTests(unittest.TestCase):
    def test_overlay_marks_both_recovery_edges_and_commit_boundary(self) -> None:
        source = (PATCHES / "parser-deepseek-v4.dsml-orphan.py").read_text()
        tree = ast.parse(source)
        config_fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "deepseek_v4_config"
        )
        provisional: list[ast.keyword] = []
        commits: list[ast.keyword] = []

        class _Visitor(ast.NodeVisitor):
            def visit_keyword(self, node: ast.keyword) -> None:
                if node.arg == "provisional_tool_call":
                    provisional.append(node)
                if node.arg == "commit_provisional_tool_call":
                    commits.append(node)
                self.generic_visit(node)

        _Visitor().visit(config_fn)
        self.assertGreaterEqual(len(provisional), 2)
        self.assertGreaterEqual(len(commits), 1)
        for keyword in provisional + commits:
            self.assertIsInstance(keyword.value, ast.Constant)
            self.assertIs(keyword.value.value, True)


class V32RecoveryCompatibilityTests(unittest.TestCase):
    def test_v32_recovery_also_waits_for_complete_invoke(self) -> None:
        engine = StreamingParserEngine(deepseek_v32_config(), tokenizer=None)
        engine.allowed_tool_names = frozenset({"apply_patch"})
        invoke = _invoke("apply_patch", ("input", "true", "patch"))

        events = engine.parse_complete(invoke)

        self.assertEqual(_tool_names(events), ["apply_patch"])

    def test_v32_truncated_recovery_stays_content(self) -> None:
        engine = StreamingParserEngine(deepseek_v32_config(), tokenizer=None)
        engine.allowed_tool_names = frozenset({"apply_patch"})
        raw = DSML_INVOKE_PREFIX + "apply_patch" + DSML_INVOKE_NAME_END

        events = engine.parse_complete(raw)

        self.assertNotIn(EventType.TOOL_CALL_START, _event_types(events))
        self.assertEqual(_joined(events, EventType.TEXT_CHUNK), raw)


if __name__ == "__main__":
    unittest.main()
