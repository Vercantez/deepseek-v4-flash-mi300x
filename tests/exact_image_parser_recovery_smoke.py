#!/usr/bin/env python3
"""Run inside the pinned vLLM image with parser overlays and adapter patch."""

from types import SimpleNamespace

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.deepseek_v4 import (
    DSML_INVOKE_END,
    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
)
from vllm.parser.engine.registered_adapters import (
    DeepSeekV4ParserReasoningAdapter,
    DeepSeekV4ParserToolAdapter,
)


class DeepSeekV4DelegatingParser(DelegatingParser):
    reasoning_parser_cls = DeepSeekV4ParserReasoningAdapter
    tool_parser_cls = DeepSeekV4ParserToolAdapter


class FakeTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}


tool = ChatCompletionToolsParam(
    function={
        "name": "apply_patch",
        "description": "Apply a patch",
        "parameters": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
        },
    }
)
request = SimpleNamespace(
    tools=[tool],
    tool_choice="auto",
    skip_special_tokens=True,
    include_reasoning=True,
)
parameter = (
    '<｜DSML｜parameter name="input" string="true">patch'
    "</｜DSML｜parameter>"
)
invoke = (
    DSML_INVOKE_PREFIX
    + "apply_patch"
    + DSML_INVOKE_NAME_END
    + parameter
    + DSML_INVOKE_END
)

# The reasoning adapter must know the current request tools and promote only a
# structurally complete invoke as content for the subsequent tool-parser pass.
reasoning_parser = DeepSeekV4ParserReasoningAdapter(
    FakeTokenizer(), chat_template_kwargs={"thinking": True}
)
reasoning, content = reasoning_parser.extract_reasoning(
    "Inspecting the edit.\n" + invoke,
    request,
)
assert reasoning == "Inspecting the edit."
assert content == invoke
assert reasoning_parser._parser_engine._engine.allowed_tool_names == frozenset(
    {"apply_patch"}
)

tool_parser = DeepSeekV4ParserToolAdapter(FakeTokenizer(), [tool])
parsed = tool_parser.extract_tool_calls(content, request)
assert parsed.tools_called
assert len(parsed.tool_calls) == 1
assert parsed.tool_calls[0].function.name == "apply_patch"
assert parsed.tool_calls[0].function.arguments == '{"input": "patch"}'

# Exercise the serving-layer streaming path that originally failed: the
# DelegatingParser must refresh request tools in its separately configured
# reasoning adapter, transition on the recovered invoke, and hand that same
# delta to the tool adapter.
delegating_parser = DeepSeekV4DelegatingParser(
    FakeTokenizer(),
    [tool],
    chat_template_kwargs={"thinking": True},
)
reasoning_delta = delegating_parser.parse_delta(
    "Delegating reasoning.\n",
    [],
    request,
    prompt_token_ids=[],
    finished=False,
)
assert reasoning_delta is not None
assert reasoning_delta.reasoning == "Delegating reasoning."
invoke_chunks = [
    invoke[: len(DSML_INVOKE_PREFIX) - 1],
    invoke[len(DSML_INVOKE_PREFIX) - 1 : -len(DSML_INVOKE_END)],
    invoke[-len(DSML_INVOKE_END) :],
]
for partial_invoke in invoke_chunks[:-1]:
    partial_delta = delegating_parser.parse_delta(
        partial_invoke,
        [],
        request,
        finished=False,
    )
    assert partial_delta is None or not partial_delta.tool_calls
tool_delta = delegating_parser.parse_delta(
    invoke_chunks[-1], [], request, finished=True
)
assert tool_delta is not None
assert not tool_delta.content
assert tool_delta.tool_calls
assert len(tool_delta.tool_calls) == 1
assert tool_delta.tool_calls[0].function.name == "apply_patch"

# Non-streaming aggregation must not lose content emitted after a recovered
# call, including the raw form of a later candidate that fails validation.
suffix_parser = DeepSeekV4ParserToolAdapter(FakeTokenizer(), [tool])
parsed = suffix_parser.extract_tool_calls(
    "VISIBLE PREFIX " + invoke + " VISIBLE SUFFIX", request
)
assert parsed.tools_called
assert parsed.content == "VISIBLE PREFIX  VISIBLE SUFFIX"

rejected = invoke.replace("apply_patch", "not_declared", 1)
rejected_parser = DeepSeekV4ParserToolAdapter(FakeTokenizer(), [tool])
parsed = rejected_parser.extract_tool_calls(invoke + rejected, request)
assert parsed.tools_called
assert len(parsed.tool_calls) == 1
assert parsed.content == rejected

# Request-scoped suppression must not leak when a parser instance is reused.
reused_parser = DeepSeekV4ParserReasoningAdapter(
    FakeTokenizer(), chat_template_kwargs={"thinking": True}
)
none_request = SimpleNamespace(**vars(request))
none_request.tool_choice = "none"
reasoning, content = reused_parser.extract_reasoning(invoke, none_request)
assert reasoning == invoke
assert content is None
reasoning, content = reused_parser.extract_reasoning(invoke, request)
assert reasoning is None
assert content == invoke
no_tools_request = SimpleNamespace(**vars(request))
no_tools_request.tools = []
reasoning, content = reused_parser.extract_reasoning(invoke, no_tools_request)
assert reasoning == invoke
assert content is None
assert reused_parser._parser_engine._engine.allowed_tool_names is None

# Closing the name alone must remain reasoning text and must never become an
# empty-argument tool execution.
truncated_parser = DeepSeekV4ParserReasoningAdapter(
    FakeTokenizer(), chat_template_kwargs={"thinking": True}
)
truncated = DSML_INVOKE_PREFIX + "apply_patch" + DSML_INVOKE_NAME_END
reasoning, content = truncated_parser.extract_reasoning(truncated, request)
assert reasoning == truncated
assert content is None

print("exact-image DeepSeek V4 parser recovery smoke: PASS")
