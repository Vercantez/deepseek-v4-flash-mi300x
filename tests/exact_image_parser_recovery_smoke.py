#!/usr/bin/env python3
"""Run inside the pinned vLLM image with parser overlays and adapter patch."""

from types import SimpleNamespace

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.deepseek_v4 import (
    DSML_INVOKE_END,
    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
)
from vllm.parser.engine.registered_adapters import (
    DeepSeekV4ParserReasoningAdapter,
    DeepSeekV4ParserToolAdapter,
)


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
