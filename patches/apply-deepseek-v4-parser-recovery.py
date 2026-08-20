#!/usr/bin/env python3
"""Synchronize request context into vLLM's engine-based reasoning adapter.

The parser overlays recover a complete bare DSML invoke while the serving
stack is still in its reasoning phase. The pinned vLLM adapters otherwise
construct that reasoning parser without request tools and do not refresh its
request state before every streaming delta. Patch only exact source blocks
from vLLM 124154a88 and fail closed if the image changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ABSTRACT_PARSER_RELATIVE = Path("vllm/parser/abstract_parser.py")
ADAPTERS_RELATIVE = Path("vllm/parser/engine/adapters.py")

OLD_STREAMING_REASONING_ENTRY = '''        # Reasoning extraction
        if self._in_reasoning_phase(state):
            delta_message = self.extract_reasoning_streaming(
'''

NEW_STREAMING_REASONING_ENTRY = '''        # Reasoning extraction
        if self._in_reasoning_phase(state):
            reasoning_parser = self._reasoning_parser
            if reasoning_parser is not None and reasoning_parser.engine_based_streaming:
                reasoning_parser.adjust_request(request)
            delta_message = self.extract_reasoning_streaming(
'''

OLD_DUPLICATE_REASONING_PARSER = '''            )
            reasoning_parser = self._reasoning_parser
            if reasoning_parser is not None and reasoning_parser.engine_based_streaming:
'''

NEW_DUPLICATE_REASONING_PARSER = '''            )
            if reasoning_parser is not None and reasoning_parser.engine_based_streaming:
'''

OLD_NONSTREAMING_REASONING_ENTRY = '''    ) -> tuple[str | None, str | None]:
        with self._skip_tool_parsing():
            return self._parser_engine.extract_reasoning(model_output, request)
'''

NEW_NONSTREAMING_REASONING_ENTRY = '''    ) -> tuple[str | None, str | None]:
        self.adjust_request(request)
        with self._skip_tool_parsing():
            return self._parser_engine.extract_reasoning(model_output, request)
'''

OLD_REASONING_ADJUST_REQUEST = '''    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        return self._parser_engine.adjust_request(request)

    def has_engine_confirmed_reasoning_end(self) -> bool:
'''

NEW_REASONING_ADJUST_REQUEST = '''    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = self._parser_engine.adjust_request(request)
        with self._skip_tool_parsing():
            self._parser_engine._check_skip_tool_parsing(request)
        return request

    def has_engine_confirmed_reasoning_end(self) -> bool:
'''


def replace_once(source: str, old: str, new: str, path: Path) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one pinned source block in {path}, found {count}; "
            "refusing to patch an unknown vLLM version"
        )
    return source.replace(old, new, 1)


def apply_parser_recovery_patch(package_root: Path) -> None:
    abstract_path = package_root / ABSTRACT_PARSER_RELATIVE
    adapters_path = package_root / ADAPTERS_RELATIVE

    abstract_source = abstract_path.read_text()
    for old, new in (
        (OLD_STREAMING_REASONING_ENTRY, NEW_STREAMING_REASONING_ENTRY),
        (OLD_DUPLICATE_REASONING_PARSER, NEW_DUPLICATE_REASONING_PARSER),
    ):
        abstract_source = replace_once(abstract_source, old, new, abstract_path)
    adapters_source = adapters_path.read_text()
    for old, new in (
        (OLD_NONSTREAMING_REASONING_ENTRY, NEW_NONSTREAMING_REASONING_ENTRY),
        (OLD_REASONING_ADJUST_REQUEST, NEW_REASONING_ADJUST_REQUEST),
    ):
        adapters_source = replace_once(adapters_source, old, new, adapters_path)

    # Validate all changes before touching either installed module.
    compile(abstract_source, str(abstract_path), "exec")
    compile(adapters_source, str(adapters_path), "exec")
    abstract_path.write_text(abstract_source)
    adapters_path.write_text(adapters_source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages"),
    )
    args = parser.parse_args()
    apply_parser_recovery_patch(args.package_root)


if __name__ == "__main__":
    main()
