#!/usr/bin/env python3
"""Backport DeepSeek V4 generation-prompt handling from vLLM PR #46257.

The pinned vLLM snapshot ignores ``add_generation_prompt`` and
``continue_final_message``.  In particular, an assistant-terminated history
ends at EOS instead of opening a fresh assistant turn, so generation continues
from a malformed prompt.  Patch only exact source blocks from the pinned
snapshot and fail closed if they change.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TOKENIZER_RELATIVE = Path("vllm/tokenizers/deepseek_v4.py")
ENCODING_RELATIVE = Path("vllm/tokenizers/deepseek_v4_encoding.py")

OLD_ENCODE_CONFIG = '''            encode_config = dict(
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )
'''

NEW_ENCODE_CONFIG = '''            encode_config = dict(
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
                add_generation_prompt=kwargs.get("add_generation_prompt", True),
                continue_final_message=kwargs.get("continue_final_message", False),
            )
'''

OLD_RENDER_SIGNATURE = '''def render_message(index: int, messages: List[Dict[str, Any]], thinking_mode: str, drop_thinking: bool = True, reasoning_effort: Optional[str] = None) -> str:
'''

NEW_RENDER_SIGNATURE = '''def render_message(index: int, messages: List[Dict[str, Any]], thinking_mode: str, drop_thinking: bool = True, reasoning_effort: Optional[str] = None, add_generation_prompt: bool = True, continue_final_message: bool = False) -> str:
'''

OLD_WO_EOS = '''    wo_eos = msg.get("wo_eos", False)
'''

NEW_WO_EOS = '''    wo_eos = msg.get("wo_eos", False) or (
        continue_final_message and index == len(messages) - 1
    )
'''

OLD_TRANSITION = '''    elif messages[index].get("role") in ["user", "developer"]:
        # Normal generation: append Assistant + thinking token
        prompt += ASSISTANT_SP_TOKEN
        if not drop_thinking and thinking_mode == "thinking":
            prompt += thinking_start_token
        elif drop_thinking and thinking_mode == "thinking" and index >= last_user_idx:
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token
'''

NEW_TRANSITION = '''    elif messages[index].get("role") in ["user", "developer"]:
        # This Assistant token is both the turn separator (when an assistant
        # message follows) and the generation prompt (when this is the last
        # message). Only the latter is suppressed by add_generation_prompt=False.
        if index + 1 != len(messages) or add_generation_prompt:
            prompt += ASSISTANT_SP_TOKEN
            if not drop_thinking and thinking_mode == "thinking":
                prompt += thinking_start_token
            elif drop_thinking and thinking_mode == "thinking" and index >= last_user_idx:
                prompt += thinking_start_token
            else:
                prompt += thinking_end_token

    elif (
        add_generation_prompt
        and index + 1 == len(messages)
        and messages[index].get("role") == "assistant"
    ):
        # add_generation_prompt after an assistant turn -> open a fresh turn.
        prompt += ASSISTANT_SP_TOKEN
        prompt += thinking_start_token if thinking_mode == "thinking" else thinking_end_token
'''

OLD_ENCODE_SIGNATURE = '''def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
'''

NEW_ENCODE_SIGNATURE = '''def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[str] = None,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> str:
'''

OLD_CONTEXT_INITIALIZATION = '''    context = context if context else []
'''

NEW_CONTEXT_INITIALIZATION = '''    if continue_final_message:
        add_generation_prompt = False
    if messages:
        _last_role = messages[-1].get("role")
        if continue_final_message and _last_role != "assistant":
            raise ValueError(
                "Cannot set `continue_final_message`=True when the last message is "
                "not from the assistant."
            )

    context = context if context else []
'''

OLD_RENDER_CALL = '''            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
        )
'''

NEW_RENDER_CALL = '''            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
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


def apply_generation_prompt_patch(package_root: Path) -> None:
    tokenizer_path = package_root / TOKENIZER_RELATIVE
    encoding_path = package_root / ENCODING_RELATIVE

    tokenizer_source = replace_once(
        tokenizer_path.read_text(),
        OLD_ENCODE_CONFIG,
        NEW_ENCODE_CONFIG,
        tokenizer_path,
    )
    encoding_source = encoding_path.read_text()
    for old, new in (
        (OLD_RENDER_SIGNATURE, NEW_RENDER_SIGNATURE),
        (OLD_WO_EOS, NEW_WO_EOS),
        (OLD_TRANSITION, NEW_TRANSITION),
        (OLD_ENCODE_SIGNATURE, NEW_ENCODE_SIGNATURE),
        (OLD_CONTEXT_INITIALIZATION, NEW_CONTEXT_INITIALIZATION),
        (OLD_RENDER_CALL, NEW_RENDER_CALL),
    ):
        encoding_source = replace_once(encoding_source, old, new, encoding_path)

    # Validate every source block before changing either file.
    tokenizer_path.write_text(tokenizer_source)
    encoding_path.write_text(encoding_source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages"),
    )
    args = parser.parse_args()
    apply_generation_prompt_patch(args.package_root)


if __name__ == "__main__":
    main()
