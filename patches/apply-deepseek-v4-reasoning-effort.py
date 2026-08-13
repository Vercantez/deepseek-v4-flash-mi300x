#!/usr/bin/env python3
"""Backport DeepSeek V4 Flash 0731's three native reasoning levels.

The pinned vLLM snapshot predates the model revision that added distinct
``low``, ``high``, and ``max`` prompt prefixes.  Patch only the two small,
version-checked source blocks involved in normalizing and rendering the effort.
The transformer fails closed if the pinned source changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TOKENIZER_RELATIVE = Path("vllm/tokenizers/deepseek_v4.py")
ENCODING_RELATIVE = Path("vllm/tokenizers/deepseek_v4_encoding.py")

OLD_NORMALIZATION = '''            reasoning_effort = kwargs.get("reasoning_effort")
            if not isinstance(reasoning_effort, str):
                reasoning_effort = None
            elif reasoning_effort == "none":
                thinking_mode = "chat"
                reasoning_effort = None
            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"
'''

NEW_NORMALIZATION = '''            reasoning_effort = kwargs.get("reasoning_effort")
            if not isinstance(reasoning_effort, str):
                reasoning_effort = None
            else:
                effort_aliases = {
                    "none": None,
                    "minimal": "low",
                    "low": "low",
                    "medium": "high",
                    "high": "high",
                    "xhigh": "max",
                    "max": "max",
                }
                if reasoning_effort not in effort_aliases:
                    raise ValueError(
                        f"Unsupported DeepSeek V4 reasoning effort: {reasoning_effort}"
                    )
                reasoning_effort = effort_aliases[reasoning_effort]
                if reasoning_effort is None:
                    thinking_mode = "chat"
'''

OLD_PROMPTS = '''REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\\n\\n"
)
'''

NEW_PROMPTS = '''REASONING_EFFORT_PROMPTS: Dict[str, str] = {
    "low": "",
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\\n"
        "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\\n"
        "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\\n\\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\\n"
        "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\\n\\n"
    ),
}
DEFAULT_REASONING_EFFORT = "low"
'''

OLD_RENDERING = '''    # Reasoning effort prefix (only at index 0 in thinking mode with max effort)
    assert reasoning_effort in ['max', None, 'high'], f"Invalid reasoning effort: {reasoning_effort}"
    if index == 0 and thinking_mode == "thinking" and reasoning_effort == 'max':
        prompt += REASONING_EFFORT_MAX
'''

NEW_RENDERING = '''    # Reasoning effort prefix (only at index 0 in thinking mode; low adds nothing)
    reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
    assert reasoning_effort in REASONING_EFFORT_PROMPTS, (
        f"Invalid reasoning effort: {reasoning_effort}, expected one of "
        f"{list(REASONING_EFFORT_PROMPTS)}"
    )
    if index == 0 and thinking_mode == "thinking":
        prompt += REASONING_EFFORT_PROMPTS[reasoning_effort]
'''


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one pinned source block in {path}, found {count}; "
            "refusing to patch an unknown vLLM version"
        )
    path.write_text(source.replace(old, new, 1))


def apply_reasoning_effort_patch(package_root: Path) -> None:
    replace_once(package_root / TOKENIZER_RELATIVE, OLD_NORMALIZATION, NEW_NORMALIZATION)
    encoding_path = package_root / ENCODING_RELATIVE
    replace_once(encoding_path, OLD_PROMPTS, NEW_PROMPTS)
    replace_once(encoding_path, OLD_RENDERING, NEW_RENDERING)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages"),
    )
    args = parser.parse_args()
    apply_reasoning_effort_patch(args.package_root)


if __name__ == "__main__":
    main()
