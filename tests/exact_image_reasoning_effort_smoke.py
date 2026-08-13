#!/usr/bin/env python3
"""Run inside the pinned vLLM image after applying the effort patch."""

from vllm.tokenizers.deepseek_v4_encoding import render_message


messages = [{"role": "system", "content": "system"}]
low = render_message(0, messages, "thinking", reasoning_effort="low")
high = render_message(0, messages, "thinking", reasoning_effort="high")
maximum = render_message(0, messages, "thinking", reasoning_effort="max")

assert "Reasoning Effort:" not in low
assert high.startswith("Reasoning Effort: Absolute maximum")
assert maximum.startswith("Reasoning Effort: Beyond maximum")
assert len({low, high, maximum}) == 3

tokenizer_source = open(
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4.py"
).read()
for alias in ('"minimal": "low"', '"medium": "high"', '"xhigh": "max"'):
    assert alias in tokenizer_source

print("exact-image DeepSeek V4 reasoning effort smoke: PASS")
