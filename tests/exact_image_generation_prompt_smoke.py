#!/usr/bin/env python3
"""Run inside the pinned vLLM image after applying tokenizer patches."""

from vllm.tokenizers.deepseek_v4_encoding import encode_messages


def encode(messages, **kwargs):
    return encode_messages(messages, thinking_mode="chat", **kwargs)


user_only = [{"role": "user", "content": "Hello"}]
assert encode(user_only).endswith("<｜Assistant｜></think>")
assert "<｜Assistant｜>" not in encode(user_only, add_generation_prompt=False)

assistant_last = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
]
fresh_turn = encode(assistant_last, add_generation_prompt=True)
assert fresh_turn.endswith("Hi<｜end▁of▁sentence｜><｜Assistant｜></think>")

continued_turn = encode(assistant_last, continue_final_message=True)
assert continued_turn.endswith("Hi")
assert not continued_turn.endswith("<｜end▁of▁sentence｜>")
assert "<｜Assistant｜>" in continued_turn

try:
    encode(user_only, continue_final_message=True)
except ValueError:
    pass
else:
    raise AssertionError("continue_final_message must require an assistant last")

print("exact-image DeepSeek V4 generation-prompt smoke: PASS")
