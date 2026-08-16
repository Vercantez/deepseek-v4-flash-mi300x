#!/usr/bin/env python3
"""Backport vLLM PR #51252 for DeepSeek V4 prefix-hit correctness.

The pinned vLLM snapshot budgets the sparse-indexer prefill workspace in
uncompressed rows even though DeepSeek V4's consumer allocates compressed
rows.  Under chunked prefill or prefix-cache hits this can admit more rows than
the gather buffer holds, leaving tail rows uninitialized and corrupting logits.

Patch only the two exact blocks from vLLM 124154a88.  The transformer is
idempotent and fails closed if the packaged source does not match that pin.
"""

from __future__ import annotations

import argparse
from pathlib import Path


INDEXER_RELATIVE = Path("vllm/v1/attention/backends/mla/indexer.py")

OLD_BUDGET = '''        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
'''

NEW_BUDGET = '''        # Resolved before the prefill budget below because the chunker feeds it
        # compressed seq_lens and must be sized in the same units.
        self.compress_ratio = 1
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = (
            get_max_prefill_buffer_size(self.vllm_config) // self.compress_ratio
        )
'''

OLD_COMPRESSION = '''        # KV compression. Default to 1 for no compression.
        self.compress_ratio = 1
        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio
'''

NEW_COMPRESSION = '''        # compress_ratio is resolved earlier (used to size the prefill budget).
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


def apply_indexer_prefill_budget_patch(package_root: Path) -> None:
    indexer_path = package_root / INDEXER_RELATIVE
    replace_once(indexer_path, OLD_BUDGET, NEW_BUDGET)
    replace_once(indexer_path, OLD_COMPRESSION, NEW_COMPRESSION)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages"),
    )
    args = parser.parse_args()
    apply_indexer_prefill_budget_patch(args.package_root)


if __name__ == "__main__":
    main()
