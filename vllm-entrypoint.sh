#!/bin/sh
set -eu

# A dead EngineCore cannot unlink its CPU-KV mmap. Every production replica has
# a private /dev/shm, so cleanup cannot remove another replica's live CPU cache.
if [ "${VLLM_CLEAN_STALE_CPU_KV:-1}" = "1" ]; then
    find /dev/shm -maxdepth 1 -type f -name 'vllm_offload_*.mmap' -delete
fi

python3 /opt/apply-deepseek-v4-reasoning-effort.py
python3 /opt/apply-deepseek-v4-generation-prompt.py
python3 /opt/apply-deepseek-v4-indexer-prefill-budget.py
python3 /opt/apply-deepseek-v4-parser-recovery.py

exec vllm serve "$@"
