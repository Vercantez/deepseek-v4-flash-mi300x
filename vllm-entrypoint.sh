#!/bin/sh
set -eu

# A dead EngineCore cannot unlink its CPU-KV mmap. Cleanup is safe on the
# single-replica profile; TP1x2 disables it because /dev/shm is shared and one
# replica must never unlink the other's live cache during a rolling restart.
if [ "${VLLM_CLEAN_STALE_CPU_KV:-1}" = "1" ]; then
    find /dev/shm -maxdepth 1 -type f -name 'vllm_offload_*.mmap' -delete
fi

python3 /opt/apply-deepseek-v4-reasoning-effort.py
python3 /opt/apply-deepseek-v4-indexer-prefill-budget.py

exec vllm serve "$@"
