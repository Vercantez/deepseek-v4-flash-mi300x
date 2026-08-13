#!/bin/sh
set -eu

# A dead EngineCore cannot unlink its CPU-KV mmap. This host serves one vLLM instance.
find /dev/shm -maxdepth 1 -type f -name 'vllm_offload_*.mmap' -delete

python3 /opt/apply-deepseek-v4-reasoning-effort.py

exec vllm serve "$@"
