#!/bin/sh
set -eu

# A dead EngineCore cannot unlink its CPU-KV mmap. This host serves one vLLM instance.
find /dev/shm -maxdepth 1 -type f -name 'vllm_offload_*.mmap' -delete

python3 /opt/apply-deepseek-v4-reasoning-effort.py

if [ -n "${VLLM_PROFILE_DIR:-}" ]; then
    mkdir -p "$VLLM_PROFILE_DIR"
    set -- "$@" \
        --profiler-config.profiler=torch \
        --profiler-config.torch-profiler-dir="$VLLM_PROFILE_DIR" \
        --profiler-config.torch-profiler-with-stack=false \
        --profiler-config.torch-profiler-record-shapes=true
fi

exec vllm serve "$@"
