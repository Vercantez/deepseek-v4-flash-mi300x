#!/bin/sh
set -eu

CACHE_ROOT=${VLLM_KV_CACHE_ROOT:-/var/lib/vllm-kv}
TRIGGER_FREE_GIB=${VLLM_KV_TRIGGER_FREE_GIB:-2560}
TARGET_FREE_GIB=${VLLM_KV_TARGET_FREE_GIB:-3072}
BATCH_SHARDS=${VLLM_KV_PRUNE_BATCH_SHARDS:-32}

if [ "$CACHE_ROOT" != "/var/lib/vllm-kv" ]; then
  echo "Refusing unexpected KV cache root: $CACHE_ROOT" >&2
  exit 1
fi
if [ ! -d "$CACHE_ROOT" ]; then
  exit 0
fi

exec 9>/run/vllm-kv-cache-prune.lock
flock -n 9 || exit 0

gib=$((1024 * 1024 * 1024))
trigger_bytes=$((TRIGGER_FREE_GIB * gib))
target_bytes=$((TARGET_FREE_GIB * gib))
available_bytes=$(df -B1 --output=avail "$CACHE_ROOT" | tail -1 | tr -d ' ')
if [ "$available_bytes" -ge "$trigger_bytes" ]; then
  exit 0
fi

logger -t vllm-kv-prune "KV filesystem has $((available_bytes / gib)) GiB free; pruning hash shards toward ${TARGET_FREE_GIB} GiB"

deleted=0
find "$CACHE_ROOT" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' \
  | sort -n \
  | cut -d' ' -f2- \
  | while IFS= read -r shard; do
      case "$shard" in
        "$CACHE_ROOT"/*) ;;
        *) echo "Refusing shard outside cache root: $shard" >&2; exit 1 ;;
      esac
      rm -rf -- "$shard"
      deleted=$((deleted + 1))
      if [ $((deleted % BATCH_SHARDS)) -eq 0 ]; then
        available_bytes=$(df -B1 --output=avail "$CACHE_ROOT" | tail -1 | tr -d ' ')
        if [ "$available_bytes" -ge "$target_bytes" ]; then
          logger -t vllm-kv-prune "Pruned $deleted shards; KV filesystem now has $((available_bytes / gib)) GiB free"
          exit 0
        fi
      fi
    done
