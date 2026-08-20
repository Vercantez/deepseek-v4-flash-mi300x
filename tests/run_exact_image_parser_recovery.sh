#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
image='vllm/vllm-openai-rocm@sha256:e68d18b2ba50298661bfc49baf01158fbf036645c2362cccf3e8a7a79fe6c69a'
site_packages='/usr/local/lib/python3.12/dist-packages'

docker run --rm --security-opt label=disable --entrypoint /bin/sh \
    -v "$repo_root/patches/apply-deepseek-v4-parser-recovery.py:/opt/apply-deepseek-v4-parser-recovery.py:ro" \
    -v "$repo_root/patches/parser-deepseek-v32.dsml-orphan.py:$site_packages/vllm/parser/deepseek_v32.py:ro" \
    -v "$repo_root/patches/parser-deepseek-v4.dsml-orphan.py:$site_packages/vllm/parser/deepseek_v4.py:ro" \
    -v "$repo_root/patches/parser-engine.dsml-orphan.py:$site_packages/vllm/parser/engine/parser_engine.py:ro" \
    -v "$repo_root/patches/parser-engine-config.dsml-orphan.py:$site_packages/vllm/parser/engine/parser_engine_config.py:ro" \
    -v "$repo_root/patches/streaming-parser-engine.dsml-orphan.py:$site_packages/vllm/parser/engine/streaming_parser_engine.py:ro" \
    -v "$repo_root/patches/tool-parser-utils.dsml-orphan.py:$site_packages/vllm/tool_parsers/utils.py:ro" \
    -v "$repo_root/tests/exact_image_parser_recovery_smoke.py:/opt/exact_image_parser_recovery_smoke.py:ro" \
    "$image" \
    -c 'python3 /opt/apply-deepseek-v4-parser-recovery.py && python3 /opt/exact_image_parser_recovery_smoke.py'
