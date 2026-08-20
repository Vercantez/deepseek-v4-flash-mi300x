# Patch provenance

Most `*.py` files in this directory are **byte-for-byte the overlays that run in
production** (see `../SHA256SUMS`). They are mounted read-only over files inside
the pinned vLLM ROCm container image by `../compose.yaml`.

The `apply-deepseek-v4-*.py` source transformers are the exceptions: the
entrypoint runs them against exact, version-checked source blocks. They fail
closed if the pinned vLLM source changes, avoiding large tokenizer overlays for
small compatibility backports.

The `diffs/*.patch` files in this directory are informational unified diffs showing
exactly what each overlay changes relative to an upstream base revision. They were
generated with `diff -u` on 2026-08-04 against:

| Overlay | Upstream base |
| --- | --- |
| `gpt_oss_triton_kernels_moe.pack128-fused-silu-fast-routing.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` |
| `mxfp4.fused-silu.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` |
| `triton-kernels-matmul-ogs-opt-flags.dsv4-mi300x.py` | `ROCm/triton` @ `0f380657dbf3ee86eb57558ff71df24f03b5d4e7` — `python/triton_kernels/triton_kernels/matmul_ogs_details/opt_flags.py` (the revision vLLM's ROCm builds vendor) |
| `fused_compress_quant_cache.fnuz-shuffle.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` |
| `aiter_pa_mqa_logits.i64.py` | `ROCm/aiter` `main` @ `4db400a90c1c1c558f3dbb40b0e6728825bbcc2b` — `aiter/ops/triton/gluon/pa_mqa_logits.py` |
| `rocm_aiter_mla_sparse.prefill-bh64.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` |
| `rocm_aiter_mla.dspark-causal.py` | `vllm-project/vllm` @ `77469c9057bec3212a64877dbbf3b9c48c22d786` — `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`. This file is **identical to the upstream file at that commit**; the diff shows the change the commit itself made. |
| `dspark-speculator.independent-draft-gumbel.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` |
| `spec-decode-utils.independent-draft-gumbel.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/v1/worker/gpu/spec_decode/utils.py` |
| `kv_offload_cpu_gpu_worker.load-war.py` | `vllm-project/vllm` `main` @ `cb8104839c141609d99f1254459ef3a4f1bd4263` — `vllm/v1/kv_offload/cpu/gpu_worker.py` (post-#46278 state; PR #47291 is not merged upstream) |
| `tiering-fs-bounded-lru.py` | New companion module for the filesystem manager overlay; implements compact shard-level lookup leases, cancellation fences, crash-safe cross-process eviction/write coordination, and background atomic LRU eviction without modifying upstream package initialization |
| `tiering-fs-manager.disk-reserve.py` | `vllm-project/vllm` @ `124154a8843d1f8e4d4e2d5d466e2d3ebc3716da` — `vllm/v1/kv_offload/tiering/fs/manager.py` |
| `async_lookup.bounded.py` | `vllm-project/vllm` @ `124154a8843d1f8e4d4e2d5d466e2d3ebc3716da` — `vllm/v1/kv_offload/tiering/async_lookup.py`; parallelizes request probes in small fair chunks and makes cancellation or overload a cache miss instead of an unbounded queue |
| `kv_lookup_fail_open.py` | New dependency-free scheduler policy; enforces the external-cache deadline and circuit breaker and is fault-tested without a GPU |
| `apply-deepseek-v4-reasoning-effort.py` | `vllm-project/vllm` @ `124154a8843d1f8e4d4e2d5d466e2d3ebc3716da`; backports the `low`/`high`/`max` prompts published in DeepSeek V4 Flash 0731 model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| `apply-deepseek-v4-generation-prompt.py` | `vllm-project/vllm` @ `124154a8843d1f8e4d4e2d5d466e2d3ebc3716da`; backports [PR #46257](https://github.com/vllm-project/vllm/pull/46257) head `8cf0094`, honoring `add_generation_prompt` and `continue_final_message` so assistant-terminated histories do not end at EOS |
| `apply-deepseek-v4-indexer-prefill-budget.py` | `vllm-project/vllm` @ `124154a8843d1f8e4d4e2d5d466e2d3ebc3716da`; backports PR #51252 head `1136a8f3d86f708fb71bed77a1c8c7b59a270fbb`, sizing the sparse-indexer prefill budget by `compress_ratio` |
| `apply-deepseek-v4-parser-recovery.py` | `vllm-project/vllm` @ `124154a8843d1f8e4d4e2d5d466e2d3ebc3716da`; synchronizes request tools into the engine-based reasoning adapter before extraction, adapting the reviewed approach in PR #52645 head `4f2aae22cd043e7c9384f578a9e546ec848abe73` |
| `parser-deepseek-v32.dsml-orphan.py`, `parser-deepseek-v4.dsml-orphan.py`, `parser-engine.dsml-orphan.py`, `parser-engine-config.dsml-orphan.py`, `streaming-parser-engine.dsml-orphan.py`, `tool-parser-utils.dsml-orphan.py` | `vllm-project/vllm` @ `124154a8843d1f8e4d4e2d5d466e2d3ebc3716da`; backport of PR #49117 head `7ef0ae2480799e95fb7cb801a8105c1db2585164`, hardened so a recovery remains provisional until the full invoke closes, based on PR #52645 head `4f2aae22cd043e7c9384f578a9e546ec848abe73` |
| `parser-engine.dsml-orphan.py` reasoning-token follow-up | Incremental adaptation of [PR #49743](https://github.com/vllm-project/vllm/pull/49743) for the streaming parser engine; counts the leading reasoning span when DeepSeek V4's prompt supplies the opening marker and generated tokens contain only the closing marker |
| `parser-deepseek-v4.dsml-orphan.py` reasoning-orphan follow-up | Incremental close of [PR #49117](https://github.com/vllm-project/vllm/pull/49117) limitation 2: recover a declared DSML invoke that starts in prompt-opened `REASONING` when the model omits both `</think>` and the `tool_calls` wrapper. Recovery is now fail-closed through the full invoke and request-aware in the reasoning adapter. |
| `structural-tag-registry.deepseek-v4-auto.py` | `vllm-project/vllm` @ `124154a8843d1f8e4d4e2d5d466e2d3ebc3716da`; adaptation of PR #46632 commit `857187ab10a951270ce1192ead64a14afd4ce41b` |

## Regenerating a diff

```bash
curl -L -o base.py \
  https://raw.githubusercontent.com/vllm-project/vllm/<sha>/vllm/...
diff -u --label "a/<upstream path>" --label "b/<overlay>" base.py <overlay>.py
```

> The overlays are the source of truth. The diffs are documentation: the pinned
> image that ran in production is a vLLM ROCm nightly
> (`0.26.1rc1.dev229+g124154a88.rocm723`), which may differ slightly from any
> single upstream revision.

`diffs/14-kv-cache-fail-open.patch` is intentionally incremental against the
previous production overlays rather than a pristine upstream file. It groups
the scheduler deadline, bounded async lookup, failed-load invalidation, and
filesystem-cache metrics as one availability fix.

`diffs/15-parser-engine.prompt-opened-reasoning-count.patch` is incremental
against the DSML recovery parser overlay. It mirrors the implicit reasoning
span behavior from vLLM PR #49743 in the parser-engine counter used by DeepSeek
V4 Responses streams.

`diffs/17-parser-deepseek-v4.reasoning-orphan.patch` is incremental against the
DSML recovery V4 parser overlay. It adds `(ParserState.REASONING,
"INVOKE_PREFIX")` so a thinking request that never leaves reasoning can still
begin recovery of a declared `apply_patch` (or any other declared tool).

`diffs/18-parser-dsml-provisional-recovery.patch` supersedes the eager behavior
in diff 17. It keeps all semantic events provisional until `</invoke>`, rolls
truncated or malformed candidates back as text, validates each later bare
invoke independently, preserves suffix content, and synchronizes request tools
through the separately configured reasoning adapter. The design follows the
reviewed hardening in vLLM PR #52645 while remaining compatible with the pinned
`124154a88` runtime.

The source transformer is checked against repository-owned snapshots of the
exact pinned modules by the normal unit suite. Before deploying parser changes,
also run the serving-layer smoke in the pinned image (it does not load model
weights or require a GPU):

```bash
tests/run_exact_image_parser_recovery.sh
```

One accounting limitation remains: when recovery closes implicit reasoning at
a bare invoke without a generated `</think>`, total and output token counts are
correct but the optional `reasoning_tokens` detail can under-report that span.
Fixing that safely requires semantic token-position plumbing rather than
guessing from the first invoke-like token sequence.

`diffs/16-aiter-pa-mqa-block256-i64.patch` is incremental against the previous
AITER overlay. It converts the alternate non-variable-context preshuffle
pipeline selected by production's `ChunkK=256`, `KVBlockSize=256` combination;
the earlier overlay covered only the other branch.

## Licensing

Overlays derived from vLLM carry vLLM's Apache-2.0 headers. The AITER-derived
overlay (`aiter_pa_mqa_logits.i64.py`) carries AITER's MIT header. See
`../LICENSE` and the individual file headers.
