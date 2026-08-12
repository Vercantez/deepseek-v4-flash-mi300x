# Patch provenance

The `*.py` files in this directory are **byte-for-byte the overlays that run in
production** (see `../SHA256SUMS`). They are mounted read-only over files inside
the pinned vLLM ROCm container image by `../compose.yaml`.

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
| `parser-deepseek-v32.dsml-orphan.py`, `parser-deepseek-v4.dsml-orphan.py`, `parser-engine.dsml-orphan.py`, `parser-engine-config.dsml-orphan.py`, `streaming-parser-engine.dsml-orphan.py`, `tool-parser-utils.dsml-orphan.py` | `vllm-project/vllm` @ `124154a8843d1f8e4d4e2d5d466e2d3ebc3716da`; backport of PR #49117 head `7ef0ae2480799e95fb7cb801a8105c1db2585164` |
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

## Licensing

Overlays derived from vLLM carry vLLM's Apache-2.0 headers. The AITER-derived
overlay (`aiter_pa_mqa_logits.i64.py`) carries AITER's MIT header. See
`../LICENSE` and the individual file headers.
