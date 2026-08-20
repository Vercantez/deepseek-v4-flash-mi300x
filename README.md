# DeepSeek V4 Flash on a single AMD MI300X

This repository contains the configuration and patches I use to run [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) on **one AMD MI300X** in production. It includes the Docker Compose stack, SHA-256-pinned file overlays, reference diffs against upstream, and tuning tables. The checkpoint runs as shipped, without additional weight quantization or offload.

Results from the pinned stack (vLLM ROCm nightly `0.26.1rc1.dev229+g124154a88.rocm723`, AITER `0.1.19`):

| Metric | Result |
| --- | ---: |
| Historical single-stream decode (median per-stream, DSpark-7) | **168.6 tok/s** |
| Prefill with tuned kernels | **≈ 7.9–8.5K tok/s** (6,988–7,019 tok/s on fresh prompts in the shipping profile) |
| 8 concurrent streams | 542 tok/s aggregate, 90.3 tok/s median per stream |
| 64-stream burst | 830 tok/s aggregate, no OOM, no engine errors |
| Context | 256K validated (the architecture supports 1M) |
| Weights in HBM | 156.67 GiB — **no additional quantization or weight offload** |

The official vLLM recipe targets NVIDIA and newer AMD hardware. Running the model reliably on MI300X required fixes for its FP8 format, MoE routing at high concurrency, causal speculative verification, CPU-KV synchronization, and several untuned kernel shapes. This repository collects those fixes and pins the versions used in production.

---

## Why MI300X

The MI300X has **192 GB of HBM3** and 5.3 TB/s of memory bandwidth, with 2.4× the HBM capacity of an H100 SXM5 ([AMD](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)). [Doubleword's write-up](https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/) estimates that it costs roughly half as much at list price. For this 304B-parameter checkpoint, the memory capacity allows a simple single-GPU deployment:

- The entire model fits in HBM without PCIe weight streaming or layer offload.
- There is room for a 20 GB GPU KV pool and a 96 GiB CPU tier for evicted prefix-cache entries.
- One card handles 2–8 typical concurrent streams and bursts of up to 64 streams.

MI300X (CDNA3) implements the AMD/Graphcore `fnuz` variant of E4M3, while MI325X and newer use OCP-standard FP8 ([background](https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/)). A kernel that assumes OCP semantics on MI300X can be wrong by a factor of two in the scale domain. Correctness on this FP8 implementation was the first priority; performance tuning came afterward.

## Prior art, and what this repo adds

[Fergus Finn's MI300X worklog](https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/) and the accompanying [Doubleword repository](https://github.com/doublewordai/vllm-amd-blog-doubleword) identified the FP8 incompatibility, missing AITER fast paths on `gfx942`, HIP-graph hazards in sparse MLA decode, and MoE routing bugs. The [official vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) covers NVIDIA hardware and newer AMD GPUs (MI325X at 4K context and MI355X), but not a single-MI300X production configuration for the 0731 checkpoint.

This repository adds:

1. **Correctness overlays** for the pinned ROCm nightly, including fixes not yet in upstream vLLM.
2. **A validated serving configuration** using native decoding, a 2,048-token scheduler budget, and a 2,048-token long-prefill cap. DSpark is disabled in production to reduce correctness risk and scheduler overhead under concurrency.
3. **AITER GEMM tuning tables** for the recurring `gfx942` shapes the packaged tables were missing, plus a `gfx942` OGS geometry override for the MXFP4 experts.
4. **A bounded hybrid KV strategy**: 26 GB of `fp8_ds_mla` GPU cache + 96 GiB native CPU offload + a persistent filesystem tier, with load-path and DSpark prefix fixes carried as overlays. Filesystem eviction is owned by vLLM: active requests and transfers take cross-process `flock` leases on their hash shards, the tier retires only idle LRU shards, and a 2 TiB hard reserve keeps inference alive if deletion cannot keep up. External lookups are strictly optional: after a 100 ms minimum budget, the next scheduler opportunity accepts an already-completed lookup or turns a still-pending lookup into a local-prefill cache miss. Only one request may probe filesystem metadata at a time; concurrent requests compute locally. A short circuit breaker prevents unhealthy storage from becoming repeated inference backpressure.

## Repository layout

```text
.
├── compose.yaml         # The production stack (vLLM ROCm + Caddy), digest-pinned
├── compose.tp1x2.yaml   # Two independent TP1 replicas for a two-GPU host
├── Caddyfile.example    # Copy to Caddyfile; set hostname, email, and source CIDR
├── vllm-entrypoint.sh   # Removes stale CPU-KV mmaps from /dev/shm before start
├── SHA256SUMS           # SHA-256 pins for every runtime artifact
├── patches/
│   ├── *.py            # Byte-for-byte production overlays (mounted read-only)
│   ├── diffs/*.patch   # Unified diffs vs. the upstream base revision
│   └── README.md       # Provenance and regeneration instructions
├── tests/               # Concurrency and atomic-eviction tests
└── tuning/
    └── *.csv           # AITER A8W8 blockscale tuning tables for gfx942
```

## Runtime configuration

The stack uses a digest-pinned official vLLM ROCm nightly with:

- `--trust-remote-code` and the DeepSeek V4 tokenizer, reasoning, and tool parsers
- `fp8_ds_mla` KV cache (UE8M0 block-scaled FP8, not generic unscaled FP8) with 256-token blocks
- `VLLM_ROCM_USE_AITER=1` and `--moe-backend triton`; Triton OGS handles the grouped MXFP4 experts, while AITER handles attention and dense linear layers
- native, non-speculative decoding for the production worker
- constrained DeepSeek V4 DSML generation plus recovery for missing outer tool-call wrappers
- persistent filesystem KV offload behind the native 96 GiB CPU tier
- full/breakable CUDA graph capture, giving one graph launch per token during steady decode
- Caddy as an IP-allowlisted HTTPS proxy

## Deploying it

### 1. Host prerequisites

One MI300X (`gfx942`, 304 CUs, ~192 GiB HBM), a working AMD kernel driver, recent Docker Compose, ~235 GiB RAM for the CPU KV tier, and ~500 GB disk (the model cache alone is ~156 GB).

Both the single-GPU and TP2 overrides read and write the CPU/filesystem KV
cache. TP2 no longer exposes its roughly 103 GB file-backed offload mmap per
rank to ROCm DMA. Loads copy mmap data into a bounded pinned staging allocation
before CPU-to-GPU DMA; stores DMA into pinned staging before a CPU copy into the
mmap. This extra host memcpy avoids the simultaneous GPU page faults previously
seen when either rank accessed the giant mapping directly.

Filesystem metadata lookup uses four request-local workers. The 500 ms
deadline, eight-request queue, and circuit breaker remain fail-open, so a long
65,536-key probe cannot head-of-line block every other sequence in the scheduler
batch. FileMapper-generated
block paths are normalized lexically under the cache root instead of calling
`realpath()` twice per key; on the production volume that removes roughly 3.6
seconds of metadata overhead from a 16,384-key batch. Each scheduler step may
submit at most 65,536 keys; overflow is still treated as a cache miss.

### 2. Pull the pinned runtime and model

```bash
VLLM_IMAGE='vllm/vllm-openai-rocm@sha256:e68d18b2ba50298661bfc49baf01158fbf036645c2362cccf3e8a7a79fe6c69a'
MODEL='deepseek-ai/DeepSeek-V4-Flash-0731'
REVISION='7872f01b1d1fe23eabc4c98b48bffcef5a386062'

docker pull "$VLLM_IMAGE"
docker run --rm --entrypoint hf \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  "$VLLM_IMAGE" download "$MODEL" --revision "$REVISION"
```

### 3. Prepare the files

```bash
cp Caddyfile.example Caddyfile   # then set your hostname, email, and remote_ip CIDR
mkdir -p aiter-cache crash-dumps
chmod +x vllm-entrypoint.sh
sha256sum -c SHA256SUMS        # verify the overlays before first start

sudo mkdir -p /var/lib/vllm-kv
```

### 4. Start

```bash
docker compose config -q
docker compose up -d
docker compose logs -f inference
```

A healthy start takes ~5 minutes and must show model loading, GPU KV sizing,
graph capture, and application startup. It must not load the DSpark draft model:

```text
Model loading took 156.67 GiB
GPU KV cache size: ... tokens
Maximum concurrency for 262,144 tokens per request: ...x
Created mmap file /dev/shm/vllm_offload_...mmap (103.08 GB)
Capturing CUDA graphs (FULL)
Application startup complete
```

After graph capture, run `rocm-smi --showmeminfo vram`. The warmed high-water mark is ~204.5 GB of 205.8 GB. If only a few hundred MB remain, the server may start but fail on the first request.

### 5. Smoke-test

```bash
HOST='your-host.example.com'
curl -fsS "https://$HOST/v1/models"
curl -sS "https://$HOST/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"deepseek-ai/DeepSeek-V4-Flash-0731\",
       \"prompt\": \"Calculate 17 * 23. Answer with the number only.\",
       \"temperature\": 0, \"max_tokens\": 32}"
```

## The patches

Most `patches/*.py` files are **full-file overlays** mounted read-only over their counterparts in the container; `compose.yaml` contains the target paths. Small compatibility backports use fail-closed source transformers run by the entrypoint against exact blocks from the pinned image. The corresponding `diffs/*.patch` records overlay changes from their upstream bases. The base image remains digest-pinned, so upgrades require changing the image reference and revalidating the stack.

| Overlay | Mounted over | Fixes | Needed when |
| --- | --- | --- | --- |
| `gpt_oss_triton_kernels_moe.pack128-fused-silu-fast-routing.py` | `vllm/.../fused_moe/experts/gpt_oss_triton_kernels_moe.py` | MXFP4 bitmatrix padding lanes + checkpoint-exact clamped fused-SiLU grouped experts + fast DeepSeek routing | **Required** for the MXFP4 Triton path; the routing mask fix is [not yet upstream](https://github.com/doublewordai/vllm-amd-blog-doubleword/commit/c32932bb9ff6ad30b942e4835dd8b41601e7569e) and the activation must honor the checkpoint's `swiglu_limit` |
| `mxfp4.fused-silu.py` | `vllm/.../fused_moe/oracle/mxfp4.py` | Gate/up interleave layout for the fused-SiLU kernel | Required with the fused-SiLU overlay; skip both if you keep the standard SiLU path |
| `triton-kernels-matmul-ogs-opt-flags.dsv4-mi300x.py` | `vllm/third_party/triton_kernels/matmul_ogs_details/opt_flags.py` | `gfx942` MXFP4 OGS tile geometry (up to 1,536 routed rows) | **Performance** on `gfx942`; the stock geometry slows sharply above 768 routed rows |
| `fused_compress_quant_cache.fnuz-shuffle.py` | `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | **FNUZ FP8 + 16×16 preshuffle** in the Lightning Indexer cache writer | **Required on MI300X**; MI325X/MI355X use OCP FP8 and must keep the stock bytes |
| `aiter_pa_mqa_logits.i64.py` | `aiter/ops/triton/gluon/pa_mqa_logits.py` | 64-bit K/scale gathers in both non-variable-context `ChunkK=256` preshuffle pipelines | Required when KV offsets can exceed 2 GiB; includes the `KVBlockSize=256` production specialization not covered by AITER PR #4774 |
| `rocm_aiter_mla_sparse.prefill-bh64.py` | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | Upstream bounded prefill top-k selection with canonical ordering + `BLOCK_H=64` head-512 sparse prefill | Upstream row-bound handling is required for correctness; canonical ordering supports reproducible tool calls; `BLOCK_H=64` is performance |
| `rocm_aiter_mla.dspark-causal.py` | `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` | Causal multi-token speculative verification | Experiment-only; not mounted by the production profiles |
| `dspark-speculator.independent-draft-gumbel.py` + `spec-decode-utils.independent-draft-gumbel.py` | `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` + `.../spec_decode/utils.py` | Draft-proposal Gumbel noise salted away from rejection/recovery noise | Experiment-only; not mounted by the production profiles |
| `kv_offload_cpu_gpu_worker.load-war.py` | `vllm/v1/kv_offload/cpu/gpu_worker.py` | Fence CPU→GPU KV restores behind in-flight compute ([#47282](https://github.com/vllm-project/vllm/issues/47282), [PR #47291](https://github.com/vllm-project/vllm/pull/47291)); stage file-backed KV through pinned memory and avoid ROCm 7.2's fault-prone batched host DMA | Required only with `--kv-offloading-backend native` |
| `offloading-scheduler.dspark-prefix.py` + `kv_lookup_fail_open.py` | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` + sibling module | Enforce a request deadline/circuit breaker around optional cache lookup and promotion; also exclude ephemeral draft groups during experiments | Required with the native/tiered KV offload stack; historical DSpark handling is based on [#47890](https://github.com/vllm-project/vllm/issues/47890) / [#47891](https://github.com/vllm-project/vllm/pull/47891) |
| `async_lookup.bounded.py` + `tiering-fs-bounded-lru.py` + `tiering-fs-manager.disk-reserve.py` | `vllm/v1/kv_offload/tiering/async_lookup.py` + filesystem tier modules | Probe request prefixes in parallel 256-key chunks, cancel obsolete work, coordinate one eviction owner and a host-wide write gate across sibling processes, track eviction leases per shard, invalidate failed loads, retire only idle LRU shards, and reject stores before consuming the 2 TiB OS reserve | Required with the filesystem tier until upstream makes secondary storage fast, fail-open, and cancellation-aware |
| `apply-deepseek-v4-reasoning-effort.py` | Patches `vllm/tokenizers/deepseek_v4.py` + `deepseek_v4_encoding.py` at startup | Restore the model revision's native `low`/`high`/`max` prefixes and normalize OpenAI aliases (`minimal→low`, `medium→high`, `xhigh→max`) | Required with pinned vLLM `124154a88`; remove after upstream adopts the 0731 encoding revision |
| `apply-deepseek-v4-generation-prompt.py` | Patches `vllm/tokenizers/deepseek_v4.py` + `deepseek_v4_encoding.py` at startup | Honor `add_generation_prompt` and `continue_final_message`; open a fresh assistant turn after assistant-terminated histories ([vLLM #46256](https://github.com/vllm-project/vllm/issues/46256), [PR #46257](https://github.com/vllm-project/vllm/pull/46257)) | Required with pinned vLLM `124154a88`; remove after upgrading to a runtime containing the upstream fix |
| `apply-deepseek-v4-indexer-prefill-budget.py` | Patches `vllm/v1/attention/backends/mla/indexer.py` at startup | Size DeepSeek V4's sparse-indexer prefill admission budget in compressed rows, preventing uninitialized gather-buffer rows after prefix hits ([vLLM #51252](https://github.com/vllm-project/vllm/pull/51252)) | Required with pinned vLLM `124154a88`; remove after upgrading to a runtime containing the upstream fix |
| `apply-deepseek-v4-parser-recovery.py` | Patches `vllm/parser/abstract_parser.py` + `vllm/parser/engine/adapters.py` at startup | Synchronize current request tools into the reasoning parser before streaming/non-streaming extraction, allowing only a complete validated orphan invoke to cross from the reasoning pass to the tool pass | Required with pinned vLLM `124154a88`; based on the hardened design in [vLLM #52645](https://github.com/vllm-project/vllm/pull/52645) |
| `structural-tag-registry.deepseek-v4-auto.py` | `vllm/tool_parsers/structural_tag_registry.py` | Keep triggered DSML grammar enabled for `tool_choice=auto` + non-strict tools | Required for reliable OpenCode-style tool calls; based on [#40801](https://github.com/vllm-project/vllm/issues/40801) / [#46632](https://github.com/vllm-project/vllm/pull/46632) |
| `parser-*.dsml-orphan.py` + `tool-parser-utils.dsml-orphan.py` | `vllm/parser/...` + `vllm/tool_parsers/utils.py` | Recover declared DSML invokes when the model omits the outer `tool_calls` wrapper, including from prompt-opened reasoning when `</think>` is also missing; keep recovery provisional through `</invoke>`, validate every bare invoke, preserve suffix text, and count prompt-opened reasoning spans in Responses usage | Required for long-context agent sessions; backported from [#49117](https://github.com/vllm-project/vllm/pull/49117), hardened using [#52645](https://github.com/vllm-project/vllm/pull/52645), with the reasoning-count adaptation from [#49743](https://github.com/vllm-project/vllm/pull/49743) |

### Four important correctness fixes

**Checkpoint-exact SwiGLU.** DeepSeek V4 declares `swiglu_limit = 10`: clamp
the gate above at 10 and clamp the up projection to `[-10, 10]` before the
SiLU multiplication. The fused MXFP4 path passes that model value into its
Triton activation and fails closed if it is absent. An ordinary, unclamped
SwiGLU can amplify quantization outliers into stray tokens and junk output.

**MXFP4 routing.** The MoE bitmatrix kernel pads its block columns to a Triton block size, but the padding lanes were masked against the global tensor bound instead of the logical block size. Under load, padded lanes corrupted the routing matrix, causing near-match tool names and forgotten schemas on long prompts. The one-line fix is `mask = (offs_local < BLOCK_SIZE) & (offs_global < nonzero_indx_size)`, taken from [Doubleword commit `c32932bb9`](https://github.com/doublewordai/vllm-amd-blog-doubleword/commit/c32932bb9ff6ad30b942e4835dd8b41601e7569e). The overlay also includes fused-SiLU and fast-routing changes for grouped MXFP4 experts.

**FP8 format.** DeepSeek V4's Lightning Indexer cache uses FP8. The stock writer emits OCP E4M3 bytes in row-major order, while AITER on MI300X consumes AMD FNUZ E4M3 bytes in a preshuffled 16×16 tile layout. In the worst case, interpreting one format as the other produces a factor-of-two scale error. The overlay selects `float8e4b8` with `FP8_MAX=224.0` and shuffled write offsets on ROCm, while leaving the OCP path unchanged elsewhere.

**Paged-indexer addressing.** AITER's gfx942 preshuffle kernel originally
formed K and scale addresses with 32-bit `buffer_load` offsets. The shared
DeepSeek V4 cache stride crosses that range at modest physical block IDs,
silently returning zero indexer logits and making sparse top-k selection
allocation-dependent. [AITER PR #4774](https://github.com/ROCm/aiter/pull/4774)
fixed the pipeline used by smaller cache blocks; this overlay also converts the
alternate pipeline selected by production's `ChunkK=256` and
`KVBlockSize=256` to 64-bit `gl.load` pointer arithmetic.

### Speculative decoding

Production uses native decoding. DSpark flags and its three runtime overlays are
absent from both production Compose profiles. The experiment-only files remain
available for controlled benchmarks but must pass correctness and concurrent
throughput gates before they can be reintroduced.

## Performance

### Two-GPU deployment

`compose.tp1x2.yaml` runs two independent copies of the validated single-GPU
profile. Each process owns one GPU, 26 GB of GPU KV cache, 96 GiB of CPU KV,
and up to 64 active sequences. Both use the same persistent filesystem cache.
Block publication is atomic and shard leases use advisory locks in a permanent
`.locks` directory, so one process cannot evict a shard while the other is
looking it up, loading it, or writing it.

```bash
HF_CACHE=/home/hotaisle/.cache/huggingface \
  docker compose -f compose.tp1x2.yaml up -d
```

The replicas publish only on loopback at ports 8000 and 8002. Give each one a
distinct private origin path and let the service router apply session affinity;
do not hide them behind round-robin balancing because their GPU and CPU caches
remain private. Keep `PYTHONHASHSEED`, model revision, block size, KV dtype, and
offload configuration identical or they will not share disk-cache keys.
Each replica has a private 110 GB `/dev/shm`. This isolates its 96 GiB CPU-KV
mmap from the other replica, lets startup safely remove crash debris, and keeps
a failed restart from filling the peer's shared-memory filesystem.
`compose.tp2.yaml` remains available as the rollback profile.

Key optimizations in the production configuration:

| Change | Effect |
| --- | --- |
| Tune 21 recurring A8W8 GEMM shapes for 304-CU `gfx942` | +42–62% single/double-stream decode; +10–35% at 8–64 streams |
| Fused SiLU, fast DeepSeek routing, batch-sensitive expert tiles | Native C1 decode 34.5 → 56.6 tok/s (+64%); routing kernel 42.6 → 11.9 µs/layer |
| `BLOCK_H=64` sparse-prefill tile | Prefill reaches 7.9–8.5K tok/s; sparse-attention trace 317 → 142 ms per request |
| Static K=5, probabilistic + block rejection, causal verify | Production latency/acceptance tradeoff |

### Filesystem lookup latency

Filesystem metadata probes are divided into 256-key chunks and distributed
round-robin over four lookup workers. Results are published after each chunk,
and a request deadline or completed lookup cancels queued work immediately.
This avoids the previous behavior where one long request occupied one worker
until its entire metadata batch finished, even after the request had failed
open to local prefill.

Eviction leases track the cache's 4,096 hash shards rather than every pending
file path. That keeps eviction lock work proportional to shard count instead
of prompt length. The production configuration admits at most 8,192 new keys
per scheduler step; overflow is an ordinary cache miss, never scheduler
backpressure.
| 2,048-token budget + 2,048-token long-prefill cap | Larger prefill slices while retaining scheduler interleaving |
| 26 GB GPU KV + 96 GiB CPU + filesystem tier | 2.51M GPU-KV tokens plus persistent prefix spill |

### Final concurrency sweep

Distinct ~400-word prompts, streaming, `temperature=1.0, top_p=0.95`; C1–C8 at 512 output tokens, C64 at 256:

| Streams | Aggregate tok/s | Median per-stream decode | TTFT p50 |
| ---: | ---: | ---: | ---: |
| 1 | 126.2 | **168.6 tok/s** | 1.026 s |
| 2 | 145.4 | 152.7 | 0.939 s |
| 4 | 316.8 | 108.6 | 0.369 s |
| 8 | 542.3 | 90.3 | 1.027 s |
| 64 | 830.2 | 16.4 | 2.190 s |

The table above was measured with DSpark enabled and is retained as historical
evidence, not a performance claim for the current native-decoding profile.

### Prefill

With the tuned kernels, uncached prefill reaches **7.9–8.5K tok/s**, depending on scheduler budget: 7.90–7.99K at C1 with an 8,192-token budget and 8.46–8.51K at C4. The production profile uses a 2,048-token budget and a 2,048-token long-prefill cap. Earlier 1,024-token-cap testing reached 5.20–5.29K tok/s for an 8.9K-token prompt while reducing a short request's TTFT behind a 52K cold prefill from 8.2 s to 0.5 s. Warm recall of 380K cached tokens takes 0.64–2.65 s after a 120–125 s cold prefill.

## Production notes

- **HBM headroom is limited.** The warmed high-water mark is 204.5 of 205.8 GB. A 30 GB KV pool loads but fails during graph capture with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. Do not raise `--kv-cache-memory-bytes`; monitor HBM usage for growth.
- **The offload tiers store cache entries, not weights.** `--kv-offloading-size 96 --kv-offloading-backend native` maps ~103 GB in each replica's private 110 GB `/dev/shm`; the configured filesystem tier persists colder prefixes under `/var/cache/vllm-kv`. Upstream's experimental filesystem tier has no bounded secondary-tier eviction, so this stack begins in-process shard LRU eviction below 2.5 TiB free and stops at 3 TiB. Positive lookups and transfers lease their shards before eviction can begin. If every shard is busy or deletion falls behind, the 2 TiB admission reserve rejects new disk stores while inference and existing loads continue. External-cache lookup receives a 100 ms minimum availability budget and one scheduler recheck: a completed result is consumed before the deadline is evaluated, while a result still pending at that recheck becomes a local-prefill miss. Only one request may own that filesystem probe; concurrent requests compute locally. One unresolved probe opens a five-second circuit before the next probe. Metadata is bounded to eight queued scheduler batches and 16,384 new keys per step. Failed loads invalidate their positive lookup state. Never delete files externally while vLLM is running. The entrypoint removes only that replica's stale CPU mmap files after crashes.
- **Watch fail-open health, not cache hit rate alone.** Prometheus exports `vllm:kv_offload_fs_request_timeouts_total`, `vllm:kv_offload_fs_circuit_bypasses_total`, `vllm:kv_offload_fs_store_eviction_bypasses_total`, `vllm:kv_offload_fs_deferred_requests`, `vllm:kv_offload_fs_oldest_deferred_seconds`, `vllm:kv_offload_fs_circuit_open`, bounded-queue/load-failure counters, eviction state, and free bytes. Eviction is a hard write gate: existing disk entries remain readable while new stores fail open until the free-space target is restored. A timeout is a preserved request, but sustained timeouts mean the disk tier is no longer providing positive value.
- **Restarts have a 15-second drain bound.** vLLM normally waits for active streams during graceful shutdown. Compose limits that wait so a stuck generation cannot leave a replacement container in `Created`; the public Worker uses OpenRouter for interrupted or unavailable local attempts.
- **ROCm KV DMA is deliberately submitted per descriptor.** ROCm 7.2's `hipMemcpyBatchAsync` faulted both TP2 ranks while a long session was being staged from GPU KV into pinned host memory. The overlay uses stream-ordered `hipMemcpyAsync` instead. Do not re-enable the batch API without a TP2 long-session store/load stress test; filesystem persistence remains asynchronous above this transfer layer.
- **DSpark is disabled in production.** The DSpark-specific overlays remain in the repository for reproducible experiments but are not mounted by either production Compose profile.
- **Warm the kernels after restart.** The first prefill initializes kernels and takes 5.3 s for 8.9K tokens; subsequent runs take 1.7 s. Run one uncached prefill before admitting traffic.
- **Test correctness as well as throughput.** The validation suite includes two-turn tool-calling fixtures, a BFCL subset (74–76/90 exact calls), OpenCode tool-schema checks, and 380K-token needle recall on both native and DSpark paths. Cold and cached prefills can take different floating-point paths, so test both.

## License and provenance

The stack, documentation, and vLLM-derived overlays are Apache-2.0 (see `LICENSE`); the AITER-derived overlay keeps its MIT header. Upstream base revisions for every diff are recorded in [`patches/README.md`](patches/README.md). The model itself is [MIT-licensed](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

## References

All links verified 2026-08-04.

- [DeepSeek-V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) — official release; 304B parameters; fused DSpark module; recommended `temperature=1.0, top_p=0.95`; MIT license
- [Official vLLM DeepSeek V4 Flash recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) — reference launch configuration, DSpark (`num_speculative_tokens=7`), FP8 KV, block size 256, `deepseek_v4` parsers; AMD guidance for MI325X/MI355X
- [Bringing up DeepSeek-V4-Flash on AMD MI300X](https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/) (Fergus Finn, Doubleword, June 2026) — the bring-up worklog this repo builds on: FNUZ vs. OCP FP8, AITER gaps on `gfx942`, HIP-graph hazards, routing bugs
- [doublewordai/vllm-amd-blog-doubleword](https://github.com/doublewordai/vllm-amd-blog-doubleword) — demo PRs for the above, including [commit `c32932bb9`](https://github.com/doublewordai/vllm-amd-blog-doubleword/commit/c32932bb9ff6ad30b942e4835dd8b41601e7569e) ("mask MXFP4 bitmatrix padding lanes by logical block size")
- [vLLM commit `77469c9`](https://github.com/vllm-project/vllm/commit/77469c9057bec3212a64877dbbf3b9c48c22d786) — "[ROCm][MLA] Mask the AITER MLA small-head verify flatten causally (#50476)"
- [vLLM issue #47282](https://github.com/vllm-project/vllm/issues/47282) — CPU-KV load path lacks cross-stream sync with compute (WAR gap)
- [vLLM PR #47291](https://github.com/vllm-project/vllm/pull/47291) — proposed WAR fix, not merged; carried as an overlay here
- [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) — 192 GB HBM3, 5.3 TB/s peak bandwidth, 2.61 PFLOPS peak FP8
- [ROCm/AITER](https://github.com/ROCm/aiter) — AMD tuned-kernel library used for ROCm attention and dense linears
- [vLLM](https://github.com/vllm-project/vllm) — the serving runtime (ROCm nightlies under `vllm/vllm-openai-rocm`)
