"""Run inside the pinned vLLM image with both filesystem overlays mounted."""

import os
import tempfile
import time
from types import SimpleNamespace

import numpy as np

import vllm.v1.kv_offload.tiering.fs.manager as manager_module
from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.kv_offload.tiering.base import JobMetadata


class FakeMapper:
    def __init__(self, root: str) -> None:
        self.base_path = os.path.join(root, "run")
        self.rank = 0

    def get_config_file_path(self) -> str:
        return os.path.join(self.base_path, "config.json")

    def get_run_config(self) -> dict:
        return {"test": True}

    def get_file_name(self, key: str) -> str:
        return os.path.join(
            f"{self.base_path}_r0", key[:3], "00_g0", f"{key}.bin"
        )


def wait_for_jobs(tier: manager_module.FileSystemTierManager):
    tier.drain_jobs()
    results = list(tier.get_finished_jobs())
    assert len(results) == 1 and results[0].success, results


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        mapper = FakeMapper(root)
        manager_module.FileMapper.from_offloading_spec = lambda **_: mapper
        manager_module.probe_o_direct = lambda _: False
        spec = SimpleNamespace(
            blocks_per_chunk=1,
            kv_events_config=SimpleNamespace(enable_kv_cache_events=False),
        )
        backing = np.zeros((4, 4096), dtype=np.uint8)
        backing[0, :] = np.arange(4096, dtype=np.uint8)
        tier = manager_module.FileSystemTierManager(
            offloading_spec=spec,
            primary_kv_view=memoryview(backing),
            tier_type="fs",
            root_dir=root,
            n_read_threads=1,
            n_write_threads=1,
            min_free_gb=0,
            eviction_trigger_free_gb=0,
        )
        ctx = SimpleNamespace(req_id="req-1")
        tier.on_new_request(ctx)
        key = "abc-block"
        tier.submit_store(
            JobMetadata(
                job_id=1,
                keys=[key],
                block_ids=np.array([0], dtype=np.int64),
                is_promotion=False,
                req_context=ctx,
            )
        )
        wait_for_jobs(tier)
        expected = backing[0].copy()
        backing[1, :] = 0

        result = tier.lookup(key, ctx)
        deadline = time.monotonic() + 5
        while result is LookupResult.RETRY and time.monotonic() < deadline:
            tier.on_schedule_end(None)
            time.sleep(0.01)
            result = tier.lookup(key, ctx)
        assert result is LookupResult.HIT, result

        tier.submit_load(
            JobMetadata(
                job_id=2,
                keys=[key],
                block_ids=np.array([1], dtype=np.int64),
                is_promotion=True,
                req_context=ctx,
            )
        )
        wait_for_jobs(tier)
        np.testing.assert_array_equal(backing[1], expected)

        # A corrupt secondary block must become a miss after one failed load,
        # not a cached HIT that is promoted forever on every scheduler step.
        with open(mapper.get_file_name(key), "wb") as output:
            output.write(b"corrupt")
        tier.submit_load(
            JobMetadata(
                job_id=3,
                keys=[key],
                block_ids=np.array([2], dtype=np.int64),
                is_promotion=True,
                req_context=ctx,
            )
        )
        tier.drain_jobs()
        failed = list(tier.get_finished_jobs())
        assert len(failed) == 1 and not failed[0].success, failed
        assert tier.lookup(key, ctx) is LookupResult.MISS
        tier.on_request_finished(ctx)
        tier.shutdown()

    with tempfile.TemporaryDirectory() as root:
        mapper = FakeMapper(root)
        manager_module.FileMapper.from_offloading_spec = lambda **_: mapper
        manager_module.probe_o_direct = lambda _: False
        spec = SimpleNamespace(
            blocks_per_chunk=1,
            kv_events_config=SimpleNamespace(enable_kv_cache_events=False),
        )
        backing = np.zeros((2, 4096), dtype=np.uint8)
        tier = manager_module.FileSystemTierManager(
            offloading_spec=spec,
            primary_kv_view=memoryview(backing),
            tier_type="fs",
            root_dir=root,
            n_read_threads=1,
            n_write_threads=1,
            min_free_gb=0,
            eviction_trigger_free_gb=1,
            eviction_target_free_gb=2,
        )
        path_a = mapper.get_file_name("aaa-hot")
        path_b = mapper.get_file_name("bbb-cold")
        for path in (path_a, path_b):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as output:
                output.write(b"kv")
        tier._leases.record_store([path_a, path_b])
        hot_ctx = SimpleNamespace(req_id="hot")
        tier.on_new_request(hot_ctx)
        tier._leases.begin_lookup(hot_ctx.req_id, path_a)
        assert tier._leases.resolve_cached_lookup(hot_ctx.req_id, path_a, True)
        tier._get_available_bytes = lambda force=False: 0
        tier._maybe_schedule_eviction()
        deadline = time.monotonic() + 5
        while tier._eviction_worker.outstanding and time.monotonic() < deadline:
            time.sleep(0.01)
            tier._drain_evictions()
        assert os.path.exists(path_a), "leased hot shard was evicted"
        assert not os.path.exists(path_b), "idle cold shard was not evicted"

        # Eviction is a hard write gate: existing entries remain readable, but
        # new stores fail open until the free-space target is restored.
        paused_key = "ccc-paused"
        tier.submit_store(
            JobMetadata(
                job_id=4,
                keys=[paused_key],
                block_ids=np.array([0], dtype=np.int64),
                is_promotion=False,
                req_context=hot_ctx,
            )
        )
        paused = list(tier.get_finished_jobs())
        assert len(paused) == 1 and not paused[0].success, paused
        assert not os.path.exists(mapper.get_file_name(paused_key))
        tier.on_request_finished(hot_ctx)
        tier.shutdown()
        print("exact-image filesystem manager store/load/eviction smoke: PASS")


if __name__ == "__main__":
    main()
