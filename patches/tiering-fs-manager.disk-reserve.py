# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
import time
from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.bounded_lru import (
    ShardEvictionWorker,
    ShardLeaseIndex,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]

        def exists(candidate_paths: list[str]) -> Iterable[bool]:
            if _HAS_BATCH_LOOKUP_C:
                # C extension: GIL released for the entire faccessat() batch.
                return batch_lookup_C(candidate_paths)
            return (os.path.exists(path) for path in candidate_paths)

        # A positive lookup is leased to the request while holding the same
        # lock used to begin eviction. This closes the exists()->load race.
        return self._tier._leases.lookup_and_pin(
            req_context.req_id, paths, exists
        )


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
        min_free_gb: float = 0,
        space_check_interval_seconds: float = 1,
        eviction_trigger_free_gb: float = 0,
        eviction_target_free_gb: float = 0,
        shard_touch_interval_seconds: float = 30,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
            min_free_gb: Stop admitting new disk-cache writes when available
                filesystem space falls below this many GiB. Loads continue and
                rejected stores are reported as failed jobs so inference can
                continue without filling the host filesystem.
            space_check_interval_seconds: Cache filesystem space checks for at
                most this many seconds to avoid a statvfs call per KV block.
            eviction_trigger_free_gb: Begin retiring least-recently-used hash
                shards below this much free space. Zero disables eviction.
            eviction_target_free_gb: Continue eviction until this much free
                space is restored. Must be at least the trigger.
            shard_touch_interval_seconds: Minimum interval between persisting
                a shard's LRU timestamp to its directory mtime.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        if min_free_gb < 0:
            raise ValueError("min_free_gb must be non-negative")
        if space_check_interval_seconds < 0:
            raise ValueError("space_check_interval_seconds must be non-negative")
        if eviction_trigger_free_gb < 0 or eviction_target_free_gb < 0:
            raise ValueError("eviction free-space thresholds must be non-negative")
        if eviction_trigger_free_gb and (
            eviction_target_free_gb < eviction_trigger_free_gb
        ):
            raise ValueError(
                "eviction_target_free_gb must be at least eviction_trigger_free_gb"
            )
        if min_free_gb > eviction_trigger_free_gb > 0:
            raise ValueError(
                "min_free_gb must not exceed eviction_trigger_free_gb"
            )
        if shard_touch_interval_seconds < 0:
            raise ValueError("shard_touch_interval_seconds must be non-negative")
        self._root_dir = root_dir
        self._min_free_bytes = int(min_free_gb * 1024**3)
        self._eviction_trigger_bytes = int(eviction_trigger_free_gb * 1024**3)
        self._eviction_target_bytes = int(eviction_target_free_gb * 1024**3)
        self._shard_touch_interval_seconds = shard_touch_interval_seconds
        self._space_check_interval_seconds = space_check_interval_seconds
        self._last_space_check = 0.0
        self._available_bytes = 0
        self._space_available = True
        self._failed_jobs: deque[JobId] = deque()
        self._reserved_store_bytes = 0
        self._store_job_reserved_bytes: dict[JobId, int] = {}
        self._eviction_active = False
        self._last_no_evictable_log = 0.0
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        # Block files live in FileMapper's rank-specific sibling of the config
        # directory. It has at most 4,096 first-level hash shards even when it
        # contains millions of block files, so this scan stays fast.
        block_run_root = f"{self.file_mapper.base_path}_r{self.file_mapper.rank}"
        os.makedirs(block_run_root, exist_ok=True)
        self._leases = ShardLeaseIndex(block_run_root)
        self._eviction_worker = (
            ShardEvictionWorker() if self._eviction_trigger_bytes else None
        )
        if self._eviction_worker is not None:
            with os.scandir(block_run_root) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False) and entry.name.startswith(
                        ".evicting-"
                    ):
                        self._eviction_worker.submit(entry.path)
        self._store_job_paths: dict[JobId, list[str]] = {}
        self._load_job_paths: dict[JobId, list[str]] = {}

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # rather than failing every block.
        self._use_o_direct = probe_o_direct(os.path.dirname(config_path))
        if not self._use_o_direct:
            logger.warning(
                "O_DIRECT is not supported at '%s'; falling back to buffered "
                "I/O for the '%s' KV offload tier.",
                root_dir,
                tier_type,
            )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._leases.open_request(req_context.req_id)
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        path = self.file_mapper.get_file_name(key)
        self._leases.begin_lookup(req_context.req_id, path)
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return (
            LookupResult.HIT
            if self._leases.resolve_cached_lookup(
                req_context.req_id, path, result
            )
            else LookupResult.MISS
        )

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        # Positive filesystem lookups are already touched and leased inside
        # lookup_and_pin(). The scheduler broadcasts touch() to every tier for
        # all request keys, including filesystem misses; walking and stating
        # those millions of potential paths here would corrupt LRU recency and
        # add substantial scheduler overhead.
        return

    def _get_available_bytes(self, force: bool = False) -> int:
        now = time.monotonic()
        if (
            not force
            and self._last_space_check
            and now - self._last_space_check < self._space_check_interval_seconds
        ):
            return self._available_bytes
        stats = os.statvfs(self._root_dir)
        self._available_bytes = stats.f_bavail * stats.f_frsize
        self._last_space_check = now
        return self._available_bytes

    def _has_store_space(self, required_bytes: int) -> bool:
        if self._min_free_bytes == 0:
            return True
        available = self._get_available_bytes()
        was_available = self._space_available
        self._space_available = (
            available - self._reserved_store_bytes - required_bytes
            >= self._min_free_bytes
        )
        if was_available and not self._space_available:
            logger.error(
                "Pausing filesystem KV-cache stores: %d GiB available is below "
                "the configured %d GiB reserve. Cache loads and inference continue.",
                available // 1024**3,
                self._min_free_bytes // 1024**3,
            )
        elif not was_available and self._space_available:
            logger.info(
                "Resuming filesystem KV-cache stores with %d GiB available.",
                available // 1024**3,
            )
        return self._space_available

    def _drain_evictions(self) -> None:
        if self._eviction_worker is None:
            return
        for shard, success, error in self._eviction_worker.get_finished():
            self._leases.finish_eviction(shard, success)
            self._last_space_check = 0.0
            if success:
                logger.info("Retired filesystem KV-cache shard %s", shard)
            else:
                logger.error("Failed to retire KV-cache shard %s: %s", shard, error)

    def _maybe_schedule_eviction(self) -> None:
        worker = self._eviction_worker
        if worker is None:
            return
        available = self._get_available_bytes()
        if not self._eviction_active and available < self._eviction_trigger_bytes:
            self._eviction_active = True
            logger.warning(
                "Filesystem KV cache reached %d GiB free; retiring idle LRU "
                "shards until %d GiB is free.",
                available // 1024**3,
                self._eviction_target_bytes // 1024**3,
            )
        if not self._eviction_active:
            return
        if available >= self._eviction_target_bytes:
            self._eviction_active = False
            logger.info(
                "Filesystem KV-cache eviction restored %d GiB free.",
                available // 1024**3,
            )
            return
        if self._leases.evictions_in_flight:
            return
        shard = self._leases.begin_oldest_eviction()
        if shard is None:
            now = time.monotonic()
            if now - self._last_no_evictable_log >= 30:
                self._last_no_evictable_log = now
                logger.warning(
                    "KV-cache eviction is waiting because every shard is leased "
                    "by an active request or transfer."
                )
            return
        worker.submit(shard)

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        self._drain_evictions()
        self._maybe_schedule_eviction()
        paths = [self.file_mapper.get_file_name(key) for key in job_metadata.keys]
        required_bytes = len(paths) * self._block_size
        if not self._has_store_space(required_bytes):
            self._failed_jobs.append(job_metadata.job_id)
            return
        if not self._leases.pin_job(job_metadata.job_id, paths):
            self._failed_jobs.append(job_metadata.job_id)
            return
        self._store_job_paths[job_metadata.job_id] = paths
        self._store_job_reserved_bytes[job_metadata.job_id] = required_bytes
        self._reserved_store_bytes += required_bytes
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        task = functools.partial(
            batch_store_block,
            paths,
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._use_o_direct,
        )
        try:
            self._pool.enqueue_store(job_metadata.job_id, 1, [task])
        except Exception:
            self._store_job_paths.pop(job_metadata.job_id, None)
            self._store_job_reserved_bytes.pop(job_metadata.job_id, None)
            self._reserved_store_bytes -= required_bytes
            self._leases.release_job(job_metadata.job_id)
            raise

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        paths = [self.file_mapper.get_file_name(key) for key in job_metadata.keys]
        if not self._leases.pin_job(job_metadata.job_id, paths):
            self._failed_jobs.append(job_metadata.job_id)
            return
        self._load_job_paths[job_metadata.job_id] = paths
        task = functools.partial(
            batch_load_block,
            paths,
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._use_o_direct,
        )

        try:
            self._pool.enqueue_load(job_metadata.job_id, 1, [task])
        except Exception:
            self._load_job_paths.pop(job_metadata.job_id, None)
            self._leases.release_job(job_metadata.job_id)
            raise

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        self._drain_evictions()
        self._maybe_schedule_eviction()
        results = [
            JobResult(job_id=job_id, success=False) for job_id in self._failed_jobs
        ]
        self._failed_jobs.clear()
        for job_id, success in self._pool.get_finished():
            store_paths = self._store_job_paths.pop(job_id, None)
            load_paths = self._load_job_paths.pop(job_id, None)
            reserved_bytes = self._store_job_reserved_bytes.pop(job_id, 0)
            self._reserved_store_bytes -= reserved_bytes
            self._leases.release_job(job_id)
            if success and store_paths:
                self._leases.record_store(store_paths)
            if success and load_paths:
                self._leases.record_access(load_paths)
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            results.append(JobResult(job_id=job_id, success=success))
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    @override
    def has_pending_work(self) -> bool:
        return bool(
            self._eviction_worker is not None
            and self._eviction_worker.outstanding
        )

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)
        self._leases.release_request(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()
        self._leases.flush_touches(self._shard_touch_interval_seconds)
        self._drain_evictions()
        self._maybe_schedule_eviction()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
        if self._eviction_worker is not None:
            self._eviction_worker.shutdown()
